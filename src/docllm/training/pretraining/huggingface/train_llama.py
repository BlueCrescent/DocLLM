import os
import pickle
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets.iterable_dataset import ExamplesIterable, IterableDataset, ShufflingConfig
from pydantic import BaseModel
from transformers import PreTrainedModel, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from zmq import Enum

from docllm.data.precomputed_epoch_loader import PrecomputedEpoch, collate_as_dict
from docllm.data.pretraining.build_hf_dataset import precompute_dataset
from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.pipeline import build_docllm_datapipeline
from docllm.data.pretraining.precomputation_pipeline import build_precomputed_data_pipeline, precompute_data
from docllm.modules.llama.causal_docllm import CausalLlamaDocLLM
from docllm.modules.llama.config import DocLLMLlamaConfig
from docllm.modules.llama.decoder_layer import DocLLMLlamaDecoderLayer


class DataFormat(Enum):
    PIPELINE = "pipeline"
    DATASET = "dataset"
    PRECOMPUTED = "precomputed"


class LlamaDocLLMTrainerConfig(BaseModel):
    checkpoint: str
    batch_size: int
    num_processes: int = 4
    # use_data_pipeline: bool = False
    data_format: DataFormat
    train_data_dir: str
    eval_data_dir_in: str
    eval_data_dir_out: str
    output_dir: str
    max_steps: int
    warmup_steps: Optional[int] = None
    learning_rate: float = 3e-4
    min_learning_rate: Optional[float] = None
    resume_from_checkpoint: bool
    eval_steps: int
    logging_steps: int = 25
    use_fsdp: bool = False
    use_packing: bool = False
    max_sequence_length_overwrite: Optional[int] = None
    vocab_size: int
    freeze_original_weights: bool = False


class LlamaDocLLMTrainer:
    def __init__(
        self,
        config: LlamaDocLLMTrainerConfig,
    ) -> None:
        self._config: LlamaDocLLMTrainerConfig = config.model_copy()
        self._prepare_model()
        self._prepare_data()
        training_args = self._build_training_args()
        print("Training arguments:\n", training_args)
        self._build_trainer(training_args)

    def train(self):
        ckpt = self._config.checkpoint if self._config.resume_from_checkpoint else None
        self._trainer.train(resume_from_checkpoint=ckpt)

    def _prepare_model(self):
        if self._config.resume_from_checkpoint:
            self._config.checkpoint = get_last_checkpoint(self._config.output_dir)
        self._model_config: DocLLMLlamaConfig = DocLLMLlamaConfig.from_pretrained(
            self._config.checkpoint, additional_training_vocab_size=3, use_cache=False, _attn_implementation="sdpa"
        )
        print("Model config:\n", self._model_config)
        self._model_config.save_pretrained(os.path.join(self._config.output_dir, "model_config"))
        self._model: CausalLlamaDocLLM = CausalLlamaDocLLM.from_pretrained(
            self._config.checkpoint, config=self._model_config
        )
        if self._config.freeze_original_weights:
            self._model.set_freeze_llama_layers(True)

    def _prepare_data(self):
        match self._config.data_format:
            case DataFormat.PIPELINE:
                self._build_data_pipelines()
            case DataFormat.DATASET:
                self._build_dataset()
            case DataFormat.PRECOMPUTED:
                self._build_dataset_from_precomputed()
            case _:
                raise ValueError(f"Unknown data format: {self._config.data_format}")

    def _build_dataset_from_precomputed(self):
        self._train_dataset = PrecomputedEpoch(self._config.train_data_dir, offset_idx=0)
        self._eval_dataset = PrecomputedEpoch(self._config.eval_data_dir_out, offset_idx=0)

    def _build_dataset(self):
        data_config = self._build_data_config()
        self._train_dataset = precompute_dataset(data_config).with_format(type="torch")
        data_config.directory = self._config.eval_data_dir_in
        self._eval_dataset = precompute_dataset(data_config).with_format(type="torch")

    def _build_data_pipelines(self):
        data_config = self._build_data_config()
        self._build_train_data(data_config)
        self._build_eval_data(data_config.model_copy())

    def _build_data_config(self) -> DocLLMPreTrainDataConfig:
        return DocLLMPreTrainDataConfig(
            batch_size=1,
            drop_last_batch_if_not_full=False,
            shuffle=False,
            use_sharding_filter=False,
            use_packing=self._config.use_packing,
            max_seq_length=self._config.max_sequence_length_overwrite or self._model_config.max_position_embeddings,
            num_masked_blocks=(0.05, 0.25),
            max_percentage_masked_blocks=0.25,
            min_num_blocks_available=8,
            mask_text_token=self._config.vocab_size + 2,
            mask_bbox_token=(0.0, 0.0, 0.0, 0.0),
            block_start_text_token=self._config.vocab_size,
            block_end_text_token=self._config.vocab_size + 1,
            bos_text_token=self._model_config.bos_token_id,
            bos_bbox_token=None,
            padding_value=0.0,
            directory=self._config.train_data_dir,
        )

    def _build_train_data(self, data_config: DocLLMPreTrainDataConfig):
        data_pipeline = build_docllm_datapipeline(data_config)
        examples_iter = ExamplesIterable(generate_examples_fn=iter_fct, kwargs={"pipeline": data_pipeline})
        train_dataset = IterableDataset(examples_iter)
        self._train_dataset = train_dataset.with_format(type="torch")

    def _build_eval_data(self, data_config_eval: DocLLMPreTrainDataConfig):
        self._precompute_eval_data_for_determinism(data_config_eval)
        data_pipeline_eval = build_precomputed_data_pipeline(self._config.eval_data_dir_out)
        examples_iter_eval = ExamplesIterable(generate_examples_fn=iter_fct, kwargs={"pipeline": data_pipeline_eval})
        gen = np.random.Generator(np.random.PCG64(42))
        shuffle_config = ShufflingConfig(generator=gen, _original_seed=42)
        eval_dataset = IterableDataset(examples_iter_eval, shuffling=shuffle_config)
        self._eval_dataset = eval_dataset.with_format(type="torch")

    def _precompute_eval_data_for_determinism(self, data_config_eval: DocLLMPreTrainDataConfig):
        data_config_eval.directory = self._config.eval_data_dir_in
        os.makedirs(self._config.eval_data_dir_out, exist_ok=True)
        if len(os.listdir(self._config.eval_data_dir_out)) == 0:
            precompute_data(data_config_eval, self._config.eval_data_dir_out)

    def _build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            run_name="llama_docllm_pretraining_epoch01a",
            do_train=True,
            bf16=True,
            output_dir=self._config.output_dir,
            resume_from_checkpoint=self._config.resume_from_checkpoint,
            # overwrite_output_dir=False,
            # max_steps=self._config.max_steps,
            max_steps=len(self._train_dataset) // (self._config.num_processes * self._config.batch_size),
            # num_train_epochs=1,
            per_device_train_batch_size=self._config.batch_size,
            auto_find_batch_size=True,  # TODO
            # gradient_checkpointing=True,  # TODO
            learning_rate=self._config.learning_rate,
            optim="adamw_torch",
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine" if self._config.min_learning_rate is None else "cosine_with_min_lr",
            lr_scheduler_kwargs=(
                {} if self._config.min_learning_rate is None else {"min_lr": self._config.min_learning_rate}
            ),
            warmup_steps=self._config.warmup_steps or 100,  # 5000 // (4 * self._config.batch_size),
            logging_steps=self._config.logging_steps,
            save_steps=self._config.eval_steps,
            eval_strategy="steps",
            eval_steps=self._config.eval_steps,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels", "loss_mask"],
            seed=42,
            data_seed=None,  # 42,  FIXME
            include_tokens_per_second=True,
            include_num_input_tokens_seen=False,  # FIXME: reactivate when this is merged: https://github.com/huggingface/transformers/pull/34554
            report_to=["tensorboard"],
            fsdp="full_shard" if self._config.use_fsdp else "",  # full_shard offload auto_wrap",  # FIXME
            # fsdp_min_num_params=int(1e2),
            fsdp_config=(
                {
                    # "min_num_params": int(1e3),
                    "transformer_layer_cls_to_wrap": [DocLLMLlamaDecoderLayer.__name__],
                    # "backward_prefetch": "backward_pre",
                    # "forward_prefetch": False,
                    # "limit_all_gathers": False,
                }
                if self._config.use_fsdp
                else {}
            ),
        )

    def _build_trainer(self, training_args: TrainingArguments):
        collator = (
            partial(collate_as_dict, padding_value=0.0, additional_col=True)
            if self._config.data_format == DataFormat.PRECOMPUTED
            else collate_data
        )
        self._trainer = Trainer(
            self._model,
            training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=collator,
            callbacks=[SkipNaNGradientsCallback()],
        )


def iter_fct(pipeline) -> Iterable[Tuple[int, Dict[str, torch.Tensor]]]:
    for i, (input_ids, bboxes, loss_mask, labels) in enumerate(pipeline):
        yield i, {
            "input_ids": input_ids,
            "input_coordinates": bboxes,
            "loss_mask": loss_mask,
            "labels": labels,
        }


def collate_data(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    debatch = (lambda x: x[0]) if len(data[0]["input_ids"].shape) == 2 else lambda x: x
    inputs = [debatch(d["input_ids"]) for d in data]
    input_batch = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    bboxes = [debatch(d["input_coordinates"]) for d in data]
    bbox_batch = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True, padding_value=0.0)
    mask = [debatch(d["loss_mask"]) for d in data]
    mask_batch = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=False)
    labels = [debatch(d["labels"]) for d in data]
    labels_batch = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return {
        "input_ids": input_batch,
        "input_coordinates": bbox_batch,
        "loss_mask": mask_batch,
        "labels": labels_batch,
    }


# class CustomTrainer(Trainer):
#     def on_step(self, args, state, control, **kwargs):
#         # Compute the gradient norm
#         grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e6, norm_type=2)

#         # Check for NaN in the gradient norm
#         if grad_norm.isnan():
#             # Retrieve the current batch
#             dataloader = self.get_train_dataloader()
#             current_batch = next(iter(dataloader))

#             # Log information
#             print(f"Gradient norm is NaN at step {state.global_step}")
#             print(f"Batch index: {state.global_step}")
#             print(f"Batch content: {current_batch}")

#             # Assuming you have access to the dataset indices
#             batch_indices = current_batch["input_ids"]  # or the specific key used in your dataset
#             samples = [self.train_dataset[i] for i in batch_indices]
#             print(f"Batch samples: {samples}")

#         super().on_step(args, state, control, **kwargs)


# class GradientNormCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         # _ = logs.pop("total_flos", None)
#         if state.is_local_process_zero:
#             # Get the logged gradient norm and loss
#             grad_norm = logs.get("gradient_norm", None)
#             loss = logs.get("loss", None)

#             # Check if gradient norm is NaN
#             if grad_norm is not None and torch.isnan(torch.tensor(grad_norm)):
#                 print(logs)
#                 # Retrieve the current batch

#                 # Log pertinent information
#                 print(f"Gradient norm is NaN at step {state.global_step}")
#                 print(f"Loss: {loss}")
#                 print(f"Batch index: {state.global_step}")
#                 if dataloader := kwargs.get("train_dataloader", None):
#                     current_batch = next(iter(dataloader))
#                     print(f"Batch content: {current_batch}")

#                 # Assuming you have access to the dataset indices
#                 # batch_indices = current_batch['input_ids']  # or the specific key used in your dataset
#                 # samples = [self.train_dataset[i] for i in batch_indices]
#                 # print(f"Batch samples: {samples}")
#             # print(logs)


class SkipNaNGradientsCallback(TrainerCallback):
    # def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
    #     local_rank = int(os.environ.get("LOCAL_RANK", -1))
    #     print(f">>>>>({local_rank}) Step: {state.global_step}", flush=True)

    def on_pre_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel | torch.nn.Module | None = kwargs.get("model", None)
        if model is None:
            print("Model not available in on_pre_optimizer_step.")
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                if torch.isnan(grad_norm):
                    local_rank = int(os.environ.get("LOCAL_RANK", -1))
                    print(
                        f">>>>>({local_rank}) NaN gradient detected at "
                        f"{state.global_step} in {name}. Zeroing all gradients...",
                        flush=True,
                    )
                    optimizer: torch.optim.Optimizer | None = kwargs.get("optimizer", None)
                    if optimizer is not None:
                        print(f">>>>>({local_rank}) Zeroing gradients for optimizer.", flush=True)
                        optimizer.zero_grad()
                    else:
                        self.zero_grads(model, local_rank)
                    break

    def zero_grads(self, model: torch.nn.Module, local_rank: int = -1):
        """
        Zero out all gradients in the model.
        """
        print(f">>>>>({local_rank}) Zeroing gradients for all model parameters.", flush=True)
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()


class NaNGradientDetectionCallback(TrainerCallback):
    def __init__(self, log_file="", output_dir=""):
        self.log_file = log_file
        self.output_dir = output_dir
        self.batch = None
        self.batch_index = None
        self.loss = None
        self.stop_training = False
        self.substep_counter = 0

    def on_train_begin(self, args, state, control, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"Local Rank: {local_rank}")
        if local_rank == 0:
            world_size = int(os.environ.get("WORLD_SIZE", -1))
            print(f"World Size: {world_size}")

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.batch_index = state.global_step
        # self.batch = kwargs.get("batch")
        print(f"Batch index: {self.batch_index}", flush=True)
        print(f"Substep counter: {self.substep_counter}", flush=True)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"Local Rank: {local_rank}")
        self.substep_counter = 0

    def on_substep_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.substep_counter += 1

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.loss = kwargs.get("loss")

    def on_pre_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        # optimizer = kwargs.get("optimizer", None)
        # if not state.is_local_process_zero:
        #     return
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if model is None:
            print(f"\n\n>>>>>({local_rank}) Model not available in on_pre_optimizer_step.", flush=True)
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                if torch.isnan(grad_norm):
                    print(f"\n\n>>>>>on_pre_optimizer_step({local_rank})<<<<<\n\n")
                    # with open(self.log_file, "a") as f:
                    print(f"NaN gradient detected at batch index {self.batch_index}\n", flush=True)
                    print(f"Loss: {self.loss}\n", flush=True)
                    # print(f"Batch content: {self.batch}\n", flush=True)
                    # Sauvegarde le batch en tant que fichier pickle
                    # try:
                    #     torch.save(self.batch, f"debug_batch_{self.batch_index}.pt")
                    # except Exception as e:
                    #     print(f"Error saving batch: {e}", flush=True)
                    # with open(f"debug_batch_{self.batch_index}.pkl", "wb") as f:
                    #     pickle.dump(self.batch, f)
                    # Sauvegarde un checkpoint du modèle et de l'optimiseur
                    # output_dir = f"checkpoint_nan_grad_batch_{self.batch_index}"
                    # model.save_pretrained(output_dir)
                    # torch.save(optimizer.state_dict(), f"{output_dir}/optimizer.pt")
                    # Termine l'entraînement
                    # control.should_training_stop = True
                    self.stop_training = True
                    break

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # _ = logs.pop("total_flos", None)
        # if state.is_local_process_zero:
        # Get the logged gradient norm and loss
        grad_norm = logs.get("gradient_norm", None)
        loss = logs.get("loss", None)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # Check if gradient norm is NaN
        grad_is_nan = grad_norm is not None and torch.isnan(torch.tensor(grad_norm))
        if grad_is_nan or self.stop_training:
            print(f"\n\n>>>>>on_log({local_rank})<<<<<\n\n")
            print(f"({local_rank}){grad_is_nan=}")
            print(f"logs({local_rank}):\n")
            print(logs, flush=True)
            print(f"\n\nstate({local_rank}):\n")
            print(state, flush=True)
            print("\n\n")
            # Retrieve the current batch

            # Log pertinent information
            print(f"({local_rank})Gradient norm is NaN at step {state.global_step}")
            print(f"({local_rank})Loss: {loss}")
            print(f"({local_rank})Batch index: {state.global_step}", flush=True)
            if dataloader := kwargs.get("train_dataloader", None):
                current_batch = next(iter(dataloader))
                print(f"({local_rank})Batch content: {current_batch}", flush=True)
                try:
                    torch.save(current_batch, f"debug{local_rank}_batch_{self.batch_index}.pt")
                except Exception as e:
                    print(f"({local_rank})Error saving batch: {e}", flush=True)
            control.should_training_stop = True
