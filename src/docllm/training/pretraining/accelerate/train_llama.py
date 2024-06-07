import math
import os
from functools import partial
from typing import Iterable

import torch
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration
from pydantic import BaseModel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from docllm.data.pretraining.collate import collate
from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.dataset import DocLLMPretrainDataset
from docllm.data.pretraining.precomputation_pipeline import build_precomputed_data_pipeline, precompute_data
from docllm.modules.llama.causal_docllm import CausalDocLLMOutputWithPast, CausalLlamaDocLLM
from docllm.modules.llama.config import DocLLMLlamaConfig
from docllm.pretraining_loss import DocLLMCrossEntropyLoss


class LlamaDocLLMTrainerConfig(BaseModel):
    initial_model_checkpoint: str
    num_epochs: int = 1
    batch_size: int
    freeze_base_weights: bool = False
    train_data_dir: str
    eval_data_dir_in: str
    eval_data_dir_out: str
    output_dir: str
    resume_from_checkpoint: bool
    eval_steps: int

    max_lr: float = 6e-4


class Trainer:
    def __init__(
        self,
        config: LlamaDocLLMTrainerConfig,
        skip_files: Iterable[str] = [],
    ) -> None:
        self._config = config.model_copy()
        self._model_config = DocLLMLlamaConfig(
            vocab_size=32000,
            additional_training_vocab_size=3,
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=4096,
            use_cache=False,
        )
        self._model = CausalLlamaDocLLM(self._model_config)

        # TODO Init new weight

        if self._config.freeze_base_weights:
            self._model.set_freeze_llama_layers(True)

        self._build_train_dataloader(skip_files)
        self._build_eval_dataloader()

        self._loss = DocLLMCrossEntropyLoss()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._config.max_lr)  # TODO
        self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self._optimizer,
            max_lr=self._config.max_lr,
            div_factor=10.0,
            final_div_factor=1.0,
            total_steps=math.ceil(len(self._train_dataloader) / self._train_dataloader.batch_size),
            pct_start=0.01,
            anneal_strategy="cos",
        )

        project_config = ProjectConfiguration(
            project_dir=self._config.output_dir,
            automatic_checkpoint_naming=True,
            total_limit=5,
            iteration=0,  # TODO should this be adapted when loading a checkpoint?
        )

        # https://huggingface.co/docs/accelerate/usage_guides/checkpoint
        self._accelerator = Accelerator(project_config=project_config, log_with=[LoggerType.TENSORBOARD])

        config_to_log = {
            "num_samples": len(self._train_dataloader.dataset),
            "max_lr": self._config.max_lr,
            "loss_function": str(self._loss),
        }

        self._accelerator.init_trackers("DocLLM_Llama-2_7B", config=config_to_log)

        self._model, self._optimizer, self._train_dataloader = self._accelerator.prepare(
            self._model, self._optimizer, self._train_dataloader
        )

        # Register the LR scheduler
        self._accelerator.register_for_checkpointing(self._scheduler)

        if self._config.resume_from_checkpoint:
            self._accelerator.load_state(get_last_checkpoint(self._config.output_dir))
        else:
            self._accelerator.save_state()

        self._device = self._accelerator.device
        self._model.to(self._device)

    def train(self):
        num_tokens_seen = 0

        # Perform training
        for epoch in range(self._config.num_epochs):
            for step, batch in enumerate(self._train_dataloader):
                self._accelerator.log({"lr": self._scheduler.get_lr()}, step=step)
                self._optimizer.zero_grad()
                inputs, bboxes, mask, labels = batch
                num_tokens_seen += inputs.size(1)
                self._accelerator.log({"num_tokens_seen": num_tokens_seen}, step=step)
                inputs = inputs.to(self._device)
                bboxes = bboxes.to(self._device)
                mask = mask.to(self._device)
                labels = labels.to(self._device)
                outputs: CausalDocLLMOutputWithPast = self._model(input_ids=inputs, input_coordinates=bboxes)
                loss: torch.Tensor = self._loss(outputs.logits, labels, mask)
                self._accelerator.log({"train_loss": loss.item()}, step=step)
                self._accelerator.backward(loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._accelerator.log({"gradient_norm": grad_norm.item()}, step=step)
                self._optimizer.step()
                self._scheduler.step()
                self._accelerator.log({"learning_rate": self._scheduler.get_last_lr()}, step=step)
                self._checkpoint_and_evaluate(step)
            self._checkpoint_and_evaluate()

        self._accelerator.end_training()

    def _build_train_dataloader(self, skip_files: Iterable[str]):
        data_config = self._build_config(self._config.train_data_dir)
        dataset = DocLLMPretrainDataset(data_config, skip_files)
        self._train_dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=0,
            collate_fn=partial(collate, padding_value=data_config.padding_value),
            drop_last=data_config.drop_last_batch_if_not_full,
        )

    def _build_eval_dataloader(self):
        data_config = self._precompute_eval_data_for_determinism()
        data_pipeline_eval = build_precomputed_data_pipeline(self._config.eval_data_dir_out)
        self._eval_dataloader = DataLoader(
            data_pipeline_eval,
            batch_size=data_config.batch_size,
            shuffle=False,
            collate_fn=partial(collate, padding_value=data_config.padding_value),
            num_workers=0,
            drop_last=False,
        )
        # examples_iter_eval = ExamplesIterable(generate_examples_fn=iter_fct, kwargs={"pipeline": data_pipeline_eval})
        # gen = np.random.Generator(np.random.PCG64(42))
        # shuffle_config = ShufflingConfig(generator=gen, _original_seed=42)
        # eval_dataset = IterableDataset(examples_iter_eval, shuffling=shuffle_config)
        # self._eval_dataset = eval_dataset.with_format(type="torch")

    def _precompute_eval_data_for_determinism(self) -> DocLLMPreTrainDataConfig:
        data_config = self._build_config(self._config.eval_data_dir_in)
        os.makedirs(self._config.eval_data_dir_out, exist_ok=True)
        if len(os.listdir(self._config.eval_data_dir_out)) == 0:
            precompute_data(data_config, self._config.eval_data_dir_out)
        return data_config

    def _build_config(self, data_dir: str) -> DocLLMPreTrainDataConfig:
        return DocLLMPreTrainDataConfig(
            batch_size=self._config.batch_size,
            drop_last_batch_if_not_full=False,
            shuffle=True,
            use_sharding_filter=False,
            use_packing=False,
            max_seq_length=self._model_config.max_position_embeddings,
            num_masked_blocks=(0.05, 0.20),
            max_percentage_masked_blocks=0.20,
            mask_text_token=self._model_config.vocab_size + 2,
            mask_bbox_token=(0.0, 0.0, 0.0, 0.0),
            block_start_text_token=self._model_config.vocab_size + 1,
            block_end_text_token=self._model_config.vocab_size,
            bos_text_token=self._model_config.bos_token_id,
            bos_bbox_token=None,
            padding_value=0.0,
            directory=data_dir,
        )

    def _checkpoint_and_evaluate(self, step: int | None = None):
        if step is None or step % self._config.eval_steps == 0:
            self._accelerator.save_state()
            self._run_evaluation()

    def _run_evaluation(self, step: int | None = None):
        with torch.no_grad():
            self._model.eval()
            total_loss = 0.0
            for batch in self._eval_dataloader:
                inputs, bboxes, mask, labels = batch
                inputs = inputs.to(self._accelerator.device)
                bboxes = bboxes.to(self._accelerator.device)
                mask = mask.to(self._accelerator.device)
                labels = labels.to(self._accelerator.device)
                outputs: CausalDocLLMOutputWithPast = self._model(inputs, bboxes)
                loss: torch.Tensor = self._loss(outputs.logits, labels, mask)
                total_loss += loss.item()
            self._accelerator.log({"eval_loss": total_loss}, step=step)
        self._model.train()


def get_last_checkpoint(project_dir: str) -> str:
    return max(os.listdir(os.path.join(project_dir, "checkpoints")), key=lambda x: int(x.split("_")[-11]))
