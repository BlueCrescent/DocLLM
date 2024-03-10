import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets.iterable_dataset import ExamplesIterable, IterableDataset, ShufflingConfig
from pydantic import BaseModel
from transformers import Trainer, TrainingArguments

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.pipeline import build_docllm_datapipeline
from docllm.data.pretraining.precomputation_pipeline import build_precomputed_data_pipeline, precompute_data
from docllm.modules.llama.causal_docllm import CausalLlamaDocLLM
from docllm.modules.llama.config import DocLLMLlamaConfig


class LlamaDocLLMTrainerConfig(BaseModel):
    model_checkpoint: str
    batch_size: int
    train_data_dir: str
    eval_data_dir_in: str
    eval_data_dir_out: str
    output_dir: str
    max_steps: int
    resume_from_checkpoint: bool
    eval_steps: int


class LlamaDocLLMTrainer:
    def __init__(
        self,
        config: LlamaDocLLMTrainerConfig,
    ) -> None:
        self._config = config
        self._model_config = DocLLMLlamaConfig.from_pretrained(
            self._config.model_checkpoint, additional_training_vocab_size=3
        )
        self._model = CausalLlamaDocLLM.from_pretrained(self._config.model_checkpoint, config=self._model_config)
        self._build_data_pipelines()
        training_args = self._build_training_args()
        self._build_trainer(training_args)

    def train(self):
        self._trainer.train()

    def _build_data_pipelines(self):
        data_config = DocLLMPreTrainDataConfig(
            batch_size=1,
            drop_last_batch_if_not_full=False,
            shuffle=False,
            use_sharding_filter=False,
            use_packing=True,
            max_seq_length=self._model_config.max_position_embeddings,
            num_masked_blocks=(0.05, 0.25),
            max_percentage_masked_blocks=0.25,
            mask_text_token=32002,  # TODO
            mask_bbox_token=(0.0, 0.0, 0.0, 0.0),
            block_start_text_token=32001,  # TODO
            block_end_text_token=32000,  # TODO
            bos_text_token=self._model_config.bos_token_id,
            bos_bbox_token=None,
            padding_value=0.0,
            directory=self._config.train_data_dir,
        )
        self._build_train_data(data_config)
        self._build_eval_data(data_config.model_copy())

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
            run_name="llama_docllm_pretraining",
            do_train=True,
            output_dir=self._config.output_dir,
            resume_from_checkpoint=self._config.resume_from_checkpoint,
            # overwrite_output_dir=False,
            max_steps=self._config.max_steps,
            per_device_train_batch_size=self._config.batch_size,
            learning_rate=3e-4,
            optim="adamw_torch",
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            lr_scheduler_kwargs={},
            warmup_steps=1000,
            logging_steps=self._config.eval_steps,
            save_steps=self._config.eval_steps,
            evaluation_strategy="steps",
            eval_steps=self._config.eval_steps,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels", "loss_mask"],
            seed=42,
            data_seed=42,
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
        )

    def _build_trainer(self, training_args: TrainingArguments):
        self._trainer = Trainer(
            self._model,
            training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=collate_data,
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
    inputs = [d["input_ids"][0] for d in data]
    input_batch = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    bboxes = [d["input_coordinates"][0] for d in data]
    bbox_batch = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True, padding_value=0.0)
    mask = [d["loss_mask"][0] for d in data]
    mask_batch = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=False)
    labels = [d["labels"][0] for d in data]
    labels_batch = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return {
        "input_ids": input_batch,
        "input_coordinates": bbox_batch,
        "loss_mask": mask_batch,
        "labels": labels_batch,
    }
