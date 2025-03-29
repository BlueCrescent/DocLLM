import logging
import os
from functools import partial
from typing import Dict, List, Tuple

import torch
from datasets import Array2D, Dataset, Features, Sequence, Value

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.precomputation_pipeline import generate_precomputed_data


def precompute_dataset(config: DocLLMPreTrainDataConfig) -> Dataset:
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "input_coordinates": Array2D(dtype="int64", shape=(config.max_seq_length, 4)),
            "loss_mask": Sequence(feature=Value(dtype="int64")),
            "labels": Sequence(feature=Value(dtype="int64")),
        }
    )
    return Dataset.from_generator(
        lambda: map(to_model_input, generate_precomputed_data(config=config)),
        # features=features # TODO: Can this be used?
    )


def to_model_input(
    data: Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor] | List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {"input_ids": data[0], "input_coordinates": data[1], "loss_mask": data[2], "labels": data[3]}


def prepare_docllm_data(dirs: List[str], max_sequence_length: int):
    for dir in dirs:
        for file_name in os.listdir(dir):
            file_data = torch.load(file_name, map_location="cpu", weights_only=True)
            if len(file_data) > 0 and len(file_data[0]) > 0 and len(file_data[0][-1]) < max_sequence_length:
                text_tokens, bbox_tokens, loss_mask, labels = file_data
                yield {"input_ids": text_tokens, "attention_mask": loss_mask, "bbox": bbox_tokens, "labels": labels}
            else:
                logging.warning(
                    f"File {file_name} is not valid. ({len(file_data)=}, {len(file_data[0])=}, {len(file_data[0][-1])=})"
                )
