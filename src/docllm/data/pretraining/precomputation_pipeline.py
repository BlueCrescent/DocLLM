import logging
import os
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import FileLister

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.pipeline import build_docllm_datapipeline


def precompute_data(config: DocLLMPreTrainDataConfig, output_dir: str):
    config = config.model_copy()
    config.batch_size = 1
    data_gen_pipeline = build_docllm_datapipeline(config)
    for i, dat in enumerate(data_gen_pipeline):
        if len(dat[0]) == 0:
            logging.warning(f"Skipping empty data {i}...")
            continue
        torch.save(list(debatch(dat)), os.path.join(output_dir, f"eval_{i}.pt"))


def debatch(
    one_element_batch_sample: (
        Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor] | List[torch.Tensor]
    )
) -> Iterable[torch.Tensor]:
    for s in one_element_batch_sample:
        assert s.size(0) == 1
        yield s[0]


def build_precomputed_data_pipeline(input_dir: str) -> IterDataPipe:
    datapipe = FileLister(input_dir, masks="*.pt", recursive=True, non_deterministic=False)
    datapipe = datapipe.load_docllm_precomputed_data()
    return datapipe


@functional_datapipe("load_docllm_precomputed_data")
class PrecomputedDataLoaderPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
        for file_name in self._source_datapipe:
            file_data = torch.load(file_name, map_location="cpu", weights_only=True)
            if len(file_data) > 0 and len(file_data[0]) > 0:
                text_tokens, bbox_tokens, loss_mask, labels = file_data
                yield text_tokens, bbox_tokens, loss_mask, labels
            else:
                logging.warning(f"File {file_name} is empty.")
