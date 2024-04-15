import logging
import math
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.masker import DocLLMPretrainingMasker


@functional_datapipe("build_docllm_train_data")
class DocLLMTrainDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, config: DocLLMPreTrainDataConfig) -> None:
        self._masker = DocLLMPretrainingMasker(config)
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]]:
        for input_tensors, bbox_tensors in self._source_datapipe:
            yield from self._try_build_inputs(input_tensors, bbox_tensors)

    def _try_build_inputs(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor]
    ) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
        try:
            yield self._masker(input_tensors, bbox_tensors)
        except ValueError as e:
            logging.warning(e)
