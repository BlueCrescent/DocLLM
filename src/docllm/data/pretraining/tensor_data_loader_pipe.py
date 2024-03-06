import logging
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("load_docllm_tensor_data")
class TensorDataLoaderPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        for file_name in self._source_datapipe:
            try:
                yield self._try_parse_file(file_name)
            except Exception as e:
                logging.warning(f"Error while parsing file {file_name}: {e}")

    def _try_parse_file(self, file_name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        file_data = torch.load(file_name, map_location="cpu")
        if len(file_data) == 0:
            raise ValueError(f"File is empty.")
        text_tokens, bbox_tokens = tuple(map(list, zip(*file_data)))
        self._check_data(text_tokens, bbox_tokens)
        return text_tokens, bbox_tokens

    def _check_data(self, text_tokens: List[torch.Tensor], bbox_tokens: List[torch.Tensor]) -> None:
        if len(text_tokens) != len(bbox_tokens):
            raise ValueError(
                f"Length of text tokens ({len(text_tokens)}) and bbox tokens ({len(bbox_tokens)}) must match."
            )
        if not all(bbox_tokens[i].size() == (text_tokens[i].size()[0], 4) for i in range(len(text_tokens))):
            raise ValueError("Unexpected bounding box shape.")
