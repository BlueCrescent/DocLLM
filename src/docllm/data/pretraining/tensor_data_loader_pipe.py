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
            file_data = torch.load(file_name, map_location="cpu")
            if len(file_data) > 0:
                text_tokens, bbox_tokens = tuple(map(list, zip(*file_data)))
                yield text_tokens, bbox_tokens
            else:
                logging.warning(f"File {file_name} is empty.")
