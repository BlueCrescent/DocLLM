from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("load_docllm_tensor_data")
class TensorDataLoaderPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        for file_name in self._source_datapipe:
            yield tuple(map(list, zip(*torch.load(file_name, map_location="cpu"))))
