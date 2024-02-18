from typing import Iterable, List, Tuple

import torch
import torchdata
from torchdata.datapipes.iter import IterDataPipe


@torchdata.datapipes.functional_datapipe("load_docllm_tensor_data")
class TensorDataLoaderPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterable[List[Tuple[torch.Tensor, torch.Tensor]]]:
        for file_name in self._source_datapipe:
            yield torch.load(file_name, map_location="cpu")
