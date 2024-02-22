from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("pack_data")
class DataPackingPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, max_sequence_length: int) -> None:
        self._source_datapipe = source_datapipe
        self._max_sequence_length = max_sequence_length

    def __iter__(self) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
        elements = []
        total_size = 0

        for item in self._source_datapipe:
            item_size = item[0].size(0)
            if total_size + item_size > self._max_sequence_length:
                inputs, bboxes, mask, labels = zip(*elements)
                yield torch.cat(inputs), torch.cat(bboxes), torch.cat(mask), torch.cat(labels)
                elements = []
                total_size = 0
            elements.append(item)
            total_size += item_size

        if elements:
            inputs, bboxes, mask, labels = zip(*elements)
            yield torch.cat(inputs), torch.cat(bboxes), torch.cat(mask), torch.cat(labels)
