from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate(
    items: List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]],
    padding_value: float,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]:
    inputs, bboxes, mask, labels = zip(*items)
    input_batch = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    bbox_batch = pad_sequence(bboxes, batch_first=True, padding_value=padding_value)
    mask_batch = pad_sequence(mask, batch_first=True, padding_value=padding_value)
    labels_batch = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    return input_batch, bbox_batch, mask_batch, labels_batch
