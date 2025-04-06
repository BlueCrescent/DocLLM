import os
from functools import partial
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from docllm.data.pretraining.collate import collate

PackedData = Tuple[
    torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor, torch.LongTensor
]


def build_epoch_dataloader(directory: str, batch_size: int, padding_value: float, offset_idx: int = 0) -> DataLoader:
    dataset = PrecomputedEpoch(directory, offset_idx=offset_idx)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=partial(collate_with_packing, padding_value=padding_value)
    )


class PrecomputedEpoch(Dataset):
    def __init__(self, directory: str, offset_idx: int = 0):
        self._directory = directory
        self._file_names = [f for f in os.listdir(directory) if f.endswith(".pt")]
        self._offset_idx = offset_idx

    def __len__(self):
        return len(self._file_names) + self._offset_idx

    def __getitem__(self, idx: int) -> PackedData:
        idx -= self._offset_idx
        if idx < 0:
            raise IndexError(f"Index {idx + self._offset_idx} is bellow given offset.")
        file_path = os.path.join(self._directory, self._file_names[idx])
        input_ids, bbox, loss_mask, labels, sizes = torch.load(file_path)

        attention_mask, position_ids = compute_packing_attention_mask_and_pos_ids(sizes)

        return input_ids, bbox, loss_mask, labels, position_ids, attention_mask


def compute_packing_attention_mask_and_pos_ids(sizes: torch.LongTensor) -> Tuple[torch.BoolTensor, torch.LongTensor]:
    total_len = sizes.sum()
    mask = torch.zeros((total_len, total_len), dtype=torch.bool)
    pos_ids = torch.zeros(total_len, dtype=torch.long)
    start = 0
    for size in sizes:
        end = start + size
        mask[start:end, start:end] = torch.tril(torch.ones((size, size), dtype=torch.bool))
        pos_ids[start:end] = torch.arange(size)
        start = end
    return mask, pos_ids


def collate_as_dict(
    inputs: List[PackedData], padding_value: float, additional_col: bool = False
) -> Dict[str, torch.Tensor]:
    input_ids, bbox, loss_mask, labels, position_ids, attention_mask = collate_with_packing(
        inputs, padding_value, additional_col
    )
    return {
        "input_ids": input_ids,
        "input_coordinates": bbox.clamp(min=0.0, max=1.0),  # FIXME
        "loss_mask": loss_mask,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def collate_with_packing(inputs: List[PackedData], padding_value: float, additional_col: bool = False) -> PackedData:
    att_mask_idx = 5
    res = collate(inputs, padding_value, ignore_indices={att_mask_idx})
    input_ids, bbox, loss_mask, labels, position_ids = res
    max_length = input_ids.size(1)
    padded_attention_masks = torch.stack(
        [_pad_attention_mask(t[att_mask_idx], max_length, additional_col) for t in inputs]
    )
    assert padded_attention_masks.size(1) == max_length
    assert padded_attention_masks.size(2) == max_length + 1
    return input_ids, bbox, loss_mask, labels, position_ids, padded_attention_masks


def _pad_attention_mask(mask: torch.BoolTensor, to_length: int, additional_col: bool) -> torch.BoolTensor:
    # if mask.size(0) == to_length:
    #     return mask
    return torch.cat(
        [
            torch.cat([mask, torch.zeros(to_length - mask.size(0), mask.size(1), device=mask.device)], dim=0),
            torch.zeros(to_length, to_length - mask.size(1) + int(additional_col), device=mask.device),
        ],
        dim=1,
    )


# attention_mask = torch.cat([attention_mask, torch.zeros(2, attention_mask.size(1), 1, dtype=torch.bool)], dim=2)
