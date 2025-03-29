from typing import List, Set, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate(
    items: List[Tuple[torch.Tensor, ...]], padding_value: float, ignore_indices: Set[int] = {}
) -> Tuple[torch.Tensor, ...]:
    """Turns a list of tuples of tensors into a tuple of collated tensors.
       The nth entry of each tuple in the list is collated together as the nth entry of the output tuple.

    Args:
        items (List[Tuple[torch.Tensor, ...]]): A list of tuples containing tensors to be collated.
        padding_value (float): The value to be used for all padding. Probably zero.
        ignore_indices (Set[int], optional): A set of indices to ignore. Defaults to {}.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple of collated tensors.
    """
    return tuple(
        collate_tensors(tensors, padding_value=padding_value)
        for i, tensors in enumerate(zip(*items))
        if i not in ignore_indices
    )


def collate_tensors(inputs: List[torch.Tensor], padding_value: float) -> torch.Tensor:
    return pad_sequence(inputs, batch_first=True, padding_value=padding_value)
