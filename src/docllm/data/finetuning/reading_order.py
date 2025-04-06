from typing import Annotated, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

ZeroOneFloat = Annotated[float, Field(strict=True, ge=0, le=1)]
CoordToken = ZeroOneFloat
TokenRect = Tuple[CoordToken, CoordToken, CoordToken, CoordToken]


class ReadingOrderConfig(BaseModel):
    max_seq_length: PositiveInt
    min_num_blocks_available: NonNegativeInt = 4
    min_num_tokens_available: NonNegativeInt = 1
    bos_text_token: Optional[int]
    bos_bbox_token: TokenRect = (0.0, 0.0, 0.0, 0.0)
    eos_text_token: Optional[int]
    eos_bbox_token: TokenRect = (1.0, 1.0, 1.0, 1.0)
    output_bbox_token: TokenRect = (0.0, 0.0, 0.0, 0.0)

    prompt_token_ids: List[NonNegativeInt] = Field(default_factory=list)


class ReadingOrderSampleBuilder:
    def __init__(self, config: ReadingOrderConfig) -> None:
        self._max_seq_length = config.max_seq_length
        self._min_num_blocks_available = config.min_num_blocks_available
        self._min_num_tokens_available = config.min_num_tokens_available

        self._bos_text_token = [torch.tensor([config.bos_text_token])] if config.bos_text_token is not None else []
        self._bos_bbox_token = [torch.tensor([config.bos_bbox_token])] if config.bos_text_token is not None else []
        assert get_num_tokens(self._bos_text_token) == get_num_tokens(self._bos_bbox_token)
        self._eos_text_token = [torch.tensor([config.eos_text_token])] if config.eos_text_token is not None else []
        self._eos_bbox_token = [torch.tensor([config.eos_bbox_token])] if config.eos_text_token is not None else []
        assert get_num_tokens(self._eos_text_token) == get_num_tokens(self._eos_bbox_token)
        self._prompt_token_ids = [torch.tensor([tid]) for tid in config.prompt_token_ids]
        self._prompt_bboxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]]) for _ in config.prompt_token_ids]
        assert get_num_tokens(self._prompt_token_ids) == get_num_tokens(self._prompt_bboxes)
        self._output_bbox_token = torch.tensor([config.output_bbox_token])

    def __call__(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        input_tensors = list(input_tensors) if isinstance(input_tensors, tuple) else input_tensors
        bbox_tensors = list(bbox_tensors) if isinstance(bbox_tensors, tuple) else bbox_tensors

        num_blocks = len(input_tensors)
        num_tokens = get_num_tokens(input_tensors)
        self._check_inputs(bbox_tensors, num_blocks, num_tokens)

        shuffled_ids, shuffled_bboxes = self._build_shuffled_input_sequence(input_tensors, bbox_tensors, num_blocks)
        input_ids = self._build_input_ids(input_tensors, shuffled_ids)
        bboxes = self._build_bboxes(shuffled_bboxes, num_tokens)
        labels = self._build_labels(input_tensors, shuffled_ids)
        loss_mask, num_target_tokens = self._build_loss_mask(num_tokens)

        return self._truncate_to_max_seq_len(input_ids, bboxes, loss_mask, labels, num_target_tokens)

    def _build_shuffled_input_sequence(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor], num_blocks: int
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
        permutation = torch.randperm(num_blocks)
        shuffled_inputs = [input_tensors[i] for i in permutation]
        shuffled_bboxes = [bbox_tensors[i] for i in permutation]
        return shuffled_inputs, shuffled_bboxes

    def _build_input_ids(
        self, input_tensors: List[torch.LongTensor], shuffled_ids: List[torch.LongTensor]
    ) -> torch.LongTensor:
        input_ids = torch.cat(
            self._bos_text_token
            + shuffled_ids
            + self._eos_text_token
            + self._prompt_token_ids
            + self._bos_text_token
            + input_tensors
        )
        return input_ids

    def _build_bboxes(
        self, shuffled_bboxes: List[torch.FloatTensor], num_tokens: int
    ) -> torch.FloatTensor:
        target_bbox_tokens = [self._output_bbox_token] * num_tokens
        bboxes = torch.cat(
            self._bos_bbox_token
            + shuffled_bboxes
            + self._eos_bbox_token
            + self._prompt_bboxes
            + self._bos_bbox_token
            + target_bbox_tokens
        )
        return bboxes

    def _build_labels(
        self, input_tensors: List[torch.LongTensor], shuffled_ids: List[torch.LongTensor]
    ) -> torch.LongTensor:
        labels = torch.cat(
            shuffled_ids
            + self._eos_text_token
            + self._prompt_token_ids
            + self._bos_text_token
            + input_tensors
            + self._eos_text_token
        )
        return labels

    def _build_loss_mask(self, num_tokens: int) -> Tuple[torch.BoolTensor, int]:
        zero_mask = torch.zeros(
            num_tokens + len(self._eos_text_token) + len(self._prompt_token_ids) + len(self._bos_text_token),
            dtype=torch.bool,
        )
        one_mask = torch.ones(num_tokens + len(self._eos_text_token), dtype=torch.bool)
        loss_mask = torch.cat([zero_mask, one_mask])
        num_target_tokens = one_mask.size(0)
        return loss_mask, num_target_tokens - len(self._bos_text_token)

    def _check_inputs(self, bbox_tensors: List[torch.FloatTensor], num_blocks: int, num_tokens: int) -> None:
        if num_blocks < self._min_num_blocks_available:
            raise ValueError(
                f"Number of blocks ({num_blocks}) is less than the minimum "
                f"number of blocks required ({self._min_num_blocks_available})."
            )
        if num_tokens < self._min_num_tokens_available:
            raise ValueError(
                f"Number of tokens ({num_tokens}) is less than the minimum "
                f"number of tokens required ({self._min_num_tokens_available})."
            )
        if num_blocks != len(bbox_tensors):
            raise ValueError(
                f"Number of blocks ({num_blocks}) does not match the number of bounding boxes "
                f"({len(bbox_tensors)})."
            )
        if num_tokens != get_num_tokens(bbox_tensors):
            raise ValueError(
                f"Number of tokens ({num_tokens}) does not match the number of bounding boxes "
                f"({get_num_tokens(bbox_tensors)})."
            )

    def _truncate_to_max_seq_len(
        self,
        text_inputs: torch.LongTensor,
        spatial_inputs: torch.FloatTensor,
        loss_mask: torch.BoolTensor,
        label_tokens: torch.LongTensor,
        num_target_tokens: int,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]:
        if text_inputs.size(0) <= self._max_seq_length:
            return text_inputs, spatial_inputs, loss_mask, label_tokens
        if text_inputs.size(0) - num_target_tokens + 1 >= self._max_seq_length:
            raise ValueError(
                f"Number of tokens ({text_inputs.size(0)}) exceeds maximal sequence "
                f"length ({self._max_seq_length}) with {num_target_tokens=}. "
                "Nothing would be learned. Skipping..."
            )
        return (
            text_inputs[: self._max_seq_length],
            spatial_inputs[: self._max_seq_length],
            loss_mask[: self._max_seq_length],
            label_tokens[: self._max_seq_length],
        )


def get_num_tokens(list_of_tensors: List[torch.Tensor]) -> int:
    """
    Calculate the number of tokens/entries in a list of tensors.

    Args:
        list_of_tensors: List of tensors where each entry in dimension 0 corresponds to one token/entry.

    Returns:
        Number of tokens.
    """
    return sum(tensor.shape[0] for tensor in list_of_tensors)
