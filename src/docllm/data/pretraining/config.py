from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Optional, Protocol, Tuple, runtime_checkable

import torch
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, ValidationInfo, field_validator

ZeroOneFloat = Annotated[float, Field(strict=True, ge=0, le=1)]
CoordToken = ZeroOneFloat
TokenRect = Tuple[CoordToken, CoordToken, CoordToken, CoordToken]
Interval = Tuple[PositiveInt, PositiveInt] | Tuple[ZeroOneFloat, ZeroOneFloat]
NumMaskedBlocksType = PositiveInt | ZeroOneFloat | Interval


class DocLLMPreTrainDataConfig(BaseModel):
    batch_size: PositiveInt
    drop_last_batch_if_not_full: bool = False
    shuffle: bool = True
    shuffle_buffer_size: PositiveInt = 10000
    use_sharding_filter: bool = True
    use_packing: bool = False
    max_seq_length: PositiveInt

    # This can encode either an absolute or relative count.
    # If an interval is given, the number of masked blocks
    # will be sampled uniformly from the interval each time.
    num_masked_blocks: NumMaskedBlocksType
    # Num blocks times the value will be ceiled to get the
    # maximum number of blocks that can be masked.
    max_percentage_masked_blocks: ZeroOneFloat = 0.15
    # Minimum of number of blocks that should be available
    # in the document. This is used to prevent masking
    # too many blocks in a document that is too small.
    min_num_blocks_available: NonNegativeInt = 6

    mask_text_token: int = 0
    mask_bbox_token: TokenRect = (0.0, 0.0, 0.0, 0.0)
    block_start_text_token: int
    block_end_text_token: int
    bos_text_token: Optional[int]
    bos_bbox_token: Optional[TokenRect] = None
    padding_value: float = 0.0

    directory: str | Sequence[str]

    @property
    def num_masked_blocks_callable(self) -> NumMaskedBlocks:
        return _build_num_masked_blocks_callable(self.num_masked_blocks)

    @field_validator("num_masked_blocks", mode="after")
    def check_interval_is_not_empty(cls, v: NumMaskedBlocksType) -> NumMaskedBlocksType:
        if isinstance(v, tuple) and v[0] >= v[1]:
            raise ValueError("num_masked_blocks intervals cannot be empty or negative")
        return v

    @field_validator("bos_bbox_token", mode="after")
    def set_bos_bbox_token_if_not_set(
        cls, v: Optional[TokenRect], info: ValidationInfo  # , values: Dict[str, Any]
    ) -> TokenRect:
        if v is None:
            return info.data["mask_bbox_token"]
        return v


@runtime_checkable
class NumMaskedBlocks(Protocol):
    def __call__(self, num_blocks: int) -> int: ...


def _build_num_masked_blocks_callable(num_masked_blocks: NumMaskedBlocksType) -> NumMaskedBlocks:
    if isinstance(num_masked_blocks, int):
        return lambda _: num_masked_blocks
    elif isinstance(num_masked_blocks, float):
        return lambda num_blocks: math.ceil(num_masked_blocks * num_blocks)
    elif isinstance(num_masked_blocks, Tuple) and isinstance(num_masked_blocks[0], int):

        def num_masked_blocks_callable(num_blocks: int) -> int:
            return torch.randint(num_masked_blocks[0], num_masked_blocks[1], (1,)).item()

        return num_masked_blocks_callable
    elif isinstance(num_masked_blocks, Tuple) and isinstance(num_masked_blocks[0], float):
        ival_length = num_masked_blocks[1] - num_masked_blocks[0]

        def num_masked_blocks_callable(num_blocks: int) -> int:
            return math.ceil((num_masked_blocks[0] + torch.rand(1).item() * ival_length) * num_blocks)

        return num_masked_blocks_callable
    else:
        raise ValueError(f"Invalid num_masked_blocks type: {num_masked_blocks}")
