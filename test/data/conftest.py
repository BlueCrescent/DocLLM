from typing import Tuple

from pytest import fixture

from docllm.data.pretraining_config import DocLLMPreTrainDataConfig, NumMaskedBlocksType


@fixture(params=[1, 2, (3, 5), 0.2, (0.2, 0.4)])
def num_masked_blocks(request) -> NumMaskedBlocksType:
    return request.param


@fixture
def batch_size() -> int:
    return 1


@fixture
def num_blocks() -> int:
    return 10


@fixture
def range_block_size() -> Tuple[int, int]:
    return (2, 5)


@fixture
def max_sequence_length(
    num_masked_blocks: NumMaskedBlocksType, range_block_size: Tuple[int, int], num_blocks: int
) -> int:
    if isinstance(num_masked_blocks, tuple):
        num_masked_blocks = num_masked_blocks[1]
    if isinstance(num_masked_blocks, int):
        max_masked_blocks = num_masked_blocks
    elif isinstance(num_masked_blocks, float):
        max_masked_blocks = int(num_masked_blocks * range_block_size[1])
    return range_block_size[1] * num_blocks + (range_block_size[1] + 1) * max_masked_blocks


@fixture
def pretraining_config(
    num_masked_blocks: NumMaskedBlocksType, batch_size: int, max_sequence_length: int
) -> DocLLMPreTrainDataConfig:
    return DocLLMPreTrainDataConfig(
        batch_size=batch_size,
        max_seq_len=max_sequence_length,
        num_masked_blocks=num_masked_blocks,
        max_percentage_masked_blocks=0.8,
        mask_text_token=0,
        mask_bbox_token=(0.0, 0.0, 0.0, 0.0),
        block_start_text_token=1337,
        block_start_bbox_token=(0.0, 0.0, 0.0, 0.0),
        bos_text_token=1,
        bos_bbox_token=(0.0, 0.0, 0.0, 0.0),
        directory="",
    )
