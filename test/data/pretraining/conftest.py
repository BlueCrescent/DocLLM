import math
import random
from tempfile import TemporaryDirectory
from typing import Iterable, List, Tuple

import pytest
import torch

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig, NumMaskedBlocksType


@pytest.fixture
def num_docs() -> int:
    return 10


@pytest.fixture
def multiple_docs_test_data(num_docs: int) -> List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]:
    docs = []
    for _ in range(num_docs):
        num_blocks = random.randint(4, 11)
        data = [
            (
                torch.randint(1, 10, (block_len := random.randint(1, 7),), dtype=torch.long),
                torch.rand(block_len, 4),
            )
            for _ in range(num_blocks)
        ]
        docs.append(data)
    return docs


@pytest.fixture
def input_dir_with_data(
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
) -> Iterable[str]:
    with TemporaryDirectory() as input_dir:
        for i, data in enumerate(multiple_docs_test_data):
            torch.save(data, f"{input_dir}/doc_{i}.pt")
        yield input_dir


@pytest.fixture
def num_blocks() -> int:
    return 10


@pytest.fixture
def batch_size(request) -> int:
    if not hasattr(request, "param") or request.param is None:
        return 1
    return request.param


@pytest.fixture
def range_block_size() -> Tuple[int, int]:
    return (2, 5)


@pytest.fixture
def max_sequence_length(
    request, num_masked_blocks: NumMaskedBlocksType, range_block_size: Tuple[int, int], num_blocks: int
) -> int:
    if hasattr(request, "param") and request.param and isinstance(request.param, int):
        return request.param
    if isinstance(num_masked_blocks, tuple):
        num_masked_blocks = num_masked_blocks[1]
    if isinstance(num_masked_blocks, int):
        max_masked_blocks = num_masked_blocks
    elif isinstance(num_masked_blocks, float):
        max_masked_blocks = math.ceil(num_masked_blocks * range_block_size[1])
    return range_block_size[1] * num_blocks + (range_block_size[1] + 1) * max_masked_blocks


@pytest.fixture(params=[1, 2, (3, 5), 0.2, (0.2, 0.4)])
def num_masked_blocks(request) -> NumMaskedBlocksType:
    return request.param


@pytest.fixture
def max_percentage_masked_blocks(request) -> float:
    if not hasattr(request, "param") or request.param is None:
        return 0.8
    return request.param


@pytest.fixture
def use_sharding_filter(request) -> float:
    if not hasattr(request, "param") or request.param is None:
        return False
    return request.param


@pytest.fixture
def pretraining_config(
    batch_size: int,
    use_sharding_filter: bool,
    max_sequence_length: int,
    num_masked_blocks: NumMaskedBlocksType,
    max_percentage_masked_blocks: float,
) -> DocLLMPreTrainDataConfig:
    return DocLLMPreTrainDataConfig(
        batch_size=batch_size,
        use_sharding_filter=use_sharding_filter,
        max_seq_length=max_sequence_length,
        num_masked_blocks=num_masked_blocks,
        max_percentage_masked_blocks=max_percentage_masked_blocks,
        mask_text_token=0,
        mask_bbox_token=(0.0, 0.0, 0.0, 0.0),
        block_start_text_token=1337,
        block_start_bbox_token=(0.0, 0.0, 0.0, 0.0),
        block_end_text_token=1338,
        bos_text_token=1,
        bos_bbox_token=(0.0, 0.0, 0.0, 0.0),
        directory="",
    )


@pytest.fixture
def pretraining_config_with_input_dir(
    pretraining_config: DocLLMPreTrainDataConfig, input_dir_with_data: str
) -> DocLLMPreTrainDataConfig:
    pretraining_config = pretraining_config.model_copy()
    pretraining_config.directory = input_dir_with_data
    return pretraining_config
