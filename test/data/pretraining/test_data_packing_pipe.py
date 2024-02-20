import random
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import IterDataPipe

from docllm.data.pretraining.data_packing_pipe import DataPackingPipe


@pytest.fixture
def test_data(
    num_docs: int,
    range_block_size: Tuple[int, int],
) -> List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
    max_token = 1111
    return [
        [
            (
                torch.randint(max_token, (b := random.randint(range_block_size[0], range_block_size[1]),)),
                torch.rand(b, 4),
                torch.ones(b, dtype=torch.bool),
                torch.randint(max_token, (b,)),
            )
        ]
        for _ in range(num_docs)
    ]


@pytest.fixture
def min_test_data_sequence_length(
    test_data: List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]
) -> int:
    return min(min(t.size(0) for t, _, _, _ in doc_data) for doc_data in test_data)


@pytest.fixture
def source_datapipe(
    test_data: List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]
) -> IterDataPipe:
    pipe = MagicMock(spec=IterDataPipe)
    pipe.__iter__.return_value = test_data
    return pipe


@pytest.fixture
def data_packing_pipe(
    source_datapipe: IterDataPipe,
    max_sequence_length: int,
) -> DataPackingPipe:
    return DataPackingPipe(source_datapipe, max_sequence_length)


@pytest.mark.parametrize("max_sequence_length", (80, 150, 1000), indirect=True)
def test_instantiation(data_packing_pipe: DataPackingPipe):
    assert isinstance(data_packing_pipe, DataPackingPipe)


@pytest.mark.parametrize("max_sequence_length", (10000,), indirect=True)
def test_datapipe_produces_single_output_for_long_sequence_length(data_packing_pipe: DataPackingPipe):
    assert len(list(data_packing_pipe)) == 1


@pytest.mark.parametrize("max_sequence_length", (80, 150), indirect=True)
def test_output_tensors_are_at_most_max_sequence_length_long(
    data_packing_pipe: DataPackingPipe, max_sequence_length: int
):
    assert all(t.size(0) <= max_sequence_length for results in data_packing_pipe for t in results)


@pytest.mark.parametrize("max_sequence_length", (80, 150), indirect=True)
def test_output_tensors_are_longer_than_min_test_sequence_length_long(
    data_packing_pipe: DataPackingPipe, min_test_data_sequence_length: int
):
    assert all(min_test_data_sequence_length < t.size(0) for results in data_packing_pipe for t in results)
