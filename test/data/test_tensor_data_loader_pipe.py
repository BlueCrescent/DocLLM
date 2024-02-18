import glob
import os
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.datapipes.iter import IterDataPipe

from docllm.data.tensor_data_loader_pipe import TensorDataLoaderPipe


@pytest.fixture
def files_datapipe(input_dir_with_data: str) -> IterDataPipe:
    pipe = MagicMock(spec=IterDataPipe)
    pipe.__iter__.return_value = sorted(
        glob.glob(os.path.join(input_dir_with_data, "*.pt")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return pipe


@pytest.fixture
def tensor_data_loader_pipe(files_datapipe: IterDataPipe) -> TensorDataLoaderPipe:
    return TensorDataLoaderPipe(files_datapipe)


def test_initialization(tensor_data_loader_pipe: TensorDataLoaderPipe):
    assert isinstance(tensor_data_loader_pipe, TensorDataLoaderPipe)


def test_iter_produces_lists(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for res in tensor_data_loader_pipe:
        assert isinstance(res, list)


def test_iter_lists_consist_of_two_tuples(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for res in tensor_data_loader_pipe:
        assert all(isinstance(tup, tuple) and len(tup) == 2 for tup in res)


def test_iter_lists_consist_of_tuples_of_tensors(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for res in tensor_data_loader_pipe:
        assert all(isinstance(t1, torch.LongTensor) and isinstance(t2, torch.FloatTensor) for t1, t2 in res)


def test_iter_lists_consist_of_tuples_of_same_length_tensors(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for res in tensor_data_loader_pipe:
        assert all(t1.size(0) == t2.size(0) for t1, t2 in res)


def test_first_tensors_are_same_as_initial_data(
    tensor_data_loader_pipe: TensorDataLoaderPipe,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for res, data in zip(tensor_data_loader_pipe, multiple_docs_test_data):
        assert all((t1 == d1).all() for (t1, _), (d1, _) in zip(res, data))


def test_second_tensors_are_same_as_initial_data(
    tensor_data_loader_pipe: TensorDataLoaderPipe,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for res, data in zip(tensor_data_loader_pipe, multiple_docs_test_data):
        assert all((t2 == d2).all() for (_, t2), (_, d2) in zip(res, data))
