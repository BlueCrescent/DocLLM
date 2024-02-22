import glob
import os
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import IterDataPipe

from docllm.data.pretraining.tensor_data_loader_pipe import TensorDataLoaderPipe


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


@pytest.fixture
def add_empty_file(input_dir_with_data: str):
    torch.save([], f"{input_dir_with_data}/empty_doc.py")


def test_initialization(tensor_data_loader_pipe: TensorDataLoaderPipe):
    assert isinstance(tensor_data_loader_pipe, TensorDataLoaderPipe)


def test_iter_produces_two_tuples(tensor_data_loader_pipe: TensorDataLoaderPipe):
    assert all(isinstance(res, tuple) and len(res) == 2 for res in tensor_data_loader_pipe)


def test_iter_tuples_consist_of_lists(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for res in tensor_data_loader_pipe:
        assert all(isinstance(lst, list) for lst in res)


def test_iter_first_list_consist_of_long_tensors(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for lst1, _ in tensor_data_loader_pipe:
        assert all(isinstance(t, torch.LongTensor) for t in lst1)


def test_iter_second_list_consist_of_long_tensors(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for _, lst2 in tensor_data_loader_pipe:
        assert all(isinstance(t, torch.FloatTensor) for t in lst2)


def test_iter_matching_tensors_have_same_length(tensor_data_loader_pipe: TensorDataLoaderPipe):
    for lst1, lst2 in tensor_data_loader_pipe:
        for t1, t2 in zip(lst1, lst2):
            assert t1.size(0) == t2.size(0)


def test_first_tensors_are_same_as_initial_data(
    tensor_data_loader_pipe: TensorDataLoaderPipe,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for (lst1, _), data in zip(tensor_data_loader_pipe, multiple_docs_test_data):
        dat1, _ = zip(*data)
        assert all((t1 == d1).all() for t1, d1 in zip(lst1, dat1))


def test_second_tensors_are_same_as_initial_data(
    tensor_data_loader_pipe: TensorDataLoaderPipe,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for (_, lst2), data in zip(tensor_data_loader_pipe, multiple_docs_test_data):
        _, dat2 = zip(*data)
        assert all((t2 == d2).all() for t2, d2 in zip(lst2, dat2))


@pytest.mark.usefixtures("add_empty_file")
def test_empty_file_is_skipped(
    tensor_data_loader_pipe: TensorDataLoaderPipe,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    assert len(list(tensor_data_loader_pipe)) == len(multiple_docs_test_data)
