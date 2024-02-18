from typing import List, Tuple

import pytest
import torch
from torch.utils.data import IterDataPipe

from docllm.data.pretraining_data_config import DocLLMPreTrainDataConfig
from docllm.data.pretraining_pipeline import build_docllm_datapipeline


@pytest.fixture
def num_batches(num_docs: int, batch_size: int) -> int:
    return num_docs // batch_size + (1 if num_docs % batch_size else 0)


@pytest.fixture
def pretraining_config_with_input_dir(
    pretraining_config: DocLLMPreTrainDataConfig, input_dir_with_data: str
) -> DocLLMPreTrainDataConfig:
    pretraining_config = pretraining_config.model_copy()
    pretraining_config.directory = input_dir_with_data
    return pretraining_config


@pytest.fixture
def pipeline(pretraining_config_with_input_dir: DocLLMPreTrainDataConfig) -> IterDataPipe:
    return build_docllm_datapipeline(pretraining_config_with_input_dir)


def test_initialization(pipeline: IterDataPipe):
    assert isinstance(pipeline, IterDataPipe)


def test_datapipe_produces_expected_number_of_document_results(pipeline: IterDataPipe, num_docs: int):
    assert len(list(pipeline)) == num_docs


@pytest.mark.parametrize("batch_size", (2, 4, 7), indirect=True)
def test_datapipe_produces_expected_number_of_document_results_with_varying_batch_sizes(
    pipeline: IterDataPipe, num_batches: int
):
    assert len(list(pipeline)) == num_batches


def test_datapipe_produces_result_list_with_batch_size_entries(pipeline: IterDataPipe):
    assert all(isinstance(res, list) and len(res) == 3 for res in list(pipeline)[:-1])


def test_datapipe_first_list_entry_is_long_tensor(pipeline: IterDataPipe):
    assert all(isinstance(res[0], torch.LongTensor) for res in pipeline)


def test_datapipe_second_list_entry_is_float_tensor(pipeline: IterDataPipe):
    assert all(isinstance(res[1], torch.FloatTensor) for res in pipeline)


def test_datapipe_third_list_entry_is_bool_tensor(pipeline: IterDataPipe):
    assert all(isinstance(res[2], torch.BoolTensor) for res in pipeline)


@pytest.mark.parametrize("batch_size", (2, 4, 7), indirect=True)
def test_datapipe_tensors_have_length_batch_size(
    pipeline: IterDataPipe,
    batch_size: int,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for batch_start_idx, res in zip(range(0, len(multiple_docs_test_data), batch_size), pipeline):
        effective_batch_size = min(batch_size, len(multiple_docs_test_data) - batch_start_idx)
        assert all(r.size(0) == effective_batch_size for r in res)
