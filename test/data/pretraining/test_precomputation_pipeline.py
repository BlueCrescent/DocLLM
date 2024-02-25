import os
from tempfile import TemporaryDirectory
from typing import Iterable, List, Tuple

import pytest
import torch

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.precomputation_pipeline import build_precomputed_data_pipeline, precompute_data


@pytest.fixture
def output_dir() -> Iterable[str]:
    with TemporaryDirectory() as output_dir:
        yield output_dir


@pytest.fixture
def precomputation_config(
    input_dir_with_data: str, pretraining_config: DocLLMPreTrainDataConfig
) -> DocLLMPreTrainDataConfig:
    config = pretraining_config.model_copy(
        update={
            "batch_size": 1,
            "drop_last_batch_if_not_full": False,
            "shuffle": False,
            "use_sharding_filter": False,
            "directory": input_dir_with_data,
        }
    )
    return config


def test_precompute_data_produces_expected_number_of_files(
    precomputation_config: DocLLMPreTrainDataConfig, output_dir: str, num_docs: int
):
    precompute_data(precomputation_config, output_dir)
    assert len(os.listdir(output_dir)) == num_docs


def test_precompute_data_produces_expected_files(
    precomputation_config: DocLLMPreTrainDataConfig, output_dir: str, num_docs: int
):
    precompute_data(precomputation_config, output_dir)
    for i in range(num_docs):
        assert os.path.exists(os.path.join(output_dir, f"eval_{i}.pt"))


def test_loading_precomputed_files_yields_expected_number_of_files(
    precomputation_config: DocLLMPreTrainDataConfig, output_dir: str, num_docs: int
):
    precompute_data(precomputation_config, output_dir)
    datapipe = build_precomputed_data_pipeline(output_dir)
    assert len(list(datapipe)) == num_docs


def test_loaded_files_have_consistent_shape(
    precomputation_config: DocLLMPreTrainDataConfig, output_dir: str, num_docs: int
):
    precompute_data(precomputation_config, output_dir)
    datapipe = build_precomputed_data_pipeline(output_dir)
    for i, data in enumerate(datapipe):
        assert len(data) == 4
        assert len(data[0]) == len(data[1]) == len(data[2]) == len(data[3])
        for j in range(len(data[0])):
            assert data[0][j].shape[0] == data[1][j].shape[0]
            assert data[0][j].shape[0] == data[2][j].shape[0]
            assert data[0][j].shape[0] == data[3][j].shape[0]


def test_loaded_files_have_original_text_tokens_where_not_masked(
    precomputation_config: DocLLMPreTrainDataConfig,
    output_dir: str,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    precomputation_config.use_packing = False
    precompute_data(precomputation_config, output_dir)
    datapipe = build_precomputed_data_pipeline(output_dir)
    for i, data in enumerate(datapipe):
        pipe_data_idx = 1
        for block in multiple_docs_test_data[i]:
            if data[0][0][pipe_data_idx] == precomputation_config.mask_text_token:
                pipe_data_idx += 1
            else:
                block_size = len(block[0])
                new_idx = pipe_data_idx + block_size
                assert torch.equal(data[0][0][pipe_data_idx:new_idx], block[0])
                pipe_data_idx = new_idx


def test_loaded_files_have_original_bbox_tokens_where_not_masked(
    precomputation_config: DocLLMPreTrainDataConfig,
    output_dir: str,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    precomputation_config.use_packing = False
    precompute_data(precomputation_config, output_dir)
    datapipe = build_precomputed_data_pipeline(output_dir)
    for i, data in enumerate(datapipe):
        pipe_data_idx = 1
        for block in multiple_docs_test_data[i]:
            if data[0][0][pipe_data_idx] == precomputation_config.mask_text_token:
                pipe_data_idx += 1
            else:
                block_size = len(block[0])
                new_idx = pipe_data_idx + block_size
                assert torch.equal(data[1][0][pipe_data_idx:new_idx], block[1])
                pipe_data_idx = new_idx


def test_loaded_files_test_and_bbox_tokens_are_masked_at_same_positions(
    precomputation_config: DocLLMPreTrainDataConfig,
    output_dir: str,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    precomputation_config.use_packing = False
    precompute_data(precomputation_config, output_dir)
    datapipe = build_precomputed_data_pipeline(output_dir)
    for i, data in enumerate(datapipe):
        pipe_data_idx = 1
        for block in multiple_docs_test_data[i]:
            if data[0][0][pipe_data_idx] == precomputation_config.mask_text_token:
                assert torch.equal(data[1][0][pipe_data_idx], torch.tensor(precomputation_config.mask_bbox_token))
                pipe_data_idx += 1
            else:
                assert not torch.equal(data[1][0][pipe_data_idx], torch.tensor(precomputation_config.mask_bbox_token))
                pipe_data_idx += len(block[0])
