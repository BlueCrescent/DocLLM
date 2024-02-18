import random
from test.util import extract_return_value
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.datapipes.iter import IterDataPipe

from docllm.data import DocLLMPreTrainDataConfig, DocLLMTrainDataPipe
from docllm.data.pretrain_data_pipe import DocLLMTrainDataPipe
from docllm.data.pretraining_config import DocLLMPreTrainDataConfig


@pytest.fixture
def num_pages() -> int:
    return 3


@pytest.fixture
def test_data(
    num_pages: int,
    num_blocks: int,
    range_block_size: Tuple[int, int],
) -> List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]:
    max_token = 1111
    return [
        [
            (
                torch.randint(max_token, (b := random.randint(range_block_size[0], range_block_size[1]),)),
                torch.rand(b, 4),
            )
            for _ in range(num_blocks)
        ]
        for _ in range(num_pages)
    ]


@pytest.fixture
def source_datapipe(test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]) -> IterDataPipe:
    def split_list(
        lst: List[Tuple[torch.LongTensor, torch.FloatTensor]]
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
        l1, l2 = zip(*lst)
        return list(l1), list(l2)

    pipe = MagicMock(spec=IterDataPipe)
    pipe.__iter__.return_value = map(split_list, test_data)
    return pipe


@pytest.fixture
def docllm_train_data_pipe(
    source_datapipe: IterDataPipe, pretraining_config: DocLLMPreTrainDataConfig
) -> DocLLMTrainDataPipe:
    return DocLLMTrainDataPipe(source_datapipe, pretraining_config)


@pytest.fixture
def num_tokens_on_pages(test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]) -> List[int]:
    return [sum(t.size(0) for t, _ in page_data) for page_data in test_data]


def test_instantiation(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert isinstance(docllm_train_data_pipe, DocLLMTrainDataPipe)


def test_iter_produces_three_tuples(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(data, tuple) and len(data) == 3 for data in docllm_train_data_pipe)


def test_all_tuple_entries_are_tensors(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(tensor, torch.Tensor) for data in docllm_train_data_pipe for tensor in data)


def test_first_tuple_entry_is_long_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t1, torch.LongTensor) for t1, _, _ in docllm_train_data_pipe)


def test_second_tuple_entry_is_float_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t2, torch.FloatTensor) for _, t2, _ in docllm_train_data_pipe)


def test_third_tuple_entry_is_bool_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t3, torch.BoolTensor) for _, _, t3 in docllm_train_data_pipe)


def test_produces_correct_input_tensor_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_on_pages: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._get_num_masks = extract_return_value(
        docllm_train_data_pipe._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (input_tensor, _, _), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_on_pages, num_masks_values
    ):
        assert input_tensor.shape == (count_bos_tokens + num_tokens + 2 * num_masks,)


def test_produces_correct_spatial_input_tensor_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_on_pages: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._get_num_masks = extract_return_value(
        docllm_train_data_pipe._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (_, spatial_input, _), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_on_pages, num_masks_values
    ):
        assert spatial_input.shape == (count_bos_tokens + num_tokens + 2 * num_masks, 4)


def test_produces_correct_loss_mask_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_on_pages: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._get_num_masks = extract_return_value(
        docllm_train_data_pipe._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (_, _, loss_mask), num_tokens, num_masks in zip(docllm_train_data_pipe, num_tokens_on_pages, num_masks_values):
        assert loss_mask.shape == (count_bos_tokens + num_tokens + 2 * num_masks,)


def test_produces_correct_loss_mask_values(docllm_train_data_pipe: DocLLMTrainDataPipe):
    extraction_results, docllm_train_data_pipe._extract_masked_tokens = extract_return_value(
        docllm_train_data_pipe._extract_masked_tokens
    )
    for (_, _, loss_mask), (_, _, target_tokens) in zip(docllm_train_data_pipe, extraction_results):
        len_masked_to_one = sum(t.size(0) for t in target_tokens) - 1
        assert not loss_mask[:-len_masked_to_one].any() and loss_mask[-len_masked_to_one:].all()


def test_get_mask_indices():
    train_data_pipe = DocLLMTrainDataPipe(MagicMock(), MagicMock(spec=DocLLMPreTrainDataConfig))
    num_blocks = 10
    num_masks = 4
    for _ in range(20):
        result = train_data_pipe._get_mask_indices(num_blocks, num_masks)
        assert len(result) == num_masks
        assert all(0 <= index < num_blocks for index in result)
