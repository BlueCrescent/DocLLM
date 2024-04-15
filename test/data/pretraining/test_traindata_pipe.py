import random
from test.util import extract_return_value
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import IterDataPipe

from docllm.data import DocLLMPreTrainDataConfig, DocLLMTrainDataPipe
from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.traindata_pipe import DocLLMTrainDataPipe


@pytest.fixture
def test_data(
    num_docs: int,
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
        for _ in range(num_docs)
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
def num_tokens_in_docs(test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]) -> List[int]:
    return [sum(t.size(0) for t, _ in doc_data) for doc_data in test_data]


def test_instantiation(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert isinstance(docllm_train_data_pipe, DocLLMTrainDataPipe)


def test_iter_produces_three_tuples(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(data, tuple) and len(data) == 4 for data in docllm_train_data_pipe)


def test_all_tuple_entries_are_tensors(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(tensor, torch.Tensor) for data in docllm_train_data_pipe for tensor in data)


def test_first_tuple_entry_is_long_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t1, torch.LongTensor) for t1, _, _, _ in docllm_train_data_pipe)


def test_second_tuple_entry_is_float_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t2, torch.FloatTensor) for _, t2, _, _ in docllm_train_data_pipe)


def test_third_tuple_entry_is_bool_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t3, torch.BoolTensor) for _, _, t3, _ in docllm_train_data_pipe)


def test_fourth_tuple_entry_is_long_tensor(docllm_train_data_pipe: DocLLMTrainDataPipe):
    assert all(isinstance(t4, torch.LongTensor) for _, _, _, t4 in docllm_train_data_pipe)


def test_produces_correct_input_tensor_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_in_docs: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._masker._get_num_masks = extract_return_value(
        docllm_train_data_pipe._masker._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (input_tensor, _, _, _), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_in_docs, num_masks_values
    ):
        assert input_tensor.shape == (count_bos_tokens + num_tokens + 2 * num_masks,)


def test_produces_correct_spatial_input_tensor_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_in_docs: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._masker._get_num_masks = extract_return_value(
        docllm_train_data_pipe._masker._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (_, spatial_input, _, _), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_in_docs, num_masks_values
    ):
        assert spatial_input.shape == (count_bos_tokens + num_tokens + 2 * num_masks, 4)


def test_produces_correct_loss_mask_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_in_docs: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._masker._get_num_masks = extract_return_value(
        docllm_train_data_pipe._masker._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (_, _, loss_mask, _), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_in_docs, num_masks_values
    ):
        assert loss_mask.shape == (count_bos_tokens + num_tokens + 2 * num_masks,)


def test_produces_correct_label_tensor_shapes(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    num_tokens_in_docs: List[int],
    pretraining_config: DocLLMPreTrainDataConfig,
):
    num_masks_values, docllm_train_data_pipe._masker._get_num_masks = extract_return_value(
        docllm_train_data_pipe._masker._get_num_masks
    )
    count_bos_tokens = 1 if pretraining_config.bos_text_token else 0
    for (_, _, _, label_tensor), num_tokens, num_masks in zip(
        docllm_train_data_pipe, num_tokens_in_docs, num_masks_values
    ):
        assert label_tensor.shape == (count_bos_tokens + num_tokens + 2 * num_masks,)


def test_produces_correct_loss_mask_values(docllm_train_data_pipe: DocLLMTrainDataPipe):
    extraction_results, docllm_train_data_pipe._masker._extract_masked_tokens = extract_return_value(
        docllm_train_data_pipe._masker._extract_masked_tokens
    )
    for (_, _, loss_mask, _), (_, _, target_tokens, _) in zip(docllm_train_data_pipe, extraction_results):
        len_masked_to_one = sum(t.size(0) for t in target_tokens)
        assert not loss_mask[:-len_masked_to_one].any() and loss_mask[-len_masked_to_one:].all()


@pytest.mark.parametrize("num_masked_blocks,max_percentage_masked_blocks", [(1.0, 1.0)], indirect=True)
def test_mask_whole_sequence_works_for_text_tokens(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    pretraining_config: DocLLMPreTrainDataConfig,
    test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    for dat, (t1, _, _, _) in zip(test_data, docllm_train_data_pipe):
        assert (t1[1 : len(dat)] == pretraining_config.mask_text_token).all()


@pytest.mark.parametrize("num_masked_blocks,max_percentage_masked_blocks", [(1.0, 1.0)], indirect=True)
def test_mask_whole_sequence_works_for_box_tokens(
    docllm_train_data_pipe: DocLLMTrainDataPipe,
    pretraining_config: DocLLMPreTrainDataConfig,
    test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
):
    mask_bbox_token = torch.tensor(pretraining_config.mask_bbox_token)
    for dat, (_, t2, _, _) in zip(test_data, docllm_train_data_pipe):
        assert all((box == mask_bbox_token).all() for box in t2[1 : len(dat)])
