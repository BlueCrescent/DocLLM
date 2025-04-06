import random
from test.util import extract_return_value
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.masker import DocLLMPretrainingMasker


def test_instantiation(docllm_pretrain_masker: DocLLMPretrainingMasker):
    assert isinstance(docllm_pretrain_masker, DocLLMPretrainingMasker)


def test_produces_four_tuple(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    res = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert isinstance(res, tuple) and len(res) == 4


def test_all_tuple_entries_are_tensors(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    res = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert all(isinstance(tensor, torch.Tensor) for tensor in res)


def test_input_tokens_are_long_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    input_tokens, _, _, _ = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert isinstance(input_tokens, torch.LongTensor)


def test_bounding_boxes_are_float_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    _, bounding_boxes, _, _ = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert isinstance(bounding_boxes, torch.FloatTensor)


def test_loss_mask_is_bool_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    _, _, loss_mask, _ = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert isinstance(loss_mask, torch.BoolTensor)


def test_label_is_long_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    docllm_pretrain_masker: DocLLMPretrainingMasker,
):
    _, _, _, label = docllm_pretrain_masker(input_tokens, bbox_tokens)
    assert isinstance(label, torch.LongTensor)


def test_get_mask_indices():
    train_data_pipe = DocLLMPretrainingMasker(MagicMock(spec=DocLLMPreTrainDataConfig))
    num_blocks = 10
    num_masks = 4
    for _ in range(20):
        result = train_data_pipe._get_mask_indices(num_blocks, num_masks)
        assert len(result) == num_masks
        assert all(0 <= index < num_blocks for index in result)


@pytest.fixture
def docllm_pretrain_masker(pretraining_config: DocLLMPreTrainDataConfig) -> DocLLMPretrainingMasker:
    return DocLLMPretrainingMasker(pretraining_config)


@pytest.fixture
def input_tokens(test_data: Tuple[List[torch.LongTensor], List[torch.FloatTensor]]) -> List[torch.LongTensor]:
    return test_data[0]


@pytest.fixture
def bbox_tokens(test_data: Tuple[List[torch.LongTensor], List[torch.FloatTensor]]) -> List[torch.FloatTensor]:
    return test_data[1]


@pytest.fixture
def test_data(
    num_blocks: int,
    range_block_size: Tuple[int, int],
) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
    max_token = 1111
    return tuple(
        map(
            list,
            zip(
                *(
                    (
                        torch.randint(max_token, (b := random.randint(range_block_size[0], range_block_size[1]),)),
                        torch.rand(b, 4),
                    )
                    for _ in range(num_blocks)
                )
            ),
        )
    )
