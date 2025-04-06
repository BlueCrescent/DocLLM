import random
from typing import List, Tuple

import pytest
import torch

from docllm.data.finetuning.reading_order import ReadingOrderConfig, ReadingOrderSampleBuilder, get_num_tokens


def test_instantiation(reading_order_sample_builder: ReadingOrderSampleBuilder):
    assert isinstance(reading_order_sample_builder, ReadingOrderSampleBuilder)


def test_produces_four_tuple(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    res = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert isinstance(res, tuple) and len(res) == 4


def test_all_tuple_entries_are_tensors(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    res = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert all(isinstance(tensor, torch.Tensor) for tensor in res)


def test_input_tokens_are_long_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    input_tokens, _, _, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert isinstance(input_tokens, torch.LongTensor)


def test_bounding_boxes_are_float_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, bounding_boxes, _, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert isinstance(bounding_boxes, torch.FloatTensor)


def test_loss_mask_is_bool_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, _, loss_mask, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert isinstance(loss_mask, torch.BoolTensor)


def test_label_is_long_tensor(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, _, _, label = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert isinstance(label, torch.LongTensor)


def test_result_shapes_match(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    input_ids, bounding_boxes, loss_mask, label = reading_order_sample_builder(input_tokens, bbox_tokens)
    seq_len = input_ids.size(0)
    assert bounding_boxes.size() == (seq_len, 4)
    assert label.size() == (seq_len,)
    assert loss_mask.size() == (seq_len,)


def test_input_tokens_shape(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
    expected_sequence_length: int,
):
    input_ids, _, _, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    assert input_ids.size() == (expected_sequence_length,)


def test_expected_output_in_input_ids_has_original_sequence(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    input_ids, _, _, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    original_token_sequence = torch.cat(input_tokens)
    num_tokens = len(original_token_sequence)
    assert torch.equal(input_ids[-(num_tokens):], original_token_sequence)


def test_expected_output_in_bounding_boxes_has_original_sequence(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, bounding_boxes, _, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    original_bounding_boxes = torch.cat(bbox_tokens)
    num_tokens = len(original_bounding_boxes)
    assert torch.equal(bounding_boxes[-(num_tokens):], original_bounding_boxes)


def test_expected_output_in_labels_has_original_sequence(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, _, _, label = reading_order_sample_builder(input_tokens, bbox_tokens)
    original_token_sequence = torch.cat(input_tokens)
    num_tokens = len(original_token_sequence)
    assert torch.equal(label[-(num_tokens + 1) : -1], original_token_sequence)


def test_loss_mask_masks_expected_output(
    input_tokens: List[torch.LongTensor],
    bbox_tokens: List[torch.FloatTensor],
    reading_order_sample_builder: ReadingOrderSampleBuilder,
):
    _, _, loss_mask, _ = reading_order_sample_builder(input_tokens, bbox_tokens)
    original_token_sequence = torch.cat(input_tokens)
    num_tokens = len(original_token_sequence)
    assert torch.equal(loss_mask[-num_tokens - 1 :], torch.ones(num_tokens + 1, dtype=torch.bool))
    assert torch.equal(loss_mask[: -num_tokens - 1], torch.zeros_like(loss_mask[: -num_tokens - 1], dtype=torch.bool))


@pytest.fixture
def expected_sequence_length(input_tokens: List[torch.LongTensor], readingorder_config: ReadingOrderConfig) -> int:
    bos_add = 1 if readingorder_config.bos_text_token is not None else 0
    eos_add = 1 if readingorder_config.eos_text_token is not None else 0
    num_tokens = get_num_tokens(input_tokens)
    prompt_length = len(readingorder_config.prompt_token_ids)

    return bos_add + num_tokens + eos_add + prompt_length + bos_add + num_tokens


@pytest.fixture
def reading_order_sample_builder(readingorder_config: ReadingOrderConfig) -> ReadingOrderSampleBuilder:
    return ReadingOrderSampleBuilder(readingorder_config)


@pytest.fixture
def readingorder_config(max_sequence_length: int) -> ReadingOrderConfig:
    return ReadingOrderConfig(
        max_seq_length=max_sequence_length,
        min_num_blocks_available=3,
        min_num_tokens_available=2,
        bos_text_token=1,
        bos_bbox_token=(0.0, 0.0, 0.0, 0.0),
        eos_text_token=2,
        eos_bbox_token=(1.0, 1.0, 1.0, 1.0),
        prompt_token_ids=[4, 5, 6],
    )


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


@pytest.fixture
def max_sequence_length() -> int:
    return 128


@pytest.fixture
def num_blocks() -> int:
    return 10


@pytest.fixture
def range_block_size() -> Tuple[int, int]:
    return (2, 5)
