import json
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory, _TemporaryFileWrapper
from test.scripts.example_json import document_dict
from typing import Callable, Iterable, List, Tuple

import pytest
import torch

from docllm.scripts.input_tokenization import (
    BoundingBoxTokenizer,
    document_tokenization,
    page_tokenization_to_block_tokens,
)


@pytest.fixture
def input_file() -> Iterable[str]:
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json") as file:
        json.dump(document_dict, file)
        file.flush()
        yield file.name


@pytest.fixture
def output_dir() -> Iterable[str]:
    with TemporaryDirectory() as output_dir:
        yield output_dir


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_document_tokenization_produces_correct_number_of_files(
    input_file: str, output_dir: str, use_page_dimensions: bool
):
    document_tokenization(input_file, output_dir, use_page_dimensions)
    assert len(os.listdir(output_dir)) == len(document_dict["pages"])


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_document_tokenization_produces_correct_number_of_tensors(
    input_file: str, output_dir: str, use_page_dimensions: bool
):
    def test(tensors: List[Tuple[torch.Tensor]], page_num: int):
        assert len(tensors) == len(document_dict["pages"][page_num]["blocks"])

    document_tokenization(input_file, output_dir, use_page_dimensions)
    _run_test_for_each_page_result_file(output_dir, test)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_document_tokenization_produces_two_tensors_for_each_block(
    input_file: str, output_dir: str, use_page_dimensions: bool
):
    def test(tensors: List[Tuple[torch.Tensor]], page_num: int):
        assert all(len(tensor) == 2 for tensor in tensors)

    document_tokenization(input_file, output_dir, use_page_dimensions)
    _run_test_for_each_page_result_file(output_dir, test)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_document_tokenization_tensor_lengths_are_number_of_tokens_in_block(
    input_file: str, output_dir: str, use_page_dimensions: bool
):
    def test(tensors: List[Tuple[torch.Tensor]], page_num: int):
        for (text_tensor, bbox_tensor), block in zip(tensors, document_dict["pages"][page_num]["blocks"]):
            num_tokens = len([t for l in block["lines"] for w in l["words"] for t in w["tokens"]])
            assert text_tensor.shape == (num_tokens,) and bbox_tensor.shape[0] == num_tokens

    document_tokenization(input_file, output_dir, use_page_dimensions)
    _run_test_for_each_page_result_file(output_dir, test)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_document_tokenization_produces_bbox_tensors_with_size_four(
    input_file: str, output_dir: str, use_page_dimensions: bool
):
    def test(tensors: List[Tuple[torch.Tensor]], page_num: int):
        for tensor in tensors:
            assert tensor[1].shape[-1] == 4

    document_tokenization(input_file, output_dir, use_page_dimensions)
    _run_test_for_each_page_result_file(output_dir, test)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_page_tokenization_to_block_tokens_produces_correct_number_of_tensors(use_page_dimensions):
    page = document_dict["pages"][1]
    tokenizer = BoundingBoxTokenizer(page, use_page_dimensions=use_page_dimensions)
    tensors = list(page_tokenization_to_block_tokens(page, tokenizer))
    for (text_tensor, bbox_tensor), b in zip(tensors, page["blocks"]):
        num_tokens = len([t for l in b["lines"] for w in l["words"] for t in w["tokens"]])
        assert text_tensor.shape == (num_tokens,) and len(bbox_tensor) == num_tokens


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_page_tokenization_to_block_tokens_produces_bbox_tensors_with_size_four(use_page_dimensions):
    page = document_dict["pages"][1]
    tokenizer = BoundingBoxTokenizer(page, use_page_dimensions=use_page_dimensions)
    tensors = list(page_tokenization_to_block_tokens(page, tokenizer))
    assert all(bbox_tensor.shape[-1] == 4 for _, bbox_tensor in tensors)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_bounding_box_tokenizer_produces_correct_output_shape(use_page_dimensions: bool):
    tokenizer = BoundingBoxTokenizer(document_dict["pages"][1], use_page_dimensions=use_page_dimensions)
    bounding_box = [10.0, 15.0, 20.0, 25.0]
    assert tokenizer(bounding_box).shape == (4,)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_tokenized_bounding_boxes_in_zero_one(use_page_dimensions: bool):
    tokenizer = BoundingBoxTokenizer(document_dict["pages"][1], use_page_dimensions=use_page_dimensions)
    token_bbs = [
        tokenizer(t["bbox"])
        for block in document_dict["pages"][1]["blocks"]
        for line in block["lines"]
        for word in line["words"]
        for t in word["tokens"]
    ]
    assert all((t >= 0).all() and (t <= 1).all() for t in token_bbs)


@pytest.mark.parametrize("use_page_dimensions", [True, False])
def test_tokenized_bounding_boxes_have_non_negative_lengths(use_page_dimensions: bool):
    tokenizer = BoundingBoxTokenizer(document_dict["pages"][1], use_page_dimensions=use_page_dimensions)
    token_bbs = [
        tokenizer(t["bbox"])
        for block in document_dict["pages"][1]["blocks"]
        for line in block["lines"]
        for word in line["words"]
        for t in word["tokens"]
    ]
    assert all(t[0] <= t[2] and t[1] <= t[3] for t in token_bbs)


def test_bounding_box_tokenizer_produces_scaled_coordinates_using_page_dimensions():
    page = document_dict["pages"][1]

    use_page_dimensions = True
    tokenizer = BoundingBoxTokenizer(page, use_page_dimensions=use_page_dimensions)
    bounding_box = [10.0, 15.0, 20.0, 25.0]
    token_bb = [
        bounding_box[0] / page["width"],
        bounding_box[1] / page["height"],
        bounding_box[2] / page["width"],
        bounding_box[3] / page["height"],
    ]
    assert tokenizer(bounding_box).tolist() == pytest.approx(token_bb, 0.0001)


def test_bounding_box_tokenizer_produces_scaled_coordinates_using_min_max():
    page = document_dict["pages"][1]
    coords = [t["bbox"] for b in page["blocks"] for l in b["lines"] for w in l["words"] for t in w["tokens"]]
    minx_coords, miny_coords, maxx_coords, maxy_coords = zip(*coords)
    minx, miny, maxx, maxy = min(minx_coords), min(miny_coords), max(maxx_coords), max(maxy_coords)
    width, height = maxx - minx, maxy - miny

    use_page_dimensions = False
    tokenizer = BoundingBoxTokenizer(page, use_page_dimensions=use_page_dimensions)
    bounding_box = [10.0, 15.0, 20.0, 25.0]
    token_bb = [
        (bounding_box[0] - minx) / width,
        (bounding_box[1] - miny) / height,
        (bounding_box[2] - minx) / width,
        (bounding_box[3] - miny) / height,
    ]
    assert tokenizer(bounding_box).tolist() == pytest.approx(token_bb, 0.0001)


def _run_test_for_each_page_result_file(output_dir: str, test: Callable[[List[Tuple[torch.Tensor]], int], None]):
    for filename in os.listdir(output_dir):
        with open(os.path.join(output_dir, filename), "rb") as file:
            tensors = torch.load(file)
            page_num = int(filename.split("_")[-1].split(".")[0])
            test(tensors, page_num)
