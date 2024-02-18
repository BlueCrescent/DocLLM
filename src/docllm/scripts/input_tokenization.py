from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

import torch

Coord = float
Rect = Tuple[Coord, Coord, Coord, Coord]


def main(input_file_or_dir: str, output_dir: str, use_page_dimensions: bool) -> None:
    if os.path.isfile(input_file_or_dir):
        input_files = [input_file_or_dir]
    else:
        input_files = [
            os.path.join(input_file_or_dir, fn) for fn in os.listdir(input_file_or_dir) if fn.endswith(".json")
        ]
    for filename in input_files:
        document_tokenization(filename, output_dir, use_page_dimensions)


def document_tokenization(filename: str, output_dir: str, use_page_dimensions: bool) -> None:
    basename = os.path.splitext(os.path.basename(filename))[0]
    for i, page_block_tokens in enumerate(
        document_tokenization_to_pagewise_block_tokens(filename, use_page_dimensions)
    ):
        torch.save(page_block_tokens, os.path.join(output_dir, f"{basename}_{i}.pt"))


def document_tokenization_to_pagewise_block_tokens(
    filename: str, use_page_dimensions: bool
) -> Iterable[List[torch.Tensor]]:
    with open(filename, "r") as file:
        document_tokenization = json.load(file)
    for page in document_tokenization["pages"]:
        tokenizer = BoundingBoxTokenizer(page, use_page_dimensions)
        yield list(page_tokenization_to_block_tokens(page, tokenizer))


def page_tokenization_to_block_tokens(
    page_tokenization: Dict[str, Any], tokenizer: BoundingBoxTokenizer
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for block in page_tokenization["blocks"]:
        tokens = [
            (token["token_id"], tokenizer(token["bbox"]))
            for line in block["lines"]
            for word in line["words"]
            for token in word["tokens"]
        ]
        if len(tokens) == 0:
            continue
        text_tokens, bb_tokens = zip(*tokens)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        bb_tokens = torch.stack(bb_tokens)
        yield text_tokens, bb_tokens


class BoundingBoxTokenizer:
    def __init__(self, page: Dict[str, Any], use_page_dimensions: bool) -> None:
        if use_page_dimensions:
            self._width = page["width"]
            self._height = page["height"]
            self._minx = 0.0
            self._miny = 0.0
        else:
            self._minx, self._miny, self._width, self._height = self._compute_min_max_dimensions(page)

    def __call__(self, bounding_box: Rect) -> torch.FloatTensor:
        return torch.tensor(
            [
                (bounding_box[0] - self._minx) / self._width,
                (bounding_box[1] - self._miny) / self._height,
                (bounding_box[2] - self._minx) / self._width,
                (bounding_box[3] - self._miny) / self._height,
            ]
        )

    def _compute_min_max_dimensions(self, page: Dict[str, Any]) -> Tuple[Coord, Coord, Coord, Coord]:
        minx, miny, maxx, maxy = float("inf"), float("inf"), float("-inf"), float("-inf")
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    for token in word["tokens"]:
                        minx = min(minx, token["bbox"][0])
                        miny = min(miny, token["bbox"][1])
                        maxx = max(maxx, token["bbox"][2])
                        maxy = max(maxy, token["bbox"][3])
        return minx, miny, maxx - minx, maxy - miny


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 3:
        print("Usage: python document_tokenization.py <input_file_or_dir> <output_dir> [<use_page_dimensions>]")
        print("Default value for use_page_dimensions is True.")
        sys.exit(1)
    input_file_or_dir = sys.argv[1]
    output_dir = sys.argv[2]
    use_page_dimensions = bool(sys.argv[3]) if len(sys.argv) > 3 else True
    main(input_file_or_dir, output_dir, use_page_dimensions)
