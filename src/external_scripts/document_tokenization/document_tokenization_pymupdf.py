import os
import re
import sys
from functools import partial
from itertools import groupby
from typing import Dict, Iterable, List, Optional, Tuple, Union

import fitz
from transformers import AutoTokenizer, PreTrainedTokenizer

from docllm.data.preprocessing.data_structure import Block, Document, Line, Page, Rect, Token, Word, WritingMode


def main(args: List[str]):
    print(args)
    if len(args) < 3:
        print("Usage: python document_tokenization.py <pdf_path> <output_dir> [<tokenizer_name_or_path>]")
        sys.exit(1)
    pdf_path = args[1]
    pdf_file_base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = args[2]
    tokenizer_name_or_path = "LeoLM/leo-hessianai-7b" if len(args) < 4 else args[3]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, legacy=True, add_bos_token=False)

    doc = parse_doc_from_pdf_path(pdf_path, tokenizer)

    with open(os.path.join(output_dir, pdf_file_base + ".json"), "w") as f:
        f.write(doc.model_dump_json())


def parse_doc_from_pdf_path(pdf_path: str, tokenizer: PreTrainedTokenizer, begin_of_word: str = "▁") -> Document:
    doc = fitz.open(pdf_path)
    pages = map(
        partial(parse_page_from_page_dict, tokenizer=tokenizer, begin_of_word=begin_of_word),
        (p.get_text("rawdict") for p in doc),
    )
    return Document(pages=list(pages), filename=pdf_path)


def parse_page_from_page_dict(
    page_dict: Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, Union[int, Rect]]]]]]]],
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str = "▁",
) -> Page:
    blocks = [
        block
        for block in map(
            partial(parse_block_from_block_dict, tokenizer=tokenizer, begin_of_word=begin_of_word),
            page_dict["blocks"],
        )
        if block is not None
    ]
    return Page(blocks=blocks, width=page_dict["width"], height=page_dict["height"])


def parse_block_from_block_dict(
    block_dict: Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, Union[int, Rect]]]]]]]],
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str = "▁",
) -> Optional[Block]:
    if block_dict["type"] != 0:
        return None
    lines = map(
        partial(parse_line_from_line_dict, tokenizer=tokenizer, begin_of_word=begin_of_word),
        block_dict["lines"],
    )
    return Block(lines=list(lines), bbox=block_dict["bbox"])


def parse_line_from_line_dict(
    line_dict: Dict[str, Union[str, List[Dict[str, Union[int, Rect]]]]],
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str = "▁",
) -> Line:
    chars = [c for span in line_dict["spans"] for c in span["chars"]]
    text = "".join(c["c"] for c in chars)
    writing_mode = WritingMode(line_dict["wmode"])
    split_chars = (list(group) for k, group in groupby(chars, key=lambda c: re.match(r"\s", c["c"])) if not k)
    words = map(
        partial(
            parse_word_from_char_dicts, writing_mode=writing_mode, tokenizer=tokenizer, begin_of_word=begin_of_word
        ),
        split_chars,
    )
    return Line(
        text=text,
        words=list(words),
        direction=(line_dict["dir"][0], line_dict["dir"][1]),
        writing_mode=writing_mode,
        bbox=line_dict["bbox"],
    )


def parse_word_from_char_dicts(
    char_dicts: List[Dict[str, Union[int, Rect]]],
    writing_mode: WritingMode,
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str = "▁",
) -> Word:
    text = "".join(char_dict["c"] for char_dict in char_dicts)
    tokens = list(_build_tokens(text, char_dicts, writing_mode, tokenizer, begin_of_word))
    bbox = _join_rects(*(t.bbox for t in tokens))
    return Word(text=text, tokens=tokens, bbox=bbox)


def _build_tokens(
    text: str,
    char_dicts: List[Dict[str, Union[int, Rect]]],
    writing_mode: WritingMode,
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str,
) -> Iterable[Token]:
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]
    token_groups = _build_token_strings_with_groups_of_escaped_tokens(token_ids, tokenizer, begin_of_word)
    for i, (t, b) in zip(token_ids, _build_token_boxes(token_groups, text, char_dicts, writing_mode)):
        yield Token(text=t, token_id=i, bbox=b)


def _build_token_strings_with_groups_of_escaped_tokens(
    token_ids: List[int], tokenizer: PreTrainedTokenizer, begin_of_word: str
):
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    tokens = _remove_begin_of_word_markers(tokens, begin_of_word)
    token_groups = _group_consecutive_escaped_tokens(tokens)
    return token_groups


def _remove_begin_of_word_markers(tokens: Iterable[str], begin_of_word: str) -> Iterable[str]:
    return filter(None, map(lambda t: re.sub(f"^{begin_of_word}", "", t), tokens))


def _group_consecutive_escaped_tokens(tokens: Iterable[str]) -> Iterable[List[str]]:
    return (
        list(map(lambda t: t[1], g))
        for _, g in groupby(enumerate(tokens), key=lambda idx_tok: -1 if _is_escaped(idx_tok[1]) else idx_tok[0])
    )


def _build_token_boxes(
    token_groups: Iterable[List[str]],
    original_text: str,
    char_dicts: List[Dict[str, Union[int, Rect]]],
    writing_mode: WritingMode,
) -> Iterable[Tuple[str, Rect]]:
    start_idx = 0
    for tg in token_groups:
        result_iter, start_idx = _create_token_boxes_for_token_group(
            tg, original_text, start_idx, char_dicts, writing_mode
        )
        yield from result_iter


def _create_token_boxes_for_token_group(
    token_group: List[str],
    original_text: str,
    start_idx: int,
    char_dicts: List[Dict[str, Union[int, Rect]]],
    writing_mode: WritingMode,
) -> Tuple[Iterable[Tuple[str, Rect]], int]:
    if len(token_group) == 1 and not _is_escaped(token_group[0]):
        detok_text = token_group[0]
        token_box, start_idx = _extract_token_box(detok_text, original_text, char_dicts, start_idx)
        return [(detok_text, token_box)], start_idx
    else:
        token_boxes, start_idx = _create_boxes_for_escaped_tokens(
            token_group, original_text, start_idx, char_dicts, writing_mode
        )
        return zip(token_group, token_boxes), start_idx


def _is_escaped(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def _create_boxes_for_escaped_tokens(
    tokens: List[str],
    original_text: str,
    start_idx: int,
    char_dicts: List[Dict[str, Union[int, Rect]]],
    writing_mode: WritingMode,
) -> Tuple[List[Rect], int]:
    token_bytes = "".join(map(_extract_bytes, tokens))
    detok_text = bytes.fromhex(token_bytes).decode("utf-8")
    token_box, start_idx = _extract_token_box(detok_text, original_text, char_dicts, start_idx)
    token_boxes = _split_token_box(token_box, len(tokens), writing_mode)
    return token_boxes, start_idx


def _extract_token_box(
    detokenized_text: str, original_text: str, char_dicts: List[Dict[str, Union[int, Rect]]], start_idx: int
) -> Tuple[Rect, int]:
    end_idx = start_idx + len(detokenized_text)
    assert detokenized_text == original_text[start_idx:end_idx]
    token_box = _join_rects(*(c["bbox"] for c in char_dicts[start_idx:end_idx]))
    return token_box, end_idx


def _join_rects(*rects: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    lefts, tops, rights, bottoms = zip(*rects)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _extract_bytes(token: str) -> bytes:
    assert len(token) == 6
    return token[3:5]


def _split_token_box(token_box: Rect, num_tokens: int, direction: WritingMode) -> Iterable[Rect]:
    left, top, right, bottom = token_box
    if direction == WritingMode.horizontal:
        width = (right - left) / num_tokens
        return ((left + i * width, top, left + (i + 1) * width, bottom) for i in range(num_tokens))
    else:
        height = (bottom - top) / num_tokens
        return ((left, top + i * height, right, top + (i + 1) * height) for i in range(num_tokens))


if __name__ == "__main__":
    main(sys.argv)
