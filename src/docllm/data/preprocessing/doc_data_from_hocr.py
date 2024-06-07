import re
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import groupby
from typing import Iterable, List, Tuple

from torch import Block
from transformers import PreTrainedTokenizer

from docllm.data.preprocessing.data_structure import Document, Line, Page, Token, Word, WritingMode


def parse_hocr_document(hocr_path: str, tokenizer: PreTrainedTokenizer) -> Document:
    tree = ET.parse(hocr_path)
    root = tree.getroot()
    body = root.findall(".//body")
    pages = body[0].findall(".//div[@class='ocr_page']")
    return Document(filename=hocr_path, pages=[parse_hocr_page(page, tokenizer) for page in pages])


def parse_hocr_page(page: ET.Element, tokenizer: PreTrainedTokenizer) -> Page:
    blocks = page.findall(".//content_area")
    return Page(
        blocks=[parse_hocr_block(block, tokenizer) for block in blocks],
        width=float(page.attrib["width"]),
        height=float(page.attrib["height"]),
    )


def parse_hocr_block(block: ET.Element, tokenizer: PreTrainedTokenizer) -> Block:
    paragraphs = block.findall(".//paragraph")
    lines = [l for p in paragraphs for l in p.findall(".//span[@class='ocr_line']")]
    lines = [parse_hocr_line(line, tokenizer) for line in lines]
    return Block(lines=lines, bbox=_join_rects(*(line.bbox for line in lines)))


def parse_hocr_line(line: ET.Element, tokenizer: PreTrainedTokenizer) -> Line:
    words = line.findall(".//span[@class='ocrx_word']")
    bbox = parse_bbox(line)
    word_writing_modes = Counter(parse_writing_mode_from_baseline(word) for word in words)
    writing_mode = WritingMode.horizontal
    return Line(
        text=" ".join(w.text for w in words),
        words=[parse_hocr_word(word, tokenizer, writing_mode) for word in words],
        direction=(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])),
        writing_mode=writing_mode,
        bbox=bbox,
    )


def parse_hocr_word(word: ET.Element, tokenizer: PreTrainedTokenizer) -> Word:
    bbox = parse_bbox(word)
    writing_mode = WritingMode.horizontal
    tokens = list(_build_word_tokens(word.text, tokenizer, "▁", bbox, writing_mode))
    return Word(text=word.text, tokens=tokens, bbox=bbox)


def parse_bbox(element: ET.Element) -> Tuple[float, float, float, float]:
    bbox_str = [el for el in element.attrib["title"].split(";") if el.strip().startswith("bbox")][0]
    return tuple(map(float, bbox_str.split(" ")[1:]))


def parse_writing_mode_from_baseline(word: ET.Element) -> WritingMode:
    baseline_str = [el for el in word.attrib["title"].split(";") if el.strip().startswith("baseline")][0]
    minx, miny, maxx, maxy = map(float, baseline_str.split(" ")[1:])
    if maxx - minx > maxy - miny:
        return WritingMode.horizontal
    return WritingMode.vertical


def _join_rects(*rects: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    lefts, tops, rights, bottoms = zip(*rects)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _build_word_tokens(
    text: str,
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str,
    word_bbox: Tuple[float, float, float, float],
    writing_mode: WritingMode,
) -> Iterable[Token]:
    grouped_tokens_and_ids = _tokenize_word(text, tokenizer, begin_of_word)
    if writing_mode == WritingMode.horizontal:
        yield from _build_horizontal_tokens(text, word_bbox, grouped_tokens_and_ids)
    else:
        yield from _build_vertical_tokens(text, word_bbox, grouped_tokens_and_ids)


def _tokenize_word(text: str, tokenizer: PreTrainedTokenizer, begin_of_word: str) -> Iterable[List[Tuple[str, int]]]:
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    tokens = map(lambda t: re.sub(f"^{begin_of_word}", "", t), tokens)
    tokens_and_ids = zip(tokens, token_ids)
    tokens_and_ids = filter(lambda t: t[0], tokens_and_ids)
    grouped_tokens_and_ids = _group_consecutive_escaped_tokens(tokens_and_ids)

    # tokens = _unescape_tokens(tokens)
    # tokens_and_ids = zip(tokens, token_ids)
    return grouped_tokens_and_ids


def _build_horizontal_tokens(
    text: str, word_bbox: Tuple[float, float, float, float], grouped_tokens_and_ids: Iterable[List[Tuple[str, int]]]
) -> Iterable[Token]:
    left, top, right, bottom = word_bbox
    char_width = (right - left) / len(text)
    for group in grouped_tokens_and_ids:
        if len(group) == 1:
            t, tid = group[0]
            token, left = _build_horizontal_token(left, top, bottom, char_width, t, tid, 1.0)
            yield token
        else:
            token_bytes = "".join(map(_extract_bytes, (t for t, _ in group)))
            detok_text = bytes.fromhex(token_bytes).decode("utf-8")
            scale = len(detok_text) / len(group)
            for t, tid in group:
                token, left = _build_horizontal_token(left, top, bottom, char_width, t, tid, scale)
                yield token


def _build_horizontal_token(
    left: float, top: float, bottom: float, char_width: float, t: str, tid: int, scale: float
) -> Tuple[Token, float]:
    token_width = len(t) * char_width / scale
    bbox = (left, top, left + token_width, bottom)
    left += token_width
    return Token(text=t, token_id=tid, bbox=bbox), left


def _build_vertical_tokens(
    text: str, word_bbox: Tuple[float, float, float, float], grouped_tokens_and_ids: Iterable[List[Tuple[str, int]]]
) -> Iterable[Token]:
    left, top, right, bottom = word_bbox
    char_width = (bottom - top) / len(text)
    for group in grouped_tokens_and_ids:
        if len(group) == 1:
            t, tid = group[0]
            token, top = _build_vertical_token(left, top, right, char_width, t, tid, 1.0)
            yield token
        else:
            scale = sum(len(t) for t, _ in group) / len(group)
            for t, tid in group:
                token, top = _build_vertical_token(left, top, right, char_width, t, tid, scale)
                yield token


def _build_vertical_token(
    left: float, top: float, right: float, char_width: float, t: str, tid: int, scale: float
) -> Tuple[Token, float]:
    token_width = len(t) * char_width / scale
    bbox = (left, top, right, top + token_width)
    top += token_width
    token = Token(text=t, token_id=tid, bbox=bbox)
    return token, top


def _extract_bytes(token: str) -> bytes:
    assert len(token) == 6
    return token[3:5]


def _group_consecutive_escaped_tokens(tokens_and_ids: Iterable[Tuple[str, int]]) -> Iterable[List[Tuple[str, int]]]:
    get_idx = lambda idx_tok_id: idx_tok_id[0]
    get_token = lambda idx_tok_id: idx_tok_id[1][0]
    get_group_index = lambda idx_tok_id: -1 if _is_escaped(get_token(idx_tok_id)) else get_idx(idx_tok_id)
    drop_index = lambda idx_tok_id: idx_tok_id[1]
    return (list(map(drop_index, g)) for _, g in groupby(enumerate(tokens_and_ids), key=get_group_index))


def _unescape_tokens(tokens: Iterable[str]) -> Iterable[str]:
    return map(lambda t: bytes.fromhex(t[3:5]).decode("utf-8") if _is_escaped(t) else t, tokens)


def _is_escaped(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")
