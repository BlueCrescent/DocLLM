import re
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import groupby
from typing import Iterable, List, Tuple

from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from docllm.data.preprocessing.data_structure import Block, Document, Line, Page, Token, Word, WritingMode


class Setup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    build_token_bboxes: bool = False
    begin_of_word: str = "▁"


def parse_hocr_document(hocr_path: str, setup: Setup) -> Document:
    root = parse_without_namespaces(hocr_path)
    body = root.findall(".//body")
    pages = body[0].findall(".//div[@class='ocr_page']")
    return Document(filename=hocr_path, pages=[pp for page in pages if (pp := parse_hocr_page(page, setup)).blocks])


def parse_without_namespaces(hocr_path: str) -> ET.Element:
    it = ET.iterparse(hocr_path)
    for _, el in it:
        _, _, el.tag = el.tag.rpartition("}")
    return it.root


def parse_hocr_page(page: ET.Element, setup: Setup) -> Page:
    blocks = page.findall(".//content_area")
    _, _, width, height = page.attrib["title"].split("bbox")[1].split(";")[0].strip().split()
    return Page(
        blocks=[pb for block in blocks if (pb := parse_hocr_block(block, setup)).lines],
        width=float(width),
        height=float(height),
    )


def parse_hocr_block(block: ET.Element, setup: Setup) -> Block:
    paragraphs = block.findall(".//paragraph")
    lines = [l for p in paragraphs for l in p.findall(".//span[@class='ocr_line']")]
    parsed_lines = [pl for line in lines if (pl := parse_hocr_line(line, setup)).words]
    if not parsed_lines:
        return Block(lines=[], bbox=(0.0, 0.0, 0.0, 0.0))
    return Block(lines=parsed_lines, bbox=join_rects(*(l.bbox for l in parsed_lines)))


def parse_hocr_line(line: ET.Element, setup: Setup) -> Line:
    words = [w for w in line.findall(".//span[@class='ocrx_word']") if w.text is not None]
    bbox = parse_bbox(line)
    writing_mode = determine_writing_mode(words)
    return Line(
        text=" ".join(w.text for w in words),
        words=[parse_hocr_word(word, writing_mode, setup) for word in words],
        direction=(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])),
        writing_mode=writing_mode,
        bbox=bbox,
    )


def determine_writing_mode(words: List[ET.Element]) -> WritingMode:
    if len(words) == 0:
        return WritingMode.horizontal
    word_writing_modes = Counter(parse_writing_mode_from_baseline(word) for word in words)
    writing_mode = word_writing_modes.most_common(1)[0][0]
    return writing_mode


def parse_writing_mode_from_baseline(word: ET.Element) -> WritingMode:
    baseline_str = [el for el in word.attrib["title"].split(";") if el.strip().startswith("baseline")][0].strip()
    minx, miny, maxx, maxy = map(float, baseline_str.split(" ")[1:])
    if maxx - minx > maxy - miny:
        return WritingMode.horizontal
    return WritingMode.vertical


def parse_hocr_word(word: ET.Element, writing_mode: WritingMode, setup: Setup) -> Word:
    bbox = parse_bbox(word)
    if setup.build_token_bboxes:
        tokens = list(build_word_tokens(word.text, setup.tokenizer, setup.begin_of_word, bbox, writing_mode))
    else:
        tokens = list(build_word_tokens_with_same_bounding_box(word.text, setup.tokenizer, bbox))
    return Word(text=word.text, tokens=tokens, bbox=bbox)


def parse_bbox(element: ET.Element) -> Tuple[float, float, float, float]:
    bbox_str = [el for el in element.attrib["title"].split(";") if el.strip().startswith("bbox")][0].strip()
    return tuple(map(float, bbox_str.split(" ")[1:]))


def join_rects(*rects: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    lefts, tops, rights, bottoms = zip(*rects)
    return min(lefts), min(tops), max(rights), max(bottoms)


def build_word_tokens_with_same_bounding_box(
    text: str, tokenizer: PreTrainedTokenizer, word_bbox: Tuple[float, float, float, float]
) -> Iterable[Token]:
    token_ids = tokenize_with_leading_space(text, tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    for t, t_id in zip(tokens, token_ids):
        yield Token(text=t, token_id=t_id, bbox=word_bbox)


def tokenize_with_leading_space(
    text: str,
    tokenizer: PreTrainedTokenizer,
) -> List[int]:
    # Always treat words as part of an ongoing text sequence by adding a leading space.
    return tokenizer(" " + text, return_attention_mask=False)["input_ids"]


def build_word_tokens(
    text: str,
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str,
    word_bbox: Tuple[float, float, float, float],
    writing_mode: WritingMode,
) -> Iterable[Token]:
    grouped_tokens_and_ids = tokenize_word(text, tokenizer, begin_of_word)
    if writing_mode == WritingMode.horizontal:
        yield from build_horizontal_tokens(text, word_bbox, grouped_tokens_and_ids)
    else:
        yield from build_vertical_tokens(text, word_bbox, grouped_tokens_and_ids)


def tokenize_word(text: str, tokenizer: PreTrainedTokenizer, begin_of_word: str) -> Iterable[List[Tuple[str, int]]]:
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    tokens = map(lambda t: re.sub(f"^{begin_of_word}", "", t), tokens)
    tokens_and_ids = zip(tokens, token_ids)
    tokens_and_ids = filter(lambda t: t[0], tokens_and_ids)
    grouped_tokens_and_ids = group_consecutive_escaped_tokens(tokens_and_ids)
    return grouped_tokens_and_ids


def build_horizontal_tokens(
    text: str, word_bbox: Tuple[float, float, float, float], grouped_tokens_and_ids: Iterable[List[Tuple[str, int]]]
) -> Iterable[Token]:
    left, top, right, bottom = word_bbox
    char_width = (right - left) / len(text)
    for group in grouped_tokens_and_ids:
        if len(group) == 1:
            t, tid = group[0]
            token, left = build_horizontal_token(left, top, bottom, char_width, t, tid, 1.0)
            yield token
        else:
            scale = compute_token_scale(group)
            for t, tid in group:
                token, left = build_horizontal_token(left, top, bottom, char_width, t, tid, scale)
                yield token


def build_horizontal_token(
    left: float, top: float, bottom: float, char_width: float, t: str, tid: int, scale: float
) -> Tuple[Token, float]:
    token_width = token_length(t) * char_width * scale
    bbox = (left, top, left + token_width, bottom)
    left += token_width
    token = Token(text=t, token_id=tid, bbox=bbox)
    return token, left


def build_vertical_tokens(
    text: str, word_bbox: Tuple[float, float, float, float], grouped_tokens_and_ids: Iterable[List[Tuple[str, int]]]
) -> Iterable[Token]:
    left, top, right, bottom = word_bbox
    char_width = (bottom - top) / len(text)
    for group in grouped_tokens_and_ids:
        if len(group) == 1:
            t, tid = group[0]
            token, top = build_vertical_token(left, top, right, char_width, t, tid, 1.0)
            yield token
        else:
            scale = compute_token_scale(group)
            for t, tid in group:
                token, top = build_vertical_token(left, top, right, char_width, t, tid, scale)
                yield token


def build_vertical_token(
    left: float, top: float, right: float, char_width: float, t: str, tid: int, scale: float
) -> Tuple[Token, float]:
    token_width = token_length(t) * char_width * scale
    bbox = (left, top, right, top + token_width)
    top += token_width
    token = Token(text=t, token_id=tid, bbox=bbox)
    return token, top


def compute_token_scale(group: List[Tuple[str, int]]) -> float:
    token_bytes = "".join(map(extract_bytes, (t for t, _ in group)))
    detok_text = bytes.fromhex(token_bytes).decode("utf-8")
    scale = len(detok_text) / len(group)
    return scale


def token_length(token: str) -> int:
    if is_escaped(token):
        return 1
    return len(token)


def extract_bytes(token: str) -> bytes:
    assert len(token) == 6
    return token[3:5]


def group_consecutive_escaped_tokens(tokens_and_ids: Iterable[Tuple[str, int]]) -> Iterable[List[Tuple[str, int]]]:
    get_idx = lambda idx_tok_id: idx_tok_id[0]
    get_token = lambda idx_tok_id: idx_tok_id[1][0]
    get_group_index = lambda idx_tok_id: -1 if is_escaped(get_token(idx_tok_id)) else get_idx(idx_tok_id)
    drop_index = lambda idx_tok_id: idx_tok_id[1]
    return (list(map(drop_index, g)) for _, g in groupby(enumerate(tokens_and_ids), key=get_group_index))


def is_escaped(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")
