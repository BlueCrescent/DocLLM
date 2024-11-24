from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Tuple
from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizer

from docllm.data.preprocessing.data_structure import WritingMode
from docllm.data.preprocessing.doc_data_from_hocr import Setup, build_word_tokens, parse_hocr_document, tokenize_word


def test_tokenize_word(tokenizer: PreTrainedTokenizer, begin_of_word: str):
    assert list(tokenize_word("Some", tokenizer, begin_of_word)) == [[("Some", 1)]]
    assert list(tokenize_word("Company", tokenizer, begin_of_word)) == [[("Comp", 2)], [("any", 3)]]
    assert list(tokenize_word("Hello", tokenizer, begin_of_word)) == [[("Hello", 4)]]
    assert list(tokenize_word("world!", tokenizer, begin_of_word)) == [[("world", 5)], [("!", 6)]]


def test_tokenize_word_with_escaped_tokens(tokenizer: PreTrainedTokenizer, begin_of_word: str):
    assert list(tokenize_word("覐", tokenizer, begin_of_word)) == [[("<0xe8>", 7), ("<0xa6>", 8), ("<0x90>", 9)]]


@pytest.mark.parametrize("writing_mode", [WritingMode.horizontal, WritingMode.vertical])
def test_build_word_tokens(
    tokenizer: PreTrainedTokenizer,
    begin_of_word: str,
    writing_mode: WritingMode,
    tokenizer_registries: Tuple[Dict[str, List[int]], Dict[int, str]],
):
    str_to_token_ids, token_id_to_token = tokenizer_registries
    word_box = (0, 0, 100, 100)
    for word_text, tids in str_to_token_ids.items():
        for tid, token in zip(tids, build_word_tokens(word_text, tokenizer, begin_of_word, word_box, writing_mode)):
            assert token.token_id == tid
            assert token.text == token_id_to_token[tid].strip(begin_of_word)
            assert is_contained_in(token.bbox, word_box)


def test_parse_hocr_document_saves_filename(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    assert parsed.filename == hocr_file


def test_parse_hocr_document_parses_all_pages(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    assert len(parsed.pages) == 2


def test_parse_hocr_document_parses_correct_page_dimensions(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_dimensions = [(2524, 3542), (2524, 3542)]
    for page, (width, height) in zip(parsed.pages, expected_dimensions):
        assert page.width == width and page.height == height


def test_parse_hocr_document_parses_all_blocks(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_num_blocks = [2, 2]
    for page, num_blocks in zip(parsed.pages, expected_num_blocks):
        assert len(page.blocks) == num_blocks


def test_parse_hocr_document_parses_correct_block_bboxes(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_bboxes = [(209, 222, 1158, 411), (1679, 533, 2493, 610), (209, 222, 1058, 311), (1679, 533, 2493, 610)]
    for block, bbox in zip((b for p in parsed.pages for b in p.blocks), expected_bboxes):
        assert block.bbox == bbox


def test_parse_hocr_document_parses_all_lines(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_num_lines = [[2, 1], [1, 1]]
    for page, page_num_lines in zip(parsed.pages, expected_num_lines):
        for block, block_num_lines in zip(page.blocks, page_num_lines):
            assert len(block.lines) == block_num_lines


def test_parse_hocr_document_parses_all_line_texts(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_line_texts = [[["Some Company", "Some Company"], ["Hello world!"]], [["Some Company"], ["Hello world!"]]]
    for page, page_texts in zip(parsed.pages, expected_line_texts):
        for block, block_texts in zip(page.blocks, page_texts):
            for line, text in zip(block.lines, block_texts):
                assert line.text == text


def test_parse_hocr_document_parses_all_words(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_num_words = [[[2, 2], [2]], [[2], [2]]]
    for page, page_num_words in zip(parsed.pages, expected_num_words):
        for block, block_num_words in zip(page.blocks, page_num_words):
            for line, word_num_words in zip(block.lines, block_num_words):
                assert len(line.words) == word_num_words


def test_parse_hocr_document_parses_all_tokens(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    expected_num_tokens = [[[[1, 2], [1, 2]], [[1, 2]]], [[[1, 2]], [[1, 2]]]]
    for page, page_num_tokens in zip(parsed.pages, expected_num_tokens):
        for block, block_num_tokens in zip(page.blocks, page_num_tokens):
            for line, line_num_tokens in zip(block.lines, block_num_tokens):
                for word, word_num_tokens in zip(line.words, line_num_tokens):
                    assert len(word.tokens) == word_num_tokens


@pytest.mark.parametrize("build_token_bboxes", (False,), indirect=True)
def test_token_bbox_is_word_bbox_if_build_token_bboxes_false(hocr_file: str, setup: Setup):
    parsed = parse_hocr_document(hocr_file, setup)
    for page in parsed.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    for token in word.tokens:
                        assert all(c_w == c_t for c_w, c_t in zip(word.bbox, token.bbox))


def is_contained_in(inner_box: Tuple[float, float, float, float], outer_box: Tuple[float, float, float, float]) -> bool:
    eps = 1e-8
    return all(
        [
            inner_box[0] >= outer_box[0],
            inner_box[1] >= outer_box[1],
            inner_box[2] <= outer_box[2] + eps,
            inner_box[3] <= outer_box[3] + eps,
        ]
    )


@pytest.fixture
def setup(tokenizer: PreTrainedTokenizer, build_token_bboxes: bool) -> Setup:
    return Setup(tokenizer=tokenizer, build_token_bboxes=build_token_bboxes)


@pytest.fixture
def build_token_bboxes(request) -> bool:
    if not hasattr(request, "param") or request.param is None:
        return False
    return request.param


@pytest.fixture
def tokenizer(tokenizer_registries: Tuple[Dict[str, List[int]], Dict[int, str]]) -> PreTrainedTokenizer:
    str_to_token_ids, token_id_to_token = tokenizer_registries
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.side_effect = lambda text, return_attention_mask: {"input_ids": str_to_token_ids[text]}
    tokenizer.convert_ids_to_tokens = MagicMock()
    tokenizer.convert_ids_to_tokens.side_effect = lambda token_ids, skip_special_tokens: [
        token_id_to_token[token_id] for token_id in token_ids
    ]
    return tokenizer


@pytest.fixture
def tokenizer_registries(begin_of_word: str) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    return {
        "Some": [1],
        "Company": [2, 3],
        "Hello": [4],
        "world!": [5, 6],
        "覐": [7, 8, 9],
    }, {
        1: begin_of_word + "Some",
        2: begin_of_word + "Comp",
        3: "any",
        4: begin_of_word + "Hello",
        5: begin_of_word + "world",
        6: "!",
        7: begin_of_word + "<0xe8>",
        8: "<0xa6>",
        9: "<0x90>",
    }


@pytest.fixture
def begin_of_word() -> str:
    return "▁"


@pytest.fixture
def word_box() -> Tuple[float, float, float, float]:
    return (0, 0, 100, 100)


@pytest.fixture
def hocr_file(hocr_data: str) -> Iterable[str]:
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".hocr") as file:
        file.write(hocr_data)
        file.flush()
        yield file.name


@pytest.fixture
def hocr_data() -> str:
    return """
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
        <title></title>
        <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
        <meta name="ocr-system" content="tesseract v5" />
        <meta name="ocr-capabilities" content="ocr_page ocrx_word" />
    </head>
    <body>
        <div class="ocr_page" id="page1" title="image document.jpeg; bbox 0 0 2524 3542; ppageno 0">
            <content_area id="block1">
                <paragraph id="par1">
                    <span class="ocr_line" id="line1" title="bbox 209 222 1058 311">
                        <span class="ocrx_word" id="word1" title="bbox 308 228 848 311; x_entity company_name 0; baseline 308 315.16 848 315.16; x_height 250.81; x_style sansSerif bold italic">Some</span>
                        <span class="ocrx_word" id="word2" title="bbox 875 227 1058 310; x_entity company_name 0; baseline 875 309.4 1058 309.4; x_height 250.03; x_style sansSerif bold italic">Company</span>
                    </span>
                    <span class="ocr_line" id="line2" title="bbox 308 227 1158 411">
                        <span class="ocrx_word" id="word1" title="bbox 308 228 848 311; x_entity company_name 0; baseline 308 315.16 848 315.16; x_height 250.81; x_style sansSerif bold italic">Some</span>
                        <span class="ocrx_word" id="word2" title="bbox 875 227 1058 310; x_entity company_name 0; baseline 875 309.4 1058 309.4; x_height 250.03; x_style sansSerif bold italic">Company</span>
                    </span>
                </paragraph>
            </content_area>
            <content_area id="block2">
                <paragraph id="par1">
                    <span class="ocr_line" id="line1" title="bbox 1679 533 2493 610">
                        <span class="ocrx_word" id="word1" title="bbox 1671 533 2293 610; x_entity ; baseline 1671 593.11 2293 593.11; x_height 547.71; x_style sansSerif bold none">Hello</span>
                        <span class="ocrx_word" id="word1" title="bbox 2293 533 2493 610; x_entity ; baseline 2293 593.11 2493 593.11; x_height 547.71; x_style sansSerif bold none">world!</span>
                    </span>
                </paragraph>
            </content_area>
        </div>
        <div class="ocr_page" id="page2" title="image document.jpeg; bbox 0 0 2524 3542; ppageno 1">
            <content_area id="block1">
                <paragraph id="par1">
                    <span class="ocr_line" id="line1" title="bbox 209 222 1058 311">
                        <span class="ocrx_word" id="word1" title="bbox 308 228 848 311; x_entity company_name 0; baseline 308 315.16 848 315.16; x_height 250.81; x_style sansSerif bold italic">Some</span>
                        <span class="ocrx_word" id="word2" title="bbox 875 227 1058 310; x_entity company_name 0; baseline 875 309.4 1058 309.4; x_height 250.03; x_style sansSerif bold italic">Company</span>
                    </span>
                </paragraph>
            </content_area>
            <content_area id="block2">
                <paragraph id="par1">
                    <span class="ocr_line" id="line1" title="bbox 1679 533 2493 610">
                        <span class="ocrx_word" id="word1" title="bbox 1671 533 2293 610; x_entity ; baseline 1671 593.11 2293 593.11; x_height 547.71; x_style sansSerif bold none">Hello</span>
                        <span class="ocrx_word" id="word1" title="bbox 2293 533 2493 610; x_entity ; baseline 2293 593.11 2493 593.11; x_height 547.71; x_style sansSerif bold none">world!</span>
                    </span>
                </paragraph>
            </content_area>
        </div>
    </body>
</html>
"""
