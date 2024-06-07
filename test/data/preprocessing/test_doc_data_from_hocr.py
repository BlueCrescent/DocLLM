from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizer

from docllm.data.preprocessing.data_structure import WritingMode
from docllm.data.preprocessing.doc_data_from_hocr import _build_word_tokens, _tokenize_word


def test_tokenize_word(tokenizer: PreTrainedTokenizer, begin_of_word: str):
    assert list(_tokenize_word("Some", tokenizer, begin_of_word)) == [[("Some", 1)]]
    assert list(_tokenize_word("Company", tokenizer, begin_of_word)) == [[("Comp", 2)], [("any", 3)]]
    assert list(_tokenize_word("Hello", tokenizer, begin_of_word)) == [[("Hello", 4)]]
    assert list(_tokenize_word("world!", tokenizer, begin_of_word)) == [[("world", 5)], [("!", 6)]]


def test_tokenize_word_with_escaped_tokens(tokenizer: PreTrainedTokenizer, begin_of_word: str):
    assert list(_tokenize_word("覐", tokenizer, begin_of_word)) == [[("<0x89>", 7), ("<0x90>", 8)]]


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
        for tid, token in zip(tids, _build_word_tokens(word_text, tokenizer, begin_of_word, word_box, writing_mode)):
            assert token.token_id == tid
            assert token.text == token_id_to_token[tid].strip(begin_of_word)
            assert is_contained_in(token.bbox, word_box)


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


# @pytest.fixture(params=[WritingMode.horizontal, WritingMode.vertical])
# def build_word_tokens(
#     tokenizer: PreTrainedTokenizer,
#     begin_of_word: str,
#     word_box: Tuple[float, float, float, float],
#     request: pytest.FixtureRequest,
# ):
#     word_text = "Some"
#     tokens = list(_build_word_tokens(word_text, tokenizer, begin_of_word, word_box, request.param))


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
        "覐": [7, 8],
    }, {
        1: begin_of_word + "Some",
        2: begin_of_word + "Comp",
        3: "any",
        4: begin_of_word + "Hello",
        5: begin_of_word + "world",
        6: "!",
        7: begin_of_word + "<0x89>",
        8: "<0x90>",
    }


@pytest.fixture
def begin_of_word() -> str:
    return "▁"


@pytest.fixture
def word_box() -> Tuple[float, float, float, float]:
    return (0, 0, 100, 100)


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
                    <span class="ocr_line" id="line2" title="bbox 209 222 1058 311">
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
