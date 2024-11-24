import os
from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel

Coord = float
Rect = Tuple[Coord, Coord, Coord, Coord]


class WritingMode(Enum):
    horizontal = 0
    vertical = 1


class Token(BaseModel):
    text: str
    token_id: int
    bbox: Rect


class Word(BaseModel):
    text: str
    tokens: List[Token]
    bbox: Rect


class Line(BaseModel):
    text: str
    words: List[Word]
    direction: Tuple[float, float]
    writing_mode: WritingMode
    bbox: Rect


class Block(BaseModel):
    lines: List[Line]
    bbox: Rect

    def get_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class Page(BaseModel):
    blocks: List[Block]
    width: Coord
    height: Coord

    def get_text(self, block_separator: str = "\n") -> str:
        return block_separator.join(block.get_text() for block in self.blocks)


class Document(BaseModel):
    pages: List[Page]
    filename: str

    def get_text(self, page: int, block_separator: str = "\n") -> str:
        return self.pages[page].get_text(block_separator=block_separator)

    def save_as_json(self, directory: str):
        basename = os.path.splitext(os.path.basename(self.filename))[0]
        filename = os.path.join(directory, basename + ".json")
        if os.path.exists(filename):
            raise IOError(f"Output file '{filename}' already exists. Cannot save document as JSON.")
        as_json = self.model_dump_json()
        with open(filename, "w") as f:
            f.write(as_json)
