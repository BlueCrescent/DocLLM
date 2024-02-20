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


class Page(BaseModel):
    blocks: List[Block]
    width: Coord
    height: Coord


class Document(BaseModel):
    pages: List[Page]
    filename: str
