from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(order=True)
class SudokuNode:
    priority: int
    grid: List[List[int]] = field(compare=False)
    g: int = field(compare=False, default=0)
    h: int = field(compare=False, default=0)
    parent: Optional["SudokuNode"] = field(compare=False, default=None)