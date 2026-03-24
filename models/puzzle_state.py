from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(order=True)
class PuzzleNode:
    priority: int
    board: Tuple[int, ...] = field(compare=False)
    g: int = field(compare=False, default=0)
    h: int = field(compare=False, default=0)
    parent: Optional["PuzzleNode"] = field(compare=False, default=None)
    move: Optional[str] = field(compare=False, default=None)

    def f(self) -> int:
        return self.g + self.h