import random
from typing import List, Tuple


def shuffle_puzzle(goal: Tuple[int, ...], size: int, steps: int = 40) -> Tuple[int, ...]:
    board = list(goal)
    zero = board.index(0)

    for _ in range(steps):
        r, c = divmod(zero, size)
        neighbors = []

        if r > 0:
            neighbors.append(zero - size)
        if r < size - 1:
            neighbors.append(zero + size)
        if c > 0:
            neighbors.append(zero - 1)
        if c < size - 1:
            neighbors.append(zero + 1)

        nxt = random.choice(neighbors)
        board[zero], board[nxt] = board[nxt], board[zero]
        zero = nxt

    return tuple(board)


BASE_SUDOKU = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def generate_sudoku(holes: int) -> List[List[int]]:
    grid = [row[:] for row in BASE_SUDOKU]
    coords = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(coords)

    for i in range(min(holes, 81)):
        r, c = coords[i]
        grid[r][c] = 0

    return grid