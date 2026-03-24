from typing import List


def is_valid_sudoku_move(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    for c in range(9):
        if grid[row][c] == num and c != col:
            return False

    for r in range(9):
        if grid[r][col] == num and r != row:
            return False

    box_r = (row // 3) * 3
    box_c = (col // 3) * 3

    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if grid[r][c] == num and (r, c) != (row, col):
                return False

    return True


def is_complete_sudoku(grid: List[List[int]]) -> bool:
    for r in range(9):
        for c in range(9):
            value = grid[r][c]
            if value == 0:
                return False
            if not is_valid_sudoku_move(grid, r, c, value):
                return False
    return True