import copy
import math
import random
from typing import List, Tuple

from utils.metrics import MetricTracker
from utils.validators import is_complete_sudoku


def subgrid_cells(box_row: int, box_col: int):
    cells = []
    for r in range(box_row * 3, box_row * 3 + 3):
        for c in range(box_col * 3, box_col * 3 + 3):
            cells.append((r, c))
    return cells


def initialize_solution(grid: List[List[int]], fixed: List[List[bool]]) -> List[List[int]]:
    new_grid = copy.deepcopy(grid)

    for box_r in range(3):
        for box_c in range(3):
            present = set()
            empty_cells = []

            for r, c in subgrid_cells(box_r, box_c):
                if new_grid[r][c] != 0:
                    present.add(new_grid[r][c])
                else:
                    empty_cells.append((r, c))

            missing = [n for n in range(1, 10) if n not in present]
            random.shuffle(missing)

            for (r, c), value in zip(empty_cells, missing):
                new_grid[r][c] = value

    return new_grid


def score_solution(grid: List[List[int]]) -> int:
    score = 0

    for r in range(9):
        score += len(set(grid[r]))

    for c in range(9):
        score += len(set(grid[r][c] for r in range(9)))

    return score


def random_neighbor(grid: List[List[int]], fixed: List[List[bool]]) -> List[List[int]]:
    new_grid = copy.deepcopy(grid)

    box_r = random.randint(0, 2)
    box_c = random.randint(0, 2)

    candidates = [(r, c) for r, c in subgrid_cells(box_r, box_c) if not fixed[r][c]]

    if len(candidates) >= 2:
        (r1, c1), (r2, c2) = random.sample(candidates, 2)
        new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]

    return new_grid


def simulated_annealing_sudoku(initial_grid: List[List[int]], max_iter: int = 50000, temp: float = 5.0, cooling: float = 0.9995):
    tracker = MetricTracker()
    tracker.start()

    fixed = [[initial_grid[r][c] != 0 for c in range(9)] for r in range(9)]
    current = initialize_solution(initial_grid, fixed)
    current_score = score_solution(current)
    best = copy.deepcopy(current)
    best_score = current_score

    history = [copy.deepcopy(current)]

    for i in range(max_iter):
        if is_complete_sudoku(current):
            tracker.stop()
            return {
                "solved": True,
                "solution": current,
                "path": history,
                "time_ms": tracker.elapsed_ms,
                "memory_kb": tracker.peak_kb,
                "nodes": i + 1,
                "steps": i + 1,
            }

        candidate = random_neighbor(current, fixed)
        candidate_score = score_solution(candidate)

        delta = candidate_score - current_score

        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
            current = candidate
            current_score = candidate_score
            if len(history) < 200:
                history.append(copy.deepcopy(current))

        if current_score > best_score:
            best = copy.deepcopy(current)
            best_score = current_score

        temp *= cooling

        if best_score == 162 and is_complete_sudoku(best):
            tracker.stop()
            return {
                "solved": True,
                "solution": best,
                "path": history,
                "time_ms": tracker.elapsed_ms,
                "memory_kb": tracker.peak_kb,
                "nodes": i + 1,
                "steps": i + 1,
            }

    tracker.stop()
    return {
        "solved": is_complete_sudoku(best),
        "solution": best,
        "path": history,
        "time_ms": tracker.elapsed_ms,
        "memory_kb": tracker.peak_kb,
        "nodes": max_iter,
        "steps": max_iter,
    }