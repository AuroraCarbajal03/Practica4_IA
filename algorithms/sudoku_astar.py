import copy
import heapq
from typing import List, Optional, Tuple

from models.sudoku_state import SudokuNode
from utils.metrics import MetricTracker
from utils.validators import is_complete_sudoku, is_valid_sudoku_move


def find_best_empty_cell(grid: List[List[int]]) -> Optional[Tuple[int, int, List[int]]]:
    best = None
    best_candidates = None

    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                candidates = []
                for num in range(1, 10):
                    if is_valid_sudoku_move(grid, r, c, num):
                        candidates.append(num)

                if best is None or len(candidates) < len(best_candidates):
                    best = (r, c, candidates)
                    best_candidates = candidates

                if len(candidates) == 1:
                    return best

    return best


def sudoku_heuristic(grid: List[List[int]]) -> int:
    return sum(1 for r in range(9) for c in range(9) if grid[r][c] == 0)


def grid_to_tuple(grid: List[List[int]]):
    return tuple(tuple(row) for row in grid)


def reconstruct_path(node: SudokuNode):
    path = []
    while node:
        path.append(node.grid)
        node = node.parent
    path.reverse()
    return path


def astar_sudoku(initial_grid: List[List[int]]):
    tracker = MetricTracker()
    tracker.start()

    start_h = sudoku_heuristic(initial_grid)
    start_node = SudokuNode(priority=start_h, grid=copy.deepcopy(initial_grid), g=0, h=start_h)

    open_heap = [start_node]
    visited = set()
    nodes_expanded = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        key = grid_to_tuple(current.grid)

        if key in visited:
            continue
        visited.add(key)

        nodes_expanded += 1

        if is_complete_sudoku(current.grid):
            tracker.stop()
            return {
                "solved": True,
                "solution": current.grid,
                "path": reconstruct_path(current),
                "time_ms": tracker.elapsed_ms,
                "memory_kb": tracker.peak_kb,
                "nodes": nodes_expanded,
                "steps": max(0, len(reconstruct_path(current)) - 1),
            }

        choice = find_best_empty_cell(current.grid)
        if choice is None:
            continue

        r, c, candidates = choice

        for num in candidates:
            new_grid = copy.deepcopy(current.grid)
            new_grid[r][c] = num
            h = sudoku_heuristic(new_grid)
            new_node = SudokuNode(
                priority=(current.g + 1 + h),
                grid=new_grid,
                g=current.g + 1,
                h=h,
                parent=current,
            )
            heapq.heappush(open_heap, new_node)

    tracker.stop()
    return {
        "solved": False,
        "solution": None,
        "path": [],
        "time_ms": tracker.elapsed_ms,
        "memory_kb": tracker.peak_kb,
        "nodes": nodes_expanded,
        "steps": 0,
    }