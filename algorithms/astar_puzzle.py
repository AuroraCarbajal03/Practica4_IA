import heapq
from typing import Dict, List, Tuple

from models.puzzle_state import PuzzleNode
from utils.metrics import MetricTracker


def get_neighbors(board: Tuple[int, ...], size: int):
    zero = board.index(0)
    r, c = divmod(zero, size)
    moves = []

    if r > 0:
        moves.append(("Arriba", zero - size))
    if r < size - 1:
        moves.append(("Abajo", zero + size))
    if c > 0:
        moves.append(("Izquierda", zero - 1))
    if c < size - 1:
        moves.append(("Derecha", zero + 1))

    result = []
    for move_name, nxt in moves:
        new_board = list(board)
        new_board[zero], new_board[nxt] = new_board[nxt], new_board[zero]
        result.append((move_name, tuple(new_board)))

    return result


def reconstruct_path(node: PuzzleNode):
    path = []
    moves = []

    while node:
        path.append(node.board)
        if node.move is not None:
            moves.append(node.move)
        node = node.parent

    path.reverse()
    moves.reverse()
    return path, moves


def astar_puzzle(start: Tuple[int, ...], goal: Tuple[int, ...], size: int, heuristic_fn):
    tracker = MetricTracker()
    tracker.start()

    open_heap = []
    h0 = heuristic_fn(start, goal, size)
    start_node = PuzzleNode(priority=h0, board=start, g=0, h=h0, parent=None, move=None)
    heapq.heappush(open_heap, start_node)

    g_score: Dict[Tuple[int, ...], int] = {start: 0}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        current = heapq.heappop(open_heap)

        if current.board in closed:
            continue

        closed.add(current.board)
        nodes_expanded += 1

        if current.board == goal:
            tracker.stop()
            path, moves = reconstruct_path(current)
            return {
                "solved": True,
                "path": path,
                "moves": moves,
                "time_ms": tracker.elapsed_ms,
                "memory_kb": tracker.peak_kb,
                "nodes": nodes_expanded,
                "steps": max(0, len(path) - 1),
            }

        for move_name, neighbor in get_neighbors(current.board, size):
            if neighbor in closed:
                continue

            tentative_g = current.g + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = heuristic_fn(neighbor, goal, size)
                neighbor_node = PuzzleNode(
                    priority=tentative_g + h,
                    board=neighbor,
                    g=tentative_g,
                    h=h,
                    parent=current,
                    move=move_name,
                )
                heapq.heappush(open_heap, neighbor_node)

    tracker.stop()
    return {
        "solved": False,
        "path": [],
        "moves": [],
        "time_ms": tracker.elapsed_ms,
        "memory_kb": tracker.peak_kb,
        "nodes": nodes_expanded,
        "steps": 0,
    }