from typing import Tuple


def misplaced_tiles(board: Tuple[int, ...], goal: Tuple[int, ...], size: int) -> int:
    return sum(1 for i, v in enumerate(board) if v != 0 and v != goal[i])


def manhattan_distance(board: Tuple[int, ...], goal: Tuple[int, ...], size: int) -> int:
    goal_pos = {value: idx for idx, value in enumerate(goal)}
    distance = 0

    for i, value in enumerate(board):
        if value == 0:
            continue
        gi = goal_pos[value]
        r1, c1 = divmod(i, size)
        r2, c2 = divmod(gi, size)
        distance += abs(r1 - r2) + abs(c1 - c2)

    return distance


def linear_conflict(board: Tuple[int, ...], goal: Tuple[int, ...], size: int) -> int:
    goal_pos = {value: idx for idx, value in enumerate(goal)}
    conflict = 0

    for row in range(size):
        row_tiles = []
        for col in range(size):
            idx = row * size + col
            tile = board[idx]
            if tile != 0:
                gidx = goal_pos[tile]
                grow, gcol = divmod(gidx, size)
                if grow == row:
                    row_tiles.append((col, gcol))
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                if row_tiles[i][1] > row_tiles[j][1]:
                    conflict += 2

    for col in range(size):
        col_tiles = []
        for row in range(size):
            idx = row * size + col
            tile = board[idx]
            if tile != 0:
                gidx = goal_pos[tile]
                grow, gcol = divmod(gidx, size)
                if gcol == col:
                    col_tiles.append((row, grow))
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                if col_tiles[i][1] > col_tiles[j][1]:
                    conflict += 2

    return conflict


def custom_heuristic(board: Tuple[int, ...], goal: Tuple[int, ...], size: int) -> int:
    return manhattan_distance(board, goal, size) + linear_conflict(board, goal, size)


def get_heuristic(name: str):
    name = name.lower().strip()
    if name == "fichas fuera de lugar":
        return misplaced_tiles
    if name == "distancia manhattan":
        return manhattan_distance
    if name == "heurística personalizada" or name == "heuristica personalizada":
        return custom_heuristic
    return manhattan_distance