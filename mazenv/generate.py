"""
Routines for generating random mazes.
"""

import random

import numpy as np

from .maze import Maze
from . import _util as util


def prim(shape):
    """
    Use randomized Prim's algorithm to generate a maze.
    """
    start_pos = tuple([random.randrange(x) for x in shape])
    walls = np.ones(shape, dtype='bool')
    walls[start_pos] = False
    maze = Maze(walls, start_pos=start_pos)

    edges = [x for x in util.iterate_neighbors(start_pos) if maze.in_bounds(x)]
    visited = set(edges)
    visited.add(start_pos)

    while edges:
        pos = random.choice(edges)
        edges.remove(pos)
        if len(list(maze.neighboring_non_walls(pos))) > 1:
            continue
        maze.walls[pos] = False
        for neighbor in util.iterate_neighbors(pos):
            if maze.in_bounds(neighbor) and neighbor not in visited:
                visited.add(neighbor)
                edges.append(neighbor)

    maze.end_pos = random.choice(list(maze.spaces()))
    return maze
