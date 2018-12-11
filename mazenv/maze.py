"""
APIs for using and manipulating mazes.
"""

from collections import deque

import numpy as np

from . import _util as util


class Maze:
    """
    A rectangular maze.

    Mazes are grids of cell.
    Each cell is either a wall or a space.
    Two spaces, as designated by the start_pos and end_pos
    attributes, represent the start and end cells.
    """

    def __init__(self, wall_array, start_pos=None, end_pos=None):
        """
        Create a maze from the given specifications.

        The wall_array should be a boolean numpy array
        indicating which cells are walls.

        The start_pos and end_pos, if not None, should be
        tuples specifying the coordintaes of the start and
        end spaces.
        """
        if start_pos:
            assert not wall_array[start_pos]
        if end_pos:
            assert not wall_array[end_pos]
        self.walls = wall_array
        self.start_pos = start_pos
        self.end_pos = end_pos

    @property
    def shape(self):
        """
        Get the shape of the maze as a tuple.
        """
        return self.walls.shape

    def in_bounds(self, pos):
        """
        Check that a coordinate tuple is in bounds.
        """
        return util.shape_contains(self.shape, pos)

    def positions(self):
        """
        Return an iterable of all the valid positions on
        the maze, in a deterministic order.
        """
        return util.iterate_positions(self.shape)

    def is_wall(self, pos):
        """
        Check if the position is a wall.

        Always returns True for out-of-bounds positions.
        """
        if self.in_bounds(pos):
            return self.walls[pos]
        return True

    def solve(self):
        """
        Compute an optimal path from the start position to
        the end position.

        Returns a list of tuples, one for each position.
        This list includes the start and end positions.

        If no solution exists, None is returned.
        """
        assert self.start_pos
        if not self.end_pos:
            return None
        path_queue = deque([[(self.start_pos)]])
        visited = set()
        while path_queue:
            path = path_queue.popleft()
            for neighbor in self.neighboring_non_walls(path[-1]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_path = path.copy()
                new_path.append(neighbor)
                if neighbor == self.end_pos:
                    return new_path
                path_queue.append(new_path)
        return None

    def neighboring_non_walls(self, pos):
        """
        Find the non-walls neighboring the given position.
        """
        for neighbor in util.iterate_neighbors(pos):
            if not self.is_wall(neighbor):
                yield neighbor

    def spaces(self):
        """
        Iterate through all the spaces with no start or
        end positions.
        """
        for pos in self.positions():
            if not (self.is_wall(pos) or
                    pos == self.start_pos or
                    pos == self.end_pos):
                yield pos

    def __str__(self):
        if len(self.walls.shape) != 2:
            return str(self.walls)
        row_strs = []
        for row in range(self.walls.shape[0]):
            row_str = ''
            for col in range(self.walls.shape[1]):
                pos = (row, col)
                if self.is_wall(pos):
                    row_str += 'w'
                elif self.start_pos == pos:
                    row_str += 'A'
                elif self.end_pos == pos:
                    row_str += 'x'
                else:
                    row_str += '.'
            row_strs.append(row_str)
        return '\n'.join(row_strs)

    def __eq__(self, other):
        return (self.start_pos == other.start_pos and
                self.end_pos == other.end_pos and
                self.shape == other.shape and
                (self.walls == other.walls).all())


def parse_2d_maze(maze_str):
    """
    Decode a 2-D maze from its string representation.
    """
    lines = [x.strip() for x in maze_str.strip().split('\n')]
    num_rows = len(lines)
    if num_rows == 0:
        raise ValueError('must have at least one row')
    num_cols = len(lines[0])
    walls = []
    start_pos = None
    end_pos = None
    for row, row_str in enumerate(lines):
        if len(row_str) != num_cols:
            raise ValueError('row length should be %d but got %d' %
                             (num_cols, len(row)))
        sub_walls = []
        for col, cell_str in enumerate(row_str):
            if cell_str == 'w':
                sub_walls.append(True)
            else:
                sub_walls.append(False)
            if cell_str == 'A':
                if start_pos:
                    raise ValueError('more than one start state')
                start_pos = (row, col)
            elif cell_str == 'x':
                if end_pos:
                    raise ValueError('more than one end state')
                end_pos = (row, col)
        walls.append(sub_walls)
    return Maze(np.array(walls), start_pos=start_pos, end_pos=end_pos)
