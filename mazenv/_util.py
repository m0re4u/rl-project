"""
Utilities for grids.
"""

import itertools


def iterate_positions(shape):
    """
    Generate an iterable of all valid indices in a tensor.
    """
    return itertools.product(*[range(n) for n in shape])


def iterate_neighbors(pos):
    """
    Iterate through all the single-axis neighbors of a
    coordinate tuple.

    Order is deterministic.
    """
    for axis, val in enumerate(pos):
        for offset in [-1, 1]:
            yield pos[:axis] + (val+offset,) + pos[axis+1:]


def shape_contains(shape, pos):
    """
    Check if a tensor of the given shape contains the
    position.
    """
    assert len(shape) == len(pos)
    for val, max_val in zip(pos, shape):
        if val < 0 or val >= max_val:
            return False
    return True
