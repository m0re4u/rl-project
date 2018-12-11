"""
Tests for generate.py
"""

import unittest

from mazenv.generate import prim


class GeneratorTest(unittest.TestCase):
    """
    Tests for maze generators.
    """
    # pylint: disable=R0914

    def test_prim(self):
        """
        Test the prim generator.
        """
        self._test_generators([
            lambda: prim((10, 10)),
            lambda: prim((8, 8, 8)),
            lambda: prim((3, 5, 8, 15))
        ])

    def _test_generators(self, gen_funcs):
        """
        Test desirable properties of maze generators.
        """
        for func in gen_funcs:
            seen = []
            for _ in range(10):
                maze = func()
                self.assertTrue(maze.solve())
                self.assertTrue(maze not in seen)
                seen.append(maze)


if __name__ == '__main__':
    unittest.main()
