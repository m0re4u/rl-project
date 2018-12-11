"""
Tests for env.py
"""

import unittest

from mazenv.env import CURRENT_CELL_FIELD, Env, HorizonEnv
from mazenv.maze import parse_2d_maze


class EnvTest(unittest.TestCase):
    """
    Tests for raw environments.
    """
    # pylint: disable=R0914

    def test_2d_trajectory(self):
        """
        Test that all the positions, rewards, and done
        signals are correct throughout a trajectory.
        """
        maze = parse_2d_maze('w...w\n' +
                             '..w.w\n' +
                             'Awx..\n' +
                             '..w..')
        env = Env(maze)
        env.reset()
        act_nop, act_up, act_down, act_left, act_right = [0, 1, 2, 3, 4]
        actions = [act_down, act_right, act_right, act_nop, act_left,
                   act_left, act_up, act_up, act_up, act_right, act_up,
                   act_right, act_right, act_down, act_left, act_down,
                   act_nop, act_left]
        positions = [(3, 0), (3, 1), (3, 1), (3, 1), (3, 0),
                     (3, 0), (2, 0), (1, 0), (1, 0), (1, 1), (0, 1),
                     (0, 2), (0, 3), (1, 3), (1, 3), (2, 3),
                     (2, 3), (2, 2)]
        dones = [False]*(len(positions)-1) + [True]
        for act, expected_pos, expected_done in zip(actions, positions, dones):
            obs, rew, done, _ = env.step(act)
            self.assertEqual(done, expected_done)
            if done:
                self.assertEqual(rew, 0.0)
            else:
                self.assertEqual(rew, -1.0)
            self._assert_unique_cell(maze, obs, expected_pos, CURRENT_CELL_FIELD)

    def _assert_unique_cell(self, maze, obs, pos, cell_idx):
        """
        Assert that the cell is set at the position in the
        observation, and nowhere else in the observation.
        """
        for some_pos in maze.positions():
            cell_val = obs[some_pos][cell_idx]
            if some_pos == pos:
                self.assertEqual(cell_val, 1)
            else:
                self.assertEqual(cell_val, 0)


class HorizonEnvTest(unittest.TestCase):
    """
    Tests for limited-horizon environments.
    """

    def test_2d_observations(self):
        """
        Test observations on a 2-D environment.
        """
        maze = parse_2d_maze('w...w\n' +
                             '..w.w\n' +
                             'Awx..\n' +
                             '.....')
        env = HorizonEnv(maze, horizon=1)
        env.reset()
        act_up, act_down, act_left, act_right = [1, 2, 3, 4]
        actions = [act_left, act_down, act_right, act_right, act_up]
        obses = [
            _centered_horizon_obs('w..\nwAw\nw..'),
            _centered_horizon_obs('wAw\nw..\nwww'),
            _centered_horizon_obs('Awx\n...\nwww'),
            _centered_horizon_obs('wx.\n...\nwww'),
            _centered_horizon_obs('.w.\nwx.\n...')
        ]
        for action, expected_obs in zip(actions, obses):
            obs, _, _, _ = env.step(action)
            self.assertTrue((obs == expected_obs).all())


def _centered_horizon_obs(maze_str):
    maze = parse_2d_maze(maze_str)

    center = tuple(x // 2 for x in maze.shape)
    old_start = maze.start_pos
    maze.start_pos = center
    env = Env(maze)
    env.reset()
    maze.start_pos = old_start

    obs, _, _, _ = env.step(0)
    return obs


if __name__ == '__main__':
    unittest.main()
