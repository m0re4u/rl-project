import os
import gym
import torch
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from QNetwork import QNetwork
from run_episode import run_episodes, train
from replay_memory import ReplayMemory
from gridworld import GridworldEnv, WindyGridworldEnv

import mazenv

IMAGE_FOLDER = "images"
MAZE_FOLDER = "mazes"

def smooth(x, N):
    """
    Smooth the input data :x over :N neighbouring values
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_maze(maze_name, maze_size=(8,8)):
    """
    Load maze environment with given :maze_name. If the maze does not exist as a
    file, create a new one.
    """
    maze_filename = os.path.join(MAZE_FOLDER,f"{maze_name}.txt")
    if not os.path.exists(maze_filename):
        print(f"Creating new maze for {maze_name}")

        maze = mazenv.prim(maze_size)
        with open(maze_filename, 'w+') as f:
            print(maze)
            f.write(str(maze))
    else:
        print(f"Loaded maze {maze_name} from file")
        with open(maze_filename, 'r') as f:
            maze = mazenv.parse_2d_maze(f.read())
        print(maze)
    env = mazenv.Env(maze)
    return env


def create_windy_gridworld(shape, wind_strengths, goal):
    pass
    # shapes = [(7, 10), (10, 10), (20, 5), (30, 15), (8, 12)]
    # wind_strengths = [(),
    #                     ,
    #                     ,
    #                     ,
    #                     ((0, 1, 2, 3, 10, 11), (5), (4), (6, 7), (8, 9))]
    # goal_states = [(3, 7), (9, 8), (12, 3), (7, 8), (8, 12)]
    # worlds = []

    # for i in range(len(shapes)):
    #     winds = np.zeros(shapes[i])

    #     for j in range(len(wind_strengths[i])):
    #         if isinstance(wind_strengths[i][j], tuple):
    #             winds[:, list(wind_strengths[i][j])] = j
    #         else:
    #             winds[:, wind_strengths[i][j]] = j

    #     worlds.append(WindyGridworldEnv(shapes[i], winds, goal_states[i]))
    # return worlds


def create_env(env_name):
    """
    Create/load the environment associated with :env_name
    """
    if env_name == "SimpleGridWorld":
        return GridworldEnv()
    elif env_name == "MediumGridWorld":
        return GridworldEnv(shape=[10,10])
    elif env_name == "LargeGridWorld":
        return GridworldEnv(shape=[20,20])
    elif env_name == "HugeGridWorld":
        return GridworldEnv(shape=[31,31])
    elif env_name == "SimpleRectangleWorld":
        return GridworldEnv(shape=[10,4])
    elif env_name == "LargeRectangleWorld":
        return GridworldEnv(shape=[15,31])
    elif env_name == "SimpleMazeWorld":
        return load_maze("SimpleMazeWorld")
    elif env_name == "MediumMazeWorld":
        return load_maze("MediumMazeWorld", (15, 15))
    elif env_name == "LargeMazeWorld":
        return load_maze("MediumMazeWorld", (25, 25))
    elif env_name == "SimpleWindyGridWorld":
        return create_windy_gridworld((10,10), ((0, 1, 6, 8, 9), (2, 7), (3), (4, 5)), (9,8))
    elif env_name == "MediumRectangularWindyGridWorld":
        return create_windy_gridworld((20,5), ((0, 1), (2, 3), (4)), (12, 3))
    elif env_name == "LargeRectangularWindyGridWorld":
        return create_windy_gridworld((30,15), ((5, 6, 7, 8, 12, 14), (0, 1, 2, 3, 13), (4), (9, 10, 11)), (7, 8))
    else:
        return gym.envs.make(env_name)


def create_model(env):
    """
    Create a model depending on the type of environment :env. Note that although it runs technically,
    the Box to Box (Continuous to Continuous) version doesn't really work.
    """
    if type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Box:
        return QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.low.shape[0])
    elif type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Box:
        return QNetwork(env.observation_space.low.shape[0], num_hidden, env.action_space.n)
    elif type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Discrete:
        return QNetwork(env.observation_space.n, num_hidden, env.action_space.low.shape[0])
    elif type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Discrete:
        return QNetwork(env.observation_space.n, num_hidden, env.action_space.n)
    else:
        raise NotImplementedError()


def plot_episode_durations(durs, env_name):
    """
    Plot the episode durations (number of steps per episode).
    """
    # And see the results
    plt.clf()
    plt.plot(smooth(durs, 10))
    plt.title('Episode durations per episode')
    plt.savefig(f"{env_name}_durations.png")


def plot_episode_rewards(rewards, env_name):
    """
    Plot the episode reward (total reward of each episode).
    """
    # And see the results
    plt.clf()
    plt.plot(rewards)
    plt.title('Episode rewards per episode')
    plt.savefig(f"{env_name}_rewards.png")


if __name__ == "__main__":
    num_episodes = 100
    batch_size = 64
    discount_factor = 0.8
    mem_size = 10000
    learn_rate = 1e-3
    num_hidden = 128
    seed = 42  # The answer to everything!

    random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(IMAGE_FOLDER):
        os.mkdir(IMAGE_FOLDER)
    if not os.path.exists(MAZE_FOLDER):
        os.mkdir(MAZE_FOLDER)

    gridworlds = [
        "SimpleGridWorld",
        "MediumGridWorld",
        "LargeGridWorld",
        "HugeGridWorld",
        "SimpleRectangleWorld",
        "LargeRectangleWorld"
    ]
    mazeworlds = [
        "SimpleMazeWorld",
        "MediumMazeWorld",
        "LargeMazeWorld"
    ]
    envs = [
        "CartPole-v0",
        "Acrobot-v1",
        "MountainCar-v0",
        "Pendulum-v0",
        *gridworlds,
        *mazeworlds
    ]

    for env_name in envs:
        print(f"Name: {env_name}")
        env = create_env(env_name)

        print(f"Doing: {env_name} - Observation space: {env.observation_space} - Action space: {env.action_space}")

        env.seed(seed)
        memory = ReplayMemory(mem_size)
        model = create_model(env)
        episode_durations, episode_rewards = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

        plot_episode_durations(episode_durations, os.path.join(IMAGE_FOLDER, env_name))
        plot_episode_rewards(episode_rewards, os.path.join(IMAGE_FOLDER, env_name))
