import os
import gym
import torch
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

from QNetwork import QNetwork
from run_episode import run_episodes, train
from replay_memory import ReplayMemory, PrioritizedGreedyMemory, PrioritizedRankbasedMemory, PrioritizedProportionalMemory
from gridworld import GridworldEnv, WindyGridworldEnv

import mazenv

IMAGE_FOLDER = "images"
MAZE_FOLDER = "mazes"
RESULTS_FOLDER = "results"

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
    """
    Create Windy Gridworld with given wind strengths, shape and goal state.
    """
    winds = np.zeros(shape)

    for j in range(len(wind_strengths[i])):
        if isinstance(wind_strengths[i][j], tuple):
            winds[:, list(wind_strengths[i][j])] = j
        else:
            winds[:, wind_strengths[i][j]] = j
    env = WindyGridworldEnv(shape, winds, goal_state)
    return env

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
        return load_maze("LargeMazeWorld", (25, 25))
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

def create_mem(mem_name, mem_size):
    if mem_name == "RandomReplay":
        return ReplayMemory(mem_size)
    elif mem_name == "GreedyReplay":
        return PrioritizedGreedyMemory(mem_size)
    elif mem_name == "RankBasedReplay":
        return PrioritizedRankbasedMemory(mem_size)
    elif mem_name == "ProportionalReplay":
        return PrioritizedProportionalMemory(mem_size)
    else:
        raise NotImplementedError()


def plot_episode_durations(durs, mem_names, env_name):
    """
    Plot the episode durations (number of steps per episode).
    """
        # # "RandomReplay",
        # # "RankBasedReplay", # works on every env besides grid and maze worlds
        # "ProportionalReplay"

    plt.clf()
    for i, dur in enumerate(durs):
        if mem_names[i] == "RandomReplay":
            color = "blue"
        elif mem_names[i] == "RankBasedReplay":
            color = "yellow"
        elif mem_names[i] == "ProportionalReplay":
            color = "green"
        plt.plot(smooth(dur, 10), label=mem_names[i], color=color)
    plt.title('Episode durations per episode')
    plt.legend()
    plt.savefig(f"{env_name}_durations.png")


def plot_episode_rewards(rewards, mem_names, env_name):
    """
    Plot the episode reward (total reward of each episode).
    """
    plt.clf()
    for i, reward in enumerate(rewards):
        if mem_names[i] == "RandomReplay":
            color = "blue"
        elif mem_names[i] == "RankBasedReplay":
            color = "yellow"
        elif mem_names[i] == "ProportionalReplay":
            color = "green"
        elif mem_names[i] == "GreedyReplay":
            continue
        plt.plot(reward, label=mem_names[i], color=color)
    plt.title('Episode rewards per episode')
    plt.legend()
    plt.savefig(f"{env_name}_rewards.png")


def save_results(durations, rewards, env_name, mem_name):
    """
    Saves the results in a Pickle file.
    """

    name = f"{env_name}" + "_" + f"{mem_name}" + "_"
    file_name = os.path.join(RESULTS_FOLDER, name)

    with open(file_name + "durations.pkl", "wb") as f:
        pickle.dump(durations, f)

    with open(file_name + "rewards.pkl", "wb") as f:
        pickle.dump(rewards, f)


if __name__ == "__main__":
    num_episodes = 250
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
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    # All environments
    gridworlds = [
        # "SimpleGridWorld",
        # "MediumGridWorld",
        # "LargeGridWorld",
        # "HugeGridWorld",
        # "SimpleRectangleWorld",
        "LargeRectangleWorld"
    ]
    mazeworlds = [
        "SimpleMazeWorld",
        "MediumMazeWorld",
        "LargeMazeWorld"
    ]
    envs = [
        # "CartPole-v0",
        # "Acrobot-v1",
        # "MountainCar-v0",
        # "Pendulum-v0",
        *gridworlds,
        # *mazeworlds
    ]

    # All types of experience replay
    mems = [
        # "RandomReplay",
        # "RankBasedReplay", # works on every env besides grid and maze worlds
        "ProportionalReplay", # works on every env besides grid and maze worlds
        # "GreedyReplay", # FIXME
    ]

    for env_name in envs:
        ep_durations = []
        ep_rewards = []
        for mem_name in mems:
            print(f"Loading environment: {env_name} - ER method: {mem_name}")
            env = create_env(env_name)
            memory = create_mem(mem_name, mem_size)

            print(f"Doing: {env_name} - Observation space: {env.observation_space} - Action space: {env.action_space}")

            env.seed(seed)
            model = create_model(env)
            episode_durations, episode_rewards = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
            save_results(episode_durations, episode_rewards, env_name, mem_name)
            ep_durations.append(episode_durations)
            ep_rewards.append(episode_rewards)
        plot_episode_durations(ep_durations, mems, os.path.join(IMAGE_FOLDER, f"{env_name}"))
        plot_episode_rewards(ep_rewards, mems, os.path.join(IMAGE_FOLDER, f"{env_name}"))
