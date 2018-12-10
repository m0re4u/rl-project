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


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == "__main__":
    num_episodes = 100
    batch_size = 64
    discount_factor = 0.8
    mem_size = 10000
    learn_rate = 1e-3
    num_hidden = 128
    seed = 42  # This is not randomly chosen

    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    random.seed(seed)
    torch.manual_seed(seed)

    envs = [
        "CartPole-v0",
        "Acrobot-v1",
        "MountainCar-v0",
        "Pendulum-v0",
        "SimpleGridWorldEnv",
        "WindyGridWorldEnv"
    ]

    for env_name in envs:
        if env_name == "SimpleGridWorldEnv":
            env = GridworldEnv()
        elif env_name == "WindyGridWorldEnv":
            shapes = [(7, 10), (10, 10), (20, 5), (30, 15), (8, 12)]
            wind_strengths = [((0, 1, 2, 9), (3, 4, 5, 8), (6, 7)),
                              ((0, 1, 6, 8, 9), (2, 7), (3), (4, 5)),
                              ((0, 1), (2, 3), (4)),
                              ((5, 6, 7, 8, 12, 14), (0, 1, 2, 3, 13), (4), (9, 10, 11)),
                              ((0, 1, 2, 3, 10, 11), (5), (4), (6, 7), (8, 9))]
            goal_states = [(3, 7), (9, 8), (12, 3), (7, 8), (8, 12)]

            for i in range(len(shapes)):
                winds = np.zeros(shapes[i])

                for j in range(len(wind_strengths[i])):
                    if isinstance(wind_strengths[i][j], tuple):
                        winds[:, list(wind_strengths[i][j])] = j
                    else:
                        winds[:, wind_strengths[i][j]] = j
                env = WindyGridworldEnv(shapes[i], winds, goal_states[i])

        else:
            env = gym.envs.make(env_name)
        print(f"Doing: {env_name} - Observation space: {env.observation_space} - Action space: {env.action_space}")
        env.seed(seed)
        memory = ReplayMemory(mem_size)
        if type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Box:
            model = QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.low.shape[0])
        elif type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Box:
            model = QNetwork(env.observation_space.low.shape[0], num_hidden, env.action_space.n)
        elif type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Discrete:
            model = QNetwork(env.observation_space.n, num_hidden, env.action_space.low.shape[0])
        elif type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Discrete:
            model = QNetwork(env.observation_space.n, num_hidden, env.action_space.n)
        else:
            raise NotImplementedError()
        episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

        # And see the results
        plt.clf()
        plt.plot(smooth(episode_durations, 10))
        plt.title('Episode durations per episode')
        plt.savefig(f"test-{env_name}.png")
