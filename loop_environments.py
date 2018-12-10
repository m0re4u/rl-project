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
from gridworld import GridworldEnv



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
        "SimpleGridWorldEnv"
    ]

    for env_name in envs:
        if env_name == "SimpleGridWorldEnv":
            env = GridworldEnv()
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