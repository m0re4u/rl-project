import gym
import torch
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from QNetwork import QNetwork
from run_episode import run_episodes, train
from replay_memory import ReplayMemory


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
        # "Pendulum-v0"
    ]

    for env_name in envs:
        env = gym.envs.make(env_name)
        print(f"Doing: {env_name} - Observation space: {env.observation_space} - Action space: {env.action_space}")
        env.seed(seed)
        memory = ReplayMemory(mem_size)
        if type(env.action_space) == gym.spaces.Box:
            print(env.action_space.low.shape[0])
            model = QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.low.shape[0])
        elif type(env.action_space) == gym.spaces.Discrete:
            model = QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.n)
        episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
        plt.clf()
        plt.plot(episode_durations)
        plt.savefig(f"test-{env_name}.png")