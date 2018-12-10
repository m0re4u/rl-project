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
    """
    Smooth the input data :x over :N neighbouring values
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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

    gridworlds = ["SimpleGridWorld", "MediumGridWorld", "LargeGridWorld", "HugeGridWorld"]
    envs = [
        # "CartPole-v0",
        # "Acrobot-v1",
        # "SimpleGridWorld",
        *gridworlds
    ]

    for env_name in envs:
        env = create_env(env_name)
        print(f"Doing: {env_name} - Observation space: {env.observation_space} - Action space: {env.action_space}")

        env.seed(seed)
        memory = ReplayMemory(mem_size)
        model = create_model(env)
        episode_durations, episode_rewards = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
        plot_episode_durations(episode_durations, env_name)
        plot_episode_rewards(episode_rewards, env_name)