import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

from replay_memory import ReplayMemory
from QNetwork import QNetwork

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def get_epsilon(it):
    N = 1000
    ep = 1 - (1*it/N)
    if ep < 0.05:
        return 0.05
    else:
        return ep


def select_action(model, state, epsilon):
    p = random.random()
    if p > epsilon:
        # Select greedy action
        with torch.no_grad():
            _, ind = model(torch.Tensor(state)).max(0)
            return ind.item()
    else:
        # Select random action
        return random.choice([0,1])


def compute_q_val(model, state, action):
    x = model(state)
    return torch.gather(x, 1, action.view(-1, 1)).view(-1)

def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    val, _ = model(next_state).max(1)
    val[done] = 0
    return reward + (discount_factor * val)


def train(model, memory, optimizer, batch_size, discount_factor):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        s = env.reset()
        done = False
        episode_length = 0

        while not done:
            a = select_action(model, s, get_epsilon(global_steps))
            s_next, r, done, _ = env.step(a)
            memory.push((s, a, r, s_next, done))
            s = s_next
            episode_length += 1
            global_steps += 1

            loss = train(model, memory, optimizer, batch_size, discount_factor)
        episode_durations.append(episode_length)

        if loss is not None:
            print("Episode: {:4d} | Loss: {}".format(i, loss))
    return episode_durations


if __name__ == "__main__":

    env = gym.envs.make("Acrobot-v1")
    print(f"Action space: {env.action_space} - State space: {env.observation_space}")

    # Let's run it!
    num_episodes = 100
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(100000)
    num_hidden = 128
    seed = 42  # This is not randomly chosen

    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    model = QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.n)

    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    plt.plot(episode_durations)
    plt.savefig("test.png")