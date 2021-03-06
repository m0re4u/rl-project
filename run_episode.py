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

from replay_memory import ReplayMemory, PrioritizedGreedyMemory, PrioritizedRankbasedMemory, PrioritizedProportionalMemory
from QNetwork import QNetwork
import copy

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def get_epsilon(it):
    N = 1000
    ep = 1 - (1*it/N)
    if ep < 0.05:
        return 0.05
    else:
        return ep

def discrete_state_to_onehot(state, batch_size, state_n=None):

    indices = torch.from_numpy(np.array(state)).type(torch.int64).view(-1, 1)
    state_n = state_n if state_n is not None else int(torch.max(indices)) + 1
    one_hots = torch.zeros(indices.size()[0], state_n).scatter_(1, indices, 1)
    one_hots = one_hots.view(*indices.shape, -1)
    # print("aaaaa",state,one_hots.view(batch_size,-1))
    if batch_size == 1:
        return one_hots.view(-1)
    else:
        return one_hots.view(batch_size,-1)


def select_action(model, state, env, epsilon):
    p = random.random()
    if p > epsilon:
        # Select greedy action
        with torch.no_grad():
            if type(env.observation_space) == gym.spaces.Discrete:
                out = model(discrete_state_to_onehot(state, 1, env.observation_space.n))
                val, ind = out.max(0)
                if type(env.action_space) == gym.spaces.Discrete:
                    return ind.item()
                elif type(env.action_space) == gym.spaces.Box:
                    return val
            elif type(env.observation_space) == gym.spaces.Box:
                out = model(torch.Tensor(state))
                val, ind = out.max(0)
                if type(env.action_space) == gym.spaces.Discrete:
                    return ind.item()
                elif type(env.action_space) == gym.spaces.Box:
                    return val
    else:
        # Select random action
        if type(env.action_space) == gym.spaces.Discrete:
            return random.choice(list(range(env.action_space.n)))
        elif type(env.action_space) == gym.spaces.Box:
            return torch.tensor(env.action_space.low) + torch.rand(env.action_space.low.shape) * (torch.tensor(env.action_space.high) - torch.tensor(env.action_space.low))
        else:
            raise NotImplementedError()


def compute_q_val(model, state, action, env):
    if type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Box:
        x = model(state)
        out = torch.gather(x, 1, action.view(-1, 1))
        return out.view(-1)
    elif type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Box:
        x = model(state).view(-1)
        return x
    elif type(env.action_space) == gym.spaces.Discrete and type(env.observation_space) == gym.spaces.Discrete:
        # print("-------------")
        x = model(discrete_state_to_onehot(state, len(state), env.observation_space.n))
        action = action.type(torch.LongTensor)
        out = torch.gather(x.view(len(state), -1), 1, action.view(-1, 1))
        return out.view(-1)
    elif type(env.action_space) == gym.spaces.Box and type(env.observation_space) == gym.spaces.Discrete:
        x = model(discrete_state_to_onehot(state, len(state), env.observation_space.n)).view(-1)
        return x

def compute_target(model, reward, next_state, done, discount_factor, env):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    if type(env.observation_space) == gym.spaces.Box:
        val, _ = model(next_state).max(1)
        val[done] = 0
        return reward + (discount_factor * val)
    elif type(env.observation_space) == gym.spaces.Discrete:
        # print("target")
        batch_n = next_state.shape[0]
        val, _ = model(discrete_state_to_onehot(next_state, batch_n, env.observation_space.n)).view(batch_n, -1).max(1)
        val[done] = 0
        return reward + (discount_factor * val)


def train(model, memory, optimizer, batch_size, discount_factor, env):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions, weights = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    # print(next_state)
    action = torch.tensor(np.array(list(action)))  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean
    state = torch.tensor(state, dtype=torch.float)

    weights = torch.tensor(weights, dtype=torch.float)


    # compute the q value
    q_val = compute_q_val(model, state, action, env)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor, env)

    # loss is measured from error between current and newly expected Q values
    unreduced_loss = F.smooth_l1_loss(q_val, target, reduction='none')

    actual_loss = torch.mean(unreduced_loss)

    weighted_loss_for_backprop = torch.mean(unreduced_loss * weights)

    for i in range(state.shape[0]):
        # memory.update_memory((state[i].item(), action[i].item(), reward[i].item(), next_state[i].item(), done[i].item()), unreduced_loss[i].item())
        memory.update_memory(transitions[i],unreduced_loss[i].item())

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    weighted_loss_for_backprop.backward()
    optimizer.step()

    return actual_loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    episode_rewards = []
    for i in range(num_episodes):
        s = env.reset()
        done = False
        episode_length = 0
        episode_reward = 0

        while not done:
            a = select_action(model, s, env, get_epsilon(global_steps))
            s_next, r, done, _ = env.step(a)
            episode_length += 1
            episode_reward += r
            global_steps += 1
            loss = train(model, memory, optimizer, batch_size, discount_factor, env)
            if loss is None:
                loss = 1000
            memory.push((s, a, r, s_next, done), loss)
            s = s_next
        episode_durations.append(episode_length)
        episode_rewards.append(episode_reward)


        if loss is not None:
            print("Episode: {:4d} | Loss: {}".format(i, loss))
    return episode_durations, episode_rewards



if __name__ == "__main__":
    import loop_environments
    env = loop_environments.create_env("SimpleWindyGridWorld")
    # Let's run it!
    num_episodes = 200
    batch_size = 10
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(10000)
    num_hidden = 128
    seed = 42  # This is not randomly chosen
    # env = gym.envs.make("Acrobot-v1")
    # print(f"Action space: {env.action_space} - State space: {env.observation_space}")
    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    print(env.observation_space.shape)
    print(env.action_space.shape)
    model = QNetwork(env.observation_space.n, num_hidden, env.action_space.n)

    episode_durations, episode_rewards = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    plt.plot(episode_durations)
    plt.savefig("test.png")


