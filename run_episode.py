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
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(np.array(list(action)))  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action, env)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor, env)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    for i in range(state.shape[0]):
        memory.update_memory((state[i], action[i], reward[i], next_state[i], done[i]), loss.item())

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


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
            global_steps += 1
            episode_reward += r

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



def experience_her_episode(env,goal,global_steps,use_her=True):
    s = np.reshape(np.array(env.reset()),-1)
    if use_her:
        state_goal = np.concatenate([s, goal], axis=-1)

    done = False
    episode_length = 0
    episode_reward = 0
    experience = []
    while not done:
        r=0
        if use_her:
            a = select_action(model, state_goal, env, get_epsilon(global_steps))
            s_next, r, done, _ = env.step(a)
            experience.append((s, a, r, s_next,done))
            state_goal = np.concatenate([np.reshape(np.array(s_next),-1), goal], axis=-1)
            s = s_next
        else:
            a = select_action(model, s, env, get_epsilon(global_steps))
            s_next, r, done, _ = env.step(a)
            experience.append((s, a, r, s_next, done))
            s = s_next

        episode_length += 1
        episode_reward += r
    return experience,episode_length, episode_reward

def eval_her_episode(episode,extract_goal,calc_reward, her_type="episode"):
    '''
    her_type = ["future","episode","last"]
    '''
    new_experience = []
    for i,(s, a, r, sn,done) in enumerate(episode):
        if her_type == "future":
            samples = np.random.randint(i, len(episode), size=3)
        elif her_type == "episode":
            samples = np.random.randint(0, len(episode), size=3)
        else:
            samples = [-1]
        for sample in samples:
            goal = extract_goal(episode[sample][0])
            reward = calc_reward(s, a, goal)
            new_experience.append((np.concatenate([np.reshape(np.array(s),-1), goal], axis=-1),a,reward,np.concatenate([np.reshape(np.array(sn),-1), goal], axis=-1),done))

    return new_experience

def run_her_episodes(train, model, memory, env, num_episodes, training_steps, epochs, batch_size,
                        discount_factor,learn_rate,sample_goal,extract_goal,calc_reward,use_her=True):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_lengths_epoch = []
    episode_rewards_epoch = []
    for epoch in range(epochs):
        episode_lengths = []
        episode_rewards = []
        loss = None
        for i in range(num_episodes):
            goal = sample_goal()
            episode, episode_length,episode_reward = experience_her_episode(env, goal,global_steps,use_her)
            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            for s, a, r, sn, done in episode:
                if use_her:
                    goal_s = np.concatenate([ np.reshape(np.array(s),-1), goal], axis = -1)
                    goal_sn = np.concatenate([ np.reshape(np.array(sn),-1), goal], axis=-1)
                    memory.push((goal_s, a, r, goal_sn, done))
                else:
                    memory.push((s, a, r, sn, done))

            if use_her:
                her_experience = eval_her_episode(episode, extract_goal,calc_reward)
                for transition in her_experience:
                    memory.push(transition)


        for training_step in range(training_steps):
            loss = train(model, memory, optimizer, batch_size, discount_factor,env)
        if loss is not None:
            print("Epoch: {:4d} | Loss: {}".format(epoch, loss))
        episode_lengths_epoch.append(np.mean(episode_lengths))
        episode_rewards_epoch.append(np.mean(episode_rewards))
        global_steps+=1
        # if loss is not None:
        #     print("Episode: {:4d} | Loss: {}".format(i, loss))
    return episode_lengths_epoch, episode_rewards_epoch


if __name__ == "__main__":

    # Let's run it!
    num_episodes = 300
    batch_size = 10
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(10000)
    num_hidden = 128
    seed = 42  # This is not randomly chosen

    # env = gym.envs.make("Acrobot-v1")
    # print(f"Action space: {env.action_space} - State space: {env.observation_space}")
    # # We will seed the algorithm (before initializing QNetwork!) for reproducability
    # random.seed(seed)
    # torch.manual_seed(seed)
    # env.seed(seed)

    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    # model = QNetwork(env.observation_space.shape[0], num_hidden, env.action_space.n)

    # episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    # plt.plot(episode_durations)
    # plt.savefig("test.png")


    import loop_environments
    env = loop_environments.create_env("LargeGridWorld")
    # env = gym.envs.make("LunarLander-v2")
    print(f"Action space: {env.action_space} - State space: {env.observation_space}")

    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)



    her = True
#  lander
    # sample_goal = lambda: (0, 0)
    # extract_goal = lambda state: state[0:2]
    # def calc_reward(state, action, goal):
    #     distance = np.linalg.norm(np.array(state[0:2]) - np.array(goal))
    #     return  - 100* distance

# functions for grid world
    def sample_goal():
        return np.random.choice([0, env.observation_space.n - 1], 1)
    extract_goal = lambda state: np.reshape(np.array(np.argmax(state)),-1)
    def calc_reward(state, action, goal):
        if state == goal:
            return 0.0
        else:
            return -1.0
# # maze
#     def sample_goal():
#         return env.maze.end_pos
#     extract_goal = lambda state: np.reshape(np.array(np.argmax(state)),-1)
#     def calc_reward(state, action, goal):
#         if state == goal:
#             return 0.0
#         else:
#             return -1.0
    num_episodes = 5
    epochs = 600
    training_steps = 10
    print(env.reset())
    if her:
        # model = QNetwork(env.observation_space.shape[0]+2, num_hidden, env.action_space.n)
        model = QNetwork(2*env.observation_space.n, num_hidden, env.action_space.n)
        episode_durations,episode_rewards  = run_her_episodes(train, model, memory, env, num_episodes, training_steps, epochs, batch_size, discount_factor, learn_rate, sample_goal, extract_goal, calc_reward)
    else:
        model = QNetwork(env.observation_space.n, num_hidden, env.action_space.n)
        episode_durations,episode_rewards = run_her_episodes(train, model, memory, env, num_episodes, training_steps, epochs, batch_size, discount_factor, learn_rate, sample_goal, extract_goal, calc_reward,use_her=False)

    plt.plot(loop_environments.smooth(episode_durations, 10))
    plt.savefig("test.png")