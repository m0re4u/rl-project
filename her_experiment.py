from run_episode import train,select_action,get_epsilon
from gridworld import GridworldEnv
import loop_environments
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from torch import optim
from replay_memory import ReplayMemory
from QNetwork import QNetwork
import pickle

def experience_her_episode(env,model,goal,batch,global_steps):
    s = np.reshape(np.array(env.reset()), -1)
    state_goal = np.concatenate([s, goal], axis=-1)

    done = False
    episode_length = 0
    episode_reward = 0
    experience = []
    while not done and episode_length<batch:
        a = select_action(model, state_goal, env, get_epsilon(global_steps))
        s_next, r, done, _ = env.step(a)

        #  this two lines work only for grid world
        # print(s,s_next,goal)
        # done = True if s_next == goal[0] else False
        # r = 0.0 if s == goal[0] else -1.0

        experience.append((s, a, r, s_next,done))
        state_goal = np.concatenate([np.reshape(np.array(s_next),-1), goal], axis=-1)
        s = s_next

        episode_length += 1
        episode_reward += r
    return experience,episode_length, episode_reward

def eval_her_episode(episode,extract_goal,calc_reward, her_type="future"):
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
    max_episode_length = 2000
    for epoch in range(epochs):
        episode_lengths = []
        episode_rewards = []
        loss = None
        for i in range(num_episodes):
            if use_her:
                goal = sample_goal()
            else:
                goal = []

            episode, episode_length, episode_reward = experience_her_episode(
                                                        env, model, goal, max_episode_length, global_steps)

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            # if episode_length == max_episode_length:
            #     continue
            for s, a, r, sn, done in episode:
                goal_s = np.concatenate([ np.reshape(np.array(s),-1), goal], axis = -1)
                goal_sn = np.concatenate([ np.reshape(np.array(sn),-1), goal], axis=-1)
                memory.push((goal_s, a, r, goal_sn, done),loss)

            if use_her:
                her_experience = eval_her_episode(episode, extract_goal,calc_reward)
                for transition in her_experience:
                    memory.push(transition,loss)

            for training_step in range(training_steps):
                global_steps+=1
                loss = train(model, memory, optimizer, batch_size, discount_factor,env)

        if loss is not None:
            print("Epoch: {:4d} | Loss: {} | Episode length: {}".format(
                epoch, loss,np.mean(episode_lengths)))
        episode_lengths_epoch.append(np.mean(episode_lengths))
        episode_rewards_epoch.append(np.mean(episode_rewards))
        # if loss is not None:
        #     print("Episode: {:4d} | Loss: {}".format(i, loss))
    return episode_lengths_epoch, episode_rewards_epoch

def line_plot_var(x_data, y_data, low_CI, upper_CI, x_label, y_label, data_labels,title,colors):
    # Create the plot object
    _, ax = plt.subplots()

    for x,y,l,h,label,color in zip(x_data,y_data,low_CI,upper_CI,data_labels,colors):
        # Plot the data, set the linewidth, color and transparency of the
        # line, provide a label for the legend
        # color = '#539caf'
        ax.plot(x, y, lw = 1, alpha = 1, label = label,color=color)
        # Shade the confidence interval
        ax.fill_between(x, l, h, alpha = 0.4, label=None,color=color)
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc='best')
    plt.savefig("test.png")



def her_experiment():
    batch_size = 256
    discount_factor = 0.8
    learn_rate = 1e-3
    num_hidden = 128
    num_episodes = 2
    epochs = 200
    training_steps = 10
    memory_size = 100000
    # her = False
    # seeds = [42, 30, 2,19,99]  # This is not randomly chosen
    seeds = [42, 30, 2,19,99]
    shape=[30,30]
    targets = lambda x,y: [0,x*y-1,x-1,(y-1)*x]
    env = GridworldEnv(shape=shape, targets=targets(*shape))
    # functions for grid world
    def sample_goal():
        return np.random.choice(env.targets, 1)

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
    means = []
    x_epochs = []
    l_stds = []
    h_stds = []
    for her in [True,False]:
        episode_durations_all = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.seed(seed)
            print(env.reset())
            memory = ReplayMemory(memory_size)
            if her:
                # model = QNetwork(env.observation_space.shape[0]+2, num_hidden, env.action_space.n)
                model = QNetwork(2*env.observation_space.n, num_hidden, env.action_space.n)
                episode_durations,episode_rewards  = run_her_episodes(train, model, memory, env,    num_episodes, training_steps, epochs, batch_size, discount_factor, learn_rate,  sample_goal, extract_goal, calc_reward,use_her=True)
            else:
                model = QNetwork(env.observation_space.n, num_hidden, env.action_space.n)
                episode_durations,episode_rewards = run_her_episodes(train, model, memory, env, num_episodes, training_steps, epochs, batch_size, discount_factor, learn_rate,   sample_goal, extract_goal, calc_reward,use_her=False)

            episode_durations_all.append(loop_environments.smooth(episode_durations,10))
        mean = np.mean(episode_durations_all,axis=0)
        means.append(mean)
        std = np.std(episode_durations_all, ddof=1, axis=0)
        l_stds.append(mean - std)
        h_stds.append(mean + std)
        x_epochs.append(list(range(len(mean))))
        # print(len(mean),mean,std)
    line_plot_var(x_epochs, means, l_stds, h_stds, "Epoch", "Duration", ["HindsightReplay", "RandomReplay"], "Episode duration per epoch",["orange","blue"])
    name = "her_" + str(shape)
    file_name = os.path.join("./results", name)

    with open(file_name + ".pkl", "wb") as f:
        pickle.dump((x_epochs, means, l_stds, h_stds), f)



if __name__ == "__main__":
    her_experiment()

    # shape=[30,30]
    # name = "her_" + str(shape)
    # file_name = os.path.join("./results", name)
    # with open(file_name + ".pkl", "rb") as f:
    #     x_epochs, means, l_stds, h_stds = pickle.load(f)
    #     line_plot_var(x_epochs, means, l_stds, h_stds, "Epoch", "Duration", ["HindsightReplay", "RandomReplay"], "Episode duration per epoch",["orange","blue"])