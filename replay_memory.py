import random
import numpy as np
import torch
from collections import deque

def hash_state(state):
    if type(state) == np.int64 or type(state) == int:
        return int(state)
    elif type(state) == np.ndarray or type(state) == torch.Tensor and state.size() != []:
        return str(state)
    elif type(state) == torch.Tensor:
        return int(state)
    elif type(state) == float and abs(int(state) - state) < 0.0001:
        return int(state)
    else:

        raise NotImplementedError(f"Hash does not exist for type {type(state)} - {state}")

class ReplayMemory:
    def __init__(self, capacity, alpha=None, beta=None):
        self.capacity = capacity
        self.memory = deque()

    def push(self, transition, error=0):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            # Remove first element if we exceed the capacity
            self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size), np.ones(batch_size)

    def update_memory(self, transition, new_error):
        pass

    def __len__(self):
        return len(self.memory)

class PrioritizedRankbasedMemory:
    def __init__(self, capacity, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.memory = deque()
        self.error_dict = {}

    def push(self, transition, error):
        self.memory.append(transition)
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)], (s, a, r, s_p, done)) for s, a, r, s_p, done in self.memory]
        things.sort(reverse=True, key=lambda tup: tup[0])
        priorities = np.array([(1/(n+1))**self.alpha for n in range(len(things))])
        probs = priorities / sum(priorities)
        # return np.random.choice([trans for error, trans in things], size = batch_size, p = probs)
        transitions = [trans for error, trans in things]
        idx = np.random.choice(len(transitions),size=batch_size, p=probs)
        
        min_prio = min(priorities)
        max_weight = (len(self) * min_prio) ** -self.beta
        weights = [(p * len(self)) ** -self.beta / max_weight for p in priorities[idx]]
        
        return np.array(transitions)[idx], weights

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = new_error

    def __len__(self):
        return len(self.memory)

class PrioritizedProportionalMemory:
    def __init__(self, capacity, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.memory = deque()
        self.error_dict = {}

    def push(self, transition, error):
        self.memory.append(transition)
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)], (s, a, r, s_p, done)) for s, a, r, s_p, done in self.memory]
        priorities = np.array([float(error)**self.alpha for error,mem in things])
        probs = priorities / sum(priorities)
        transitions = [trans for error, trans in things]
        idx = np.random.choice(len(transitions),size=batch_size, p=probs)
        
        min_prio = min(priorities)
        max_weight = (len(self) * min_prio) ** -self.beta
        weights = [(p * len(self)) ** -self.beta / max_weight for p in priorities[idx]]

        return np.array(transitions)[idx], weights

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = new_error

    def __len__(self):
        return len(self.memory)

class PrioritizedGreedyMemory:
    def __init__(self, capacity, alpha=None, beta=None):
        self.capacity = capacity
        self.memory = deque()
        self.error_dict = {}

    def push(self, transition, error):
        self.memory.append(transition)
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)], (s, a, r, s_p, done)) for s, a, r, s_p, done in self.memory]
        things.sort(reverse=True, key=lambda tup: tup[0])
        result = [trans for error, trans in things[:batch_size]]
        if len(result) < batch_size:
            return (result * int((batch_size/len(result))+1))[:batch_size], np.ones(batch_size)
        return result, np.ones(batch_size)

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
        key = (hash_state(s), a, r, hash_state(s_p), done)
        if key not in error_dict:
            # logging professionally, as you can see. thank god this won't be maintained...
            print('WARNING: The transition you tried to update has not ever been in memory before. This is most likely because the input format is different to the one originally used.')
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = new_error

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    normalmem = ReplayMemory(100)
    greedymem = PrioritizedGreedyMemory(100)
    rankmem = PrioritizedRankbasedMemory(100)
    propmem = PrioritizedProportionalMemory(100)
    mems = [normalmem, greedymem, rankmem, propmem]

    for mem in mems:
        print(type(mem))
        for i in range(200):
            mem.push(i,i)
        print(mem.sample(10))
        for i in range(200):
            mem.update_memory(i,100-i)
        print(mem.sample(10), '\n')
