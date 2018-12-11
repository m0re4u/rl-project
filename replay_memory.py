import random
import numpy as np
import torch
from collections import deque

def hash_state(state):
    if type(state) == np.int64 or type(state) == int:
        return int(state)
    elif type(state) == np.ndarray or type(state) == torch.Tensor and state.size() != []:
        return state.__repr__()
    elif type(state) == torch.Tensor:
        return int(state)
    else:
        raise NotImplementedError(f"Hash does not exist for type {type(state)} - {state}")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque()

    def push(self, transition, error=0):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            # Remove first element if we exceed the capacity
            self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def update_memory(self, transition, new_error):
        pass

    def __len__(self):
        return len(self.memory)

class PrioritizedRankbasedMemory:
    def __init__(self, capacity):
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
        probs = np.array([1/(n+1) for n in range(len(things))])
        probs /= sum(probs)
        # return np.random.choice([trans for error, trans in things], size = batch_size, p = probs)
        transitions = [trans for error, trans in things]
        idx = np.random.choice(len(transitions),size=batch_size, p=probs)
        return np.array(transitions)[idx]

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = new_error

    def __len__(self):
        return len(self.memory)

class PrioritizedProportionalMemory:
    def __init__(self, capacity):
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
        probs = np.array([float(error) for error,mem in things])
        probs /= sum(probs)
        transitions = [trans for error, trans in things]
        idx = np.random.choice(len(transitions),size=batch_size, p=probs)
        return np.array(transitions)[idx]

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
        self.error_dict[(hash_state(s), a, r, hash_state(s_p), done)] = new_error

    def __len__(self):
        return len(self.memory)

class PrioritizedGreedyMemory:
    def __init__(self, capacity):
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
            return (result * int((batch_size/len(result))+1))[:batch_size]
        return result

    def update_memory(self, transition, new_error):
        s, a, r, s_p, done = transition
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
