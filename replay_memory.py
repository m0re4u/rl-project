import random
import numpy as np
from collections import deque

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
        self.error_dict[transition] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[m], m) for m in self.memory]
        things.sort(reverse=True)
        probs = np.array([1/(n+1) for n in range(len(things))]) 
        probs /= sum(probs)
        return np.random.choice([trans for error, trans in things], size = batch_size, p = probs)

    def update_memory(self, transition, new_error):
        self.error_dict[transition] = new_error

    def __len__(self):
        return len(memory)

class PrioritizedProportionalMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque()
        self.error_dict = {}

    def push(self, transition, error):
        self.memory.append(transition)
        self.error_dict[transition] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[m], m) for m in self.memory]
        probs = np.array([float(error) for error,mem in things]) 
        probs /= sum(probs)
        return np.random.choice([trans for error, trans in things], size = batch_size, p = probs)

    def update_memory(self, transition, new_error):
        self.error_dict[transition] = new_error

    def __len__(self):
        return len(memory)    

class PrioritizedGreedyMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque()
        self.error_dict = {}

    def push(self, transition, error):
        self.memory.append(transition)
        self.error_dict[transition] = error
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        things = [(self.error_dict[m], m) for m in self.memory]
        things.sort(reverse=True)
        result = [trans for error, trans in things[:batch_size]]
        if len(result) < batch_size:
            return (result * int((batch_size/len(result))+1))[:batch_size]
        return result

    def update_memory(self, transition, new_error):
        self.error_dict[transition] = new_error

    def __len__(self):
        return len(memory)    


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