import random
import torch
from collections import deque
from SegmentTree import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Memory():
    def __init__(self, buffer_size, batch_size):
        self.state = deque(maxlen=buffer_size)
        self.action = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.next_state = deque(maxlen=buffer_size)
        self.done = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def update(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)
        # Add logging
        print(f"Updated memory: state len={len(self.state)}, action len={len(self.action)}")

    def sample(self):
        if len(self.state) < self.batch_size:
            raise ValueError("Sample larger than population or is negative")
        idx = random.sample(range(len(self.state)), self.batch_size)
        # Add logging
        print(f"Sampled indices: {idx}")
        return (torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), 
               torch.Tensor(self.reward)[idx].to(device), torch.Tensor(self.next_state)[idx].to(device), 
               torch.Tensor(self.done)[idx].to(device))

    def reset(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.next_state.clear()
        self.done.clear()

    def __len__(self):
        return len(self.state)



class PrioritizedMemory(Memory):
    def __init__(self, buffer_size, batch_size, alpha):
        super(PrioritizedMemory, self).__init__(buffer_size, batch_size)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        
        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def update(self, state, action, reward, next_state, done):
        super().update(state, action, reward, next_state, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % len(self.state)

    def sample_batch(self, beta):
        idx = np.array(self._sample_proportional())
        
        return (torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), 
               torch.Tensor(self.reward)[idx].to(device), torch.Tensor(self.next_state)[idx].to(device), 
               torch.Tensor(self.done)[idx].to(device), torch.from_numpy(np.array([self._calculate_weight(i, beta) for i in idx])).to(device))
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self.state) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices
    
    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.state)) ** (-beta)
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self.state)) ** (-beta)
        weight = weight / max_weight
        
        return weight

    def __len__(self):
        return len(self.state)