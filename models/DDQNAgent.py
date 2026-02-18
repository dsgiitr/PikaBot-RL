import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from Network import Network 
from Memory import Memory 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DDQNAgent:
    def __init__(self, state_size=16, action_size=9, buffer_size=100000, batch_size=1, gamma=0.95, lr=1e-1, 
                 learn_delay=2, update_rate=16, target_tau=0.99, max_epsilon=0.1, min_epsilon=0.001, epsilon_decay=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learn_delay = learn_delay
        self.update_rate = update_rate
        self.target_tau = target_tau

        self.network = Network(state_size, action_size).to(device)
        self.target_network = Network(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.memory = Memory(buffer_size, batch_size)
        self.t_step = 0

        self.initial_weights = self.get_weights()

    def get_weights(self):
        return {name: param.clone() for name, param in self.network.named_parameters()}

    def compare_weights(self, old_weights):
        for name, param in self.network.named_parameters():
            if not torch.equal(param, old_weights[name]):
                print(f"Weights updated for layer: {name}")
            else:
                print(f"Weights not updated for layer: {name}")


    def step(self, state, action, reward, next_state, done):
        self.memory.update(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.learn_delay == 0 :
            print("Learning")
            self.learn()
        if self.t_step % self.update_rate == 0:
            print("Soft update")
            self.soft_update()

        # self.compare_weights(self.initial_weights)

    def act(self, state):
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * self.t_step)
        if random.random() > epsilon:
            state = torch.as_tensor(state).float().unsqueeze(0).to(device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(range(self.action_size))

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        actions = actions.unsqueeze(1)
        Q_ts = self.network(states).gather(1, actions)
        Q_t1s = self.network(next_states).detach().argmax(1).unsqueeze(1)
        Q_t1t = self.target_network(next_states).gather(1, Q_t1s)
        Q_tt = rewards + (self.gamma * Q_t1t * (1 - dones))
        loss = self.criterion(Q_ts, Q_tt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_param, source_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.target_tau * source_param.data + (1.0 - self.target_tau) * target_param.data)