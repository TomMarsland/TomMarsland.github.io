import torch
import torch.nn as nn
import random

class QNetwork(nn.Module):
    # nn.module is a base class for all neural network modules. Your models should also subclass this class
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    # we create a nn network with 3 fully connected layers. The first layer takes the state as input and outputs 64 features, the second layer takes those 64 features and outputs another 64 features, and the final layer takes those 64 features and outputs a value for each action (the Q-values).

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        # push function to save experience in replay buffer. If the buffer is full, it will overwrite the oldest experience.
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)

        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001

    def step(self, state, action, reward, next_state, done):
        # step function to save experience in replay buffer and   from it
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, eps=0.):
        # act function to select action using epsilon-greedy policy
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return torch.argmax(action_values).item()
        else:
            return random.choice(range(self.action_size))
