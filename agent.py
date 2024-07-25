import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
from collections import deque
from gym import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

class RandomAgent():
    def __init__(self):
        self.action_space = spaces.Discrete(5)

    def act(self, observation):
        return self.action_space.sample()
    

    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self, batch_size):
        pass

    def load(self, name):
        pass

    def save(self, name):
        pass

#An agent that always moves towards the closest tree (used )
class greedyAgent():
    def __init__(self, ):
        self.action_space = spaces.Discrete(5)

    def act(self, observation):
        distance=[]

        agent_position = np.array(np.where(observation[0,:,:]==1)).T
        tree_positions = np.array(np.where(observation[1,:,:]==1)).T

        if len(tree_positions)==0:
            #If no more trees, just move randomly
            return self.action_space.sample()
        for p in tree_positions:
            distance.append(np.sum(np.abs(p-agent_position)))
        
        closest_tree = np.argmin(distance)

        direction = tree_positions[closest_tree]-agent_position[0]

        if np.abs(direction[0])>np.abs(direction[1]): #Move vertically
            if direction[0]<0: #move up
                return 3
            else: #move down
                return 2
        else: #Move horizontally
            if direction[1]>0: #move right
                return 0
            else: #move left
                return 1
    
    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self, batch_size):
        pass

    def load(self, name):
        pass

    def save(self, name):
        pass


        


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_size, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, action_size)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(device) #convert to tensor and send to device
        
        x = self.conv(x)
        x = x.view(-1)
        x = self.fc(x)
        return x


class DQNAgent:
    def __init__(self):
        self.state_size = 2
        self.action_size = 5
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999

        self.model = QNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0, 0.999))


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.forward(state).detach().cpu().numpy()
        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            
            if done:
                target = reward
            else:

                Q_future = max(self.model.forward(next_state).cpu())
                target = reward + Q_future * self.gamma
            
            #Train model
            predicted = self.model(state)[action]
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(predicted,  target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        #save memory
        state, action, reward, next_state, done = zip(*self.memory)
        np.save(f'checkpoints/{name}_state.npy', state)
        np.save(f'checkpoints/{name}_reward.npy', action)

        #save model
        torch.save(self.model.state_dict(), f'checkpoints/{name}_model.pt')
