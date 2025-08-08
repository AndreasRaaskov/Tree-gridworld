"""
Agent Implementations for TreeGridWorld Environment.

This module contains different agent implementations for the TreeGridWorld:
- RandomAgent: Takes random actions
- GreedyAgent: Always moves towards the closest tree using Manhattan distance
- DQNAgent: Deep Q-Network agent using convolutional neural networks for learning optimal policies

The DQNAgent uses experience replay and epsilon-greedy exploration to learn from interactions
with the environment. It includes functionality for saving/loading trained models and experiences.
"""

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
from collections import deque
from gym import spaces
import torch.nn.functional as F

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
        # Paper specification: 2-section convolutional network with channels (16, 16), 3x3 conv, stride 1
        self.conv = nn.Sequential(
            nn.Conv2d(state_size, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Calculate flattened size for 12x12 grid after conv layers
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, action_size)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(device)  # Make sure 'device' is defined
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.9, 0.999))


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
            state = torch.from_numpy(state).float().to(device)
            predicted = self.model(state)[action]
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(predicted,  target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name,map_location=torch.device('cpu')))

    def save(self, name):
        #save memory
        state, action, reward, next_state, done = zip(*self.memory)
        np.save(f'checkpoints/{name}_state.npy', state)
        np.save(f'checkpoints/{name}_reward.npy', action)

        #save model
        torch.save(self.model.state_dict(), f'checkpoints/{name}_model.pt')


# AI-generated V-MPO implementation based on paper specifications
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        # Paper specification: 2-section convolutional network with channels (16, 16), 3x3 conv, stride 1
        self.conv = nn.Sequential(
            nn.Conv2d(state_size, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Shared features
        self.fc_shared = nn.Linear(16 * 12 * 12, 256)
        
        # Actor head (policy)
        self.actor = nn.Linear(256, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(device)
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Return both policy logits and value
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value


# AI-generated V-MPO agent implementation
class VMPOAgent:
    def __init__(self):
        self.state_size = 2
        self.action_size = 5
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount factor
        self.epsilon_c = 0.2  # constraint for policy update
        self.epsilon_mu = 0.1  # constraint for temperature update
        self.alpha = 0.1  # temperature parameter
        
        self.model = ActorCriticNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))  # Paper: lr=10^-4
        
        # V-trace parameters
        self.rho_bar = 1.0
        self.c_bar = 1.0

    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        policy_logits, _ = self.model(state)
        
        # Sample action from policy
        probs = F.softmax(policy_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def v_trace_returns(self, rewards, values, next_values, dones, log_probs, old_log_probs):
        # Simplified V-trace implementation
        returns = []
        vs = []
        
        for i in range(len(rewards)):
            if i == len(rewards) - 1:
                next_value = next_values[i] if not dones[i] else 0
            else:
                next_value = values[i + 1]
            
            # Importance sampling ratio
            rho = torch.exp(log_probs[i] - old_log_probs[i]).clamp(max=self.rho_bar)
            c = torch.exp(log_probs[i] - old_log_probs[i]).clamp(max=self.c_bar)
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            
            if i == len(rewards) - 1:
                vs_minus_v = 0
            else:
                vs_minus_v = vs[i - 1] - values[i] if i > 0 else 0
            
            v_s = values[i] + delta * rho + self.gamma * c * vs_minus_v
            vs.append(v_s)
            returns.append(v_s)
        
        return torch.stack(returns)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        
        # Forward pass
        policy_logits, values = self.model(states)
        _, next_values = self.model(next_states)
        
        # Current policy log probs
        probs = F.softmax(policy_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions)
        
        # V-trace returns
        v_trace_returns = self.v_trace_returns(rewards, values.squeeze(), next_values.squeeze(), 
                                              dones, log_probs, old_log_probs)
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), v_trace_returns.detach())
        
        # Policy loss (simplified V-MPO)
        advantages = v_trace_returns - values.squeeze()
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))

    def save(self, name):
        # Save memory
        if len(self.memory) > 0:
            states, actions, rewards, next_states, dones, log_probs = zip(*self.memory)
            np.save(f'{name}_state.npy', states)
            np.save(f'{name}_actions.npy', actions)
            np.save(f'{name}_rewards.npy', rewards)
        
        # Save model
        torch.save(self.model.state_dict(), f'{name}_model.pt')
