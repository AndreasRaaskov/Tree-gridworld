from agent import RandomAgent, DQNAgent, greedyAgent
from env import TreeGridWorld
from visualize import render_log,plot_log
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Initialize the environment and the agent
env = TreeGridWorld(rmax=100)
agent = greedyAgent(env.action_space)
#agent.load('checkpoints/checkpoints4000_model.pt')

# Run the agent on the environment
num_episodes = 1
num_steps_per_episode = 200
log={'episode':[],'reward':[],'trees':[],'checkpoint':[]} #log for plotting
temp_log={'episode':[],'reward':[],'trees':[],'checkpoint':[]} 
for episode in range(num_episodes):
    env = TreeGridWorld(rmax=100,num_trees=num)
    observation = env.reset()
    cumulative_reward = 0
    for step in range(num_steps_per_episode):
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        
        cumulative_reward += reward
        env.render_cv()
        if done:
            break
    
    # train the agent with the experience of the episode

    temp_log['episode'].append(episode)
    temp_log['reward'].append(cumulative_reward)
    temp_log['trees'].append(env.num_trees)
    if episode%100==0:
        log['episode'].append(np.mean(temp_log['episode']))
        log['reward'].append(np.mean(temp_log['reward']))
        log['trees'].append(np.mean(temp_log['trees']))

        #Reset temp_log
        temp_log={'episode':[],'reward':[],'trees':[],'checkpoint':[]}
    print('Episode: {} Reward: {} Trees: {}'.format(episode, cumulative_reward, env.num_trees))