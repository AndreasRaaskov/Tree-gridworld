"""
Script for training the model.
"""

from agent import RandomAgent, DQNAgent
from env import TreeGridWorld
from visualize import render_log,plot_log
import numpy as np

import matplotlib.pyplot as plt


# Initialize the environment and the agent
env = TreeGridWorld()
agent = DQNAgent()


# Run the agent on the environment
num_episodes = 2*10**9
num_steps_per_episode = 400
log={'episode':[],'reward':[],'trees':[],'checkpoint':[]} #log for plotting
temp_log={'episode':[],'reward':[],'trees':[],'checkpoint':[]} 
for episode in range(num_episodes):
    observation = env.reset()
    cumulative_reward = 0
    for step in range(num_steps_per_episode):
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action)

        # save to agent's memory
        agent.remember(observation, action , reward, next_observation, done)
        observation = next_observation

        cumulative_reward += reward

        if done:
            break
    
    # train the agent with the experience of the episode
    agent.replay(num_steps_per_episode)
    temp_log['episode'].append(episode)
    temp_log['reward'].append(cumulative_reward)
    temp_log['trees'].append(env.current_num_trees)

    # Save the mean of the last 100 episodes
    if episode%100==0:
        log['episode'].append(np.mean(temp_log['episode']))
        log['reward'].append(np.mean(temp_log['reward']))
        log['trees'].append(np.mean(temp_log['trees']))

        #Reset temp_log
        temp_log={'episode':[],'reward':[],'trees':[],'checkpoint':[]}


    # Save the model and plot the progress every 10000 episodes
    if episode%10000==0:
        agent.save(f'c_{episode}')
        fig=plot_log(log)
        fig.savefig(f'Progres_plot.png')
    


    

