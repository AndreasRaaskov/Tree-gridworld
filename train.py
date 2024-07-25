from agent import RandomAgent, DQNAgent
from env import TreeGridWorld

import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_log(log):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    step=log['step']

    trees=np.array(log['trees']).T
    n, m = trees.shape
    ax1.matshow(trees,cmap="GnBu",interpolation='nearest',aspect='auto')
    ax1.set_ylabel('Trees left')
    ax1.set_ylim(0, 10)

    

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    ax2.plot(np.arange(m),log['reward'],color='orange',label='Reward')
    ax2.set_xlabel('steeps 10^3')
    ax2.set_ylabel('Reward')
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    #fig.show()
    return fig


# Initialize the environment and the agent
env = TreeGridWorld()
agent = DQNAgent()
agent.load('checkpoints/s_39000_model.pt')


# Run the agent on the environment
reward_log=[]
trees_log=[]
log={'step':[],'reward':[],'trees':[],'checkpoint':[]} #log for plotting
step=1
observation = env.reset()
while True:

    action = agent.act(observation)
    next_observation, reward, done, _ = env.step(action)

    # save to agent's memory
    agent.remember(observation, action , reward, next_observation, done)
    observation = next_observation

    reward_log.append(reward)
    trees_log.append(env.num_trees)

    if step%128==0:
        agent.replay(100)

    if step%1000==0:
        
        log['step'].append(step)
        mean_reward=np.mean(reward_log)
        log['reward'].append(np.mean(mean_reward))
        print(f'Step: {step-1000} to {step} Reward: {mean_reward}')
        log_reward=[]

        thee_count=[np.sum(np.array(trees_log)==i) for i in range(0,11)]
        log['trees'].append(thee_count)
        print(thee_count)
        trees_log=[]

        agent.save(f's_{step}')
        fig=plot_log(log)
        fig.savefig(f's_{step}.png')


    env.render_cv()
    step+=1
    if done:
        break



    
