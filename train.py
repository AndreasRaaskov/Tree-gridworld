"""
Alternative Training Script - Step-based training with real-time visualization.

This script provides an alternative training approach using step-based learning instead of episodes.
It loads a pre-trained model and continues training with real-time OpenCV visualization and 
detailed logging of tree distributions and rewards per step.

Key differences from main.py:
- Step-based rather than episode-based training  
- Real-time visualization during training
- Detailed tree count distribution tracking
"""

from agent import RandomAgent, DQNAgent, VMPOAgent
from env import TreeGridWorld

import numpy as np
import cv2
import matplotlib.pyplot as plt

checkpoint_step=10000  # Save every 10,000 steps

def plot_log(log):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    steps = log['step']
    rewards = log['reward']
    tree_distributions = log['tree_distributions']
    
    # Plot tree distribution matrix on ax1 (like the working version)
    tree_matrix = np.array(tree_distributions).T  # Shape: (11, num_checkpoints)
    n, m = tree_matrix.shape
    ax1.imshow(tree_matrix, cmap='GnBu', interpolation='nearest', aspect='auto', origin='lower')
    ax1.set_ylabel('Number of Trees')
    ax1.set_ylim(-0.5, 10.5)
    
    # Set y-ticks to show tree counts properly
    ax1.set_yticks(range(11))
    ax1.set_yticklabels(range(11))
    
    # Move x-axis to bottom and fix orientation
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_label_position('bottom')
    
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    
    # Plot reward line on ax2 (like the working version)
    ax2.plot(np.arange(m), rewards, color='orange', label='Average Reward')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Training Progress - V-MPO Agent')
    
    # Set x-tick labels to show actual step values
    if len(steps) > 0:
        ax1.set_xticks(range(len(steps)))
        ax1.set_xticklabels(steps)
        ax2.set_xticks(range(len(steps)))
        ax2.set_xticklabels(steps)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    return fig


# Initialize the environment and the agent
env = TreeGridWorld()
agent = VMPOAgent()  # Using new V-MPO agent
# agent.load('checkpoints_vmpo/vmpo_1000_model.pt')  # Uncomment to load existing V-MPO checkpoint


# Run the agent on the environment
reward_log=[]
trees_log=[]
log={'step':[],'reward':[],'tree_distributions':[]} #log for plotting
step=1
observation = env.reset()
while True:

    action, log_prob = agent.act(observation)
    next_observation, reward, done, _ = env.step(action)

    # save to agent's memory (V-MPO needs log_prob)
    agent.remember(observation, action, reward, next_observation, done, log_prob)
    observation = next_observation

    reward_log.append(reward)
    trees_log.append(env.current_num_trees)

    if step%128==0:
        agent.replay(100)

    if step%checkpoint_step==0:  # save every n steps
        
        # Calculate averages for this interval
        mean_reward = np.mean(reward_log) if len(reward_log) > 0 else 0
        
        # Calculate tree count distribution as percentages
        if len(trees_log) > 0:
            tree_counts = [trees_log.count(i) for i in range(11)]  # Count occurrences of 0-10 trees (correct order)
            total_steps = len(trees_log)
            tree_percentages = [(count / total_steps) * 100 for count in tree_counts]
        else:
            tree_percentages = [0] * 11
        
        # Add to main log
        log['step'].append(step)
        log['reward'].append(mean_reward)
        log['tree_distributions'].append(tree_percentages)
        
        print(f'Step: {step-checkpoint_step} to {step} | Avg Reward: {mean_reward:.3f}')
        
        # Reset interval logs
        reward_log = []
        trees_log = []
        
        # Save checkpoint and plot
        agent.save(f'checkpoints_vmpo/vmpo_{step}')
        fig = plot_log(log)
        fig.savefig(f'vmpo_progress.png')
        plt.close(fig)  # Free memory


    env.render_cv()
    step+=1
    if done:
        break



    
