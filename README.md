# Tree-gridworld

This project is an unfinished attempt to replicate the behavior of the Tree Gridworld example in the DeepMind paper: "Goal Misgeneralization: Why Correct Specifications Aren’t Enough for Correct Goals."  
[Link to the paper](https://arxiv.org/pdf/2210.01790)

## What is completed
* `env.py` contains a 2x10x10 gridworld environment as described in the paper. 
* `Show_run.ipynb` contains code for visualizing an agent in OpenCV.

## What's semi-complete
* `main.py` trains the agent. The agent is trained in episodes of 400 steps and then resets, running only one instance at a time. This is different from the paper, which trains 256 continuous instances.
* `agent.py` contains two different dummy agents and a standard `DQNAgent` that has several differences from the one described in the original paper. These differences arose due to experiments with various settings and the poor descriptions in the paper

