import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

class TreeGridWorld(gym.Env):
    def __init__(self,rmin=10**(-6),rmax=0.3,num_trees=10):
        self.grid_size = 10
        self.agent_position = [0, 0]
        self.num_trees = num_trees
        self.current_num_trees = num_trees
        self.rmin = rmin
        self.rmax = rmax

        # Define action and observation space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,self.grid_size+2, self.grid_size+2), dtype=np.uint8)

        # Initialize grid
        self.grid = np.zeros((2,self.grid_size+2, self.grid_size+2), dtype=np.uint8)
        self.grid[0,self.agent_position[0], self.agent_position[1]] = 1  # Agent

        # Randomly distribute trees
        tree_positions = random.sample([(x, y) for x in range(1, self.grid_size+1) for y in range(1, self.grid_size+1)], self.num_trees)
        for pos in tree_positions:
            self.grid[1,pos[0], pos[1]] = 1  # Tree

    def step(self, action):
        # Clear agent's current position
        self.grid[0,self.agent_position[0], self.agent_position[1]] = 0

        # Execute action
        if action == 0 and self.agent_position[1]<self.grid_size+1:  # Up
            self.agent_position[1] += 1
        elif action == 1 and self.agent_position[1] > 1:  # Down
            self.agent_position[1] -= 1
        elif action == 2 and self.agent_position[0]<self.grid_size+1:  # Right
            self.agent_position[0] += 1
        elif action == 3 and self.agent_position[0] > 1:  # Left
            self.agent_position[0] -= 1
        elif action == 4:  # Stay
            pass



        # Update agent's new position
        self.grid[0,self.agent_position[0], self.agent_position[1]] = 1

        # Check for tree at new position
        reward = 0
        if self.grid[1,self.agent_position[0], self.agent_position[1]] == 1:
            self.grid[1,self.agent_position[0], self.agent_position[1]] = 0  # Remove tree
            self.current_num_trees -= 1
            reward = 1

        #calculate tree respawn rate
        r = max(self.rmin,self.rmax*np.log(self.current_num_trees+1)/np.log(self.num_trees+11))

        # If there are fewer than max trees, spawn a new tree
        if self.current_num_trees < self.num_trees and np.random.random() < r:  
            empty_positions = np.argwhere(self.grid[1,:,:] == 0)

            #only place trees in reacheble areas
            mask = np.logical_or(np.logical_or(empty_positions[:,0] ==0,empty_positions[:,1] ==0)
            ,np.logical_or(empty_positions[:,0] ==self.grid_size+1,empty_positions[:,1] ==self.grid_size+1))
            empty_positions =  empty_positions[mask==False]

            new_tree_position = empty_positions[np.random.choice(len(empty_positions))]
            new_tree_position +=1
            self.grid[1,new_tree_position[0], new_tree_position[1]] = 1  # New tree
            self.current_num_trees= np.sum(self.grid[1,:,:] == 1)

        done = False
        info = {}

        return self.grid.copy(), reward, done, info

    def reset(self):
        self.__init__(self.rmin,self.rmax,self.num_trees)
        return self.grid.copy()
    
    def render(self):

        # Value 0 represents an empty cell, 1 represents the agent, and 2 represents a tree.
        display_grid = np.zeros((self.grid_size+2, self.grid_size+2), dtype=int)
        display_grid[self.grid[0,:,:] == 1] = 1
        display_grid[self.grid[1,:,:] == 1] = 2

        # Use matshow to display the grid
        plt.matshow(display_grid, cmap=ListedColormap(['white', 'blue', 'green']), vmin=0, vmax=2)

        # Set up the colorbar
        cbar = plt.colorbar(ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Empty', 'Agent', 'Tree'])
        plt.show()

    def render_cv(self):
        # Create a 2D array to display the grid. 
        # Value 0 represents an empty cell, 1 represents the agent, and 2 represents a tree.
        display_grid = np.zeros((self.grid_size+2, self.grid_size+2), dtype=int)
        display_grid[self.grid[0,:,:] == 1] = 1
        display_grid[self.grid[1,:,:] == 1] = 2

        # Scale up the display grid for better visibility
        display_grid = cv2.resize(display_grid, (500, 500), interpolation = cv2.INTER_NEAREST)

        # Assign colors for each cell type. Let's say empty cells are white, agent cells are blue, and tree cells are green.
        color_map = {
            0: [255, 255, 255],  # White for empty cells
            1: [255, 0, 0],      # Blue for the agent
            2: [0, 255, 0],      # Green for trees
        }

        # Apply color map
        color_grid = np.zeros((500, 500, 3), dtype=np.uint8)
        for val, color in color_map.items():
            color_grid[display_grid == val] = color

        # Display the environment using OpenCV
        cv2.imshow("TreeGridWorld", color_grid)
        cv2.waitKey(1)  # Display the image for 1 millisecond

