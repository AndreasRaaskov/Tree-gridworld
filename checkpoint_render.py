
import cv2
import matplotlib.pyplot as plt
import numpy as np

path_name="checkpoints/c_24000"

def render_state(grid):
    # Create a 2D array to display the grid. 
        # Value 0 represents an empty cell, 1 represents the agent, and 2 represents a tree.
        display_grid = np.zeros((12,12), dtype=int)
        display_grid[grid[0,:,:] == 1] = 1
        display_grid[grid[1,:,:] == 1] = 2

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

states = np.load(path_name+"_state.npy")
#rewards = np.load(path_name+"_reward.npy")

for i in range(len(states)-200,len(states)):
    render_state(states[i])

cv2.destroyAllWindows()


