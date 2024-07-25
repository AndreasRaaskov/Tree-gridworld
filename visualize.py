import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_log(log):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(log['episode'],log['reward'],color='blue',label='Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')


    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(log['episode'],log['trees'],color='green',label='Trees left')
    ax2.set_ylabel('Trees left')
    ax2.set_ylim(0, 10)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    return fig

def render_log(log):
    fig = plot_log(log)
    
    fig.canvas.draw()
    
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow('log',img)
    cv2.waitKey(1)
    
    