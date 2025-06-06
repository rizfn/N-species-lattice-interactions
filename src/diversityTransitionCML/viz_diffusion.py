import matplotlib.pyplot as plt
import numpy as np
import os

L = 100  # Size of the lattice
N_STEPS = 1000  # Number of timesteps

def load_lattice(filename):
    return np.loadtxt(filename)

def visualize_lattice():
    plt.ion()
    fig, ax = plt.subplots()

    for t in range(N_STEPS):
        filename = f"src/diversityTransitionCML/outputs/lattice/{t}.txt"
        if os.path.exists(filename):
            lattice = load_lattice(filename)
            ax.imshow(lattice, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Time step: {t}")
            plt.draw()
            plt.pause(0.01)
            ax.cla()
        else:
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    visualize_lattice()