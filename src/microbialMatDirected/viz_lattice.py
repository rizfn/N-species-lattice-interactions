import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ast
import os

def parse_lattice(s):
    return ast.literal_eval(s)

def viz_final_lattice(N, L, theta, D, S):
    filepath = f"src/microbialMatDirected/outputs/chainLatticeSnapshots/N_{N}_L_{L}_theta_{theta}_D_{D}_S_{S}.tsv"
    
    # Create custom colormap: white for empty (0), plasma for species (1-N)
    plasma_colors = plt.cm.plasma(np.linspace(0, 1, N))
    colors = np.vstack([[1, 1, 1, 1], plasma_colors])  # White + plasma colors
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=N)
    
    # Read the file and get the last lattice
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    
    # Get the final lattice
    final_line = lines[-1]
    step, lattice_str = final_line.split('\t')
    lattice = parse_lattice(lattice_str)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Final Lattice State (Step: {step})\nN={N} species chain, θ={theta}, D={D}, S={S}')
    
    # Add colorbar with species labels
    cbar = plt.colorbar(img, ax=ax, ticks=range(N+1))
    labels = ['Empty'] + [f'Species {i}' for i in range(1, N+1)]
    cbar.ax.set_yticklabels(labels)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs('src/microbialMatDirected/plots/chainLattice', exist_ok=True)
    plt.savefig(f'src/microbialMatDirected/plots/chainLattice/N_{N}_theta_{theta}_D_{D}_S_{S}.png', 
                dpi=300, bbox_inches='tight')

def viz_final_lattice_directedFlow(N, L, theta, D, S, v):
    filepath = f"src/microbialMatDirected/outputs/chainDirectedFlowSnapshots/N_{N}_L_{L}_theta_{theta}_D_{D}_v_{v}_S_{S}.tsv"
    
    # Create custom colormap: white for empty (0), plasma for species (1-N)
    plasma_colors = plt.cm.plasma(np.linspace(0, 1, N))
    colors = np.vstack([[1, 1, 1, 1], plasma_colors])  # White + plasma colors
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=N)
    
    # Read the file and get the last lattice
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    
    # Get the final lattice
    final_line = lines[-1]
    step, lattice_str = final_line.split('\t')
    lattice = parse_lattice(lattice_str)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Final Lattice State (Step: {step})\nN={N} species chain, θ={theta}, D={D}, S={S}')
    
    # Add colorbar with species labels
    cbar = plt.colorbar(img, ax=ax, ticks=range(N+1))
    labels = ['Empty'] + [f'Species {i}' for i in range(1, N+1)]
    cbar.ax.set_yticklabels(labels)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs('src/microbialMatDirected/plots/directedFlowLattice', exist_ok=True)
    plt.savefig(f'src/microbialMatDirected/plots/directedFlowLattice/N_{N}_theta_{theta}_D_{D}_S_{S}_v_{v}.png', 
                dpi=300, bbox_inches='tight')

def viz_final_lattice_chemotaxis_flow(N, L, theta, D, S, v):
    filepath = f"src/microbialMatDirected/outputs/chainChemotaxisFlowLattice/N_{N}_L_{L}_theta_{theta}_D_{D}_v_{v}_S_{S}.tsv"
    
    # Create custom colormap: white for empty (0), plasma for species (1-N)
    plasma_colors = plt.cm.plasma(np.linspace(0, 1, N))
    colors = np.vstack([[1, 1, 1, 1], plasma_colors])  # White + plasma colors
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=N)
    
    # Read the file and get the last lattice
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    
    # Get the final lattice
    final_line = lines[-1]
    step, lattice_str = final_line.split('\t')
    lattice = parse_lattice(lattice_str)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Final Lattice State (Step: {step})\nN={N} species chain, θ={theta}, D={D}, S={S}, v={v}')
    
    # Add colorbar with species labels
    cbar = plt.colorbar(img, ax=ax, ticks=range(N+1))
    labels = ['Empty'] + [f'Species {i}' for i in range(1, N+1)]
    cbar.ax.set_yticklabels(labels)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs('src/microbialMatDirected/plots/chemotaxisFlowLattice', exist_ok=True)
    plt.savefig(f'src/microbialMatDirected/plots/chemotaxisFlowLattice/N_{N}_theta_{theta}_D_{D}_S_{S}_v_{v}.png', 
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # viz_final_lattice(2, 256, 0.01, 0.25, 100)
    # viz_final_lattice_directedFlow(3, 256, 0.01, 0.25, 100, 0.5)
    viz_final_lattice_chemotaxis_flow(3, 256, 0.01, 0.25, 1, 1)
