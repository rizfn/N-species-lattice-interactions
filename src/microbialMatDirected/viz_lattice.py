import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ast
import os

def parse_lattice(s):
    return ast.literal_eval(s)

def create_plasma_colormap(N):
    """Create a colormap with white for empty (0) and plasma colors for species (1-N)"""
    plasma_colors = plt.cm.plasma(np.linspace(0, 1, N))
    colors = np.vstack([[1, 1, 1, 1], plasma_colors])  # White + plasma colors
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=N)
    return cmap, norm

def visualize_lattice(filepath, N, title_params, output_dir, output_filename):
    """Common visualization function for all lattice types"""
    # Create colormap
    cmap, norm = create_plasma_colormap(N)
    
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
    ax.set_title(f'Final Lattice State (Step: {step})\n{title_params}')
    
    # Add colorbar with species labels
    cbar = plt.colorbar(img, ax=ax, ticks=range(N+1))
    labels = ['Empty'] + [f'Species {i}' for i in range(1, N+1)]
    cbar.ax.set_yticklabels(labels)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{output_filename}', dpi=300, bbox_inches='tight')

def viz_final_lattice(N, L, theta, D, S):
    filepath = f"src/microbialMatDirected/outputs/chainLatticeSnapshots/N_{N}_L_{L}_theta_{theta}_D_{D}_S_{S}.tsv"
    title_params = f"N={N} species chain, θ={theta}, D={D}, S={S}"
    output_dir = "src/microbialMatDirected/plots/chainLattice"
    output_filename = f"N_{N}_theta_{theta}_D_{D}_S_{S}.png"
    
    visualize_lattice(filepath, N, title_params, output_dir, output_filename)

def viz_final_lattice_directedFlow(N, L, theta, D, S, v):
    filepath = f"src/microbialMatDirected/outputs/chainDirectedFlowSnapshots/N_{N}_L_{L}_theta_{theta}_D_{D}_v_{v}_S_{S}.tsv"
    title_params = f"N={N} species chain, θ={theta}, D={D}, S={S}"
    output_dir = "src/microbialMatDirected/plots/directedFlowLattice"
    output_filename = f"N_{N}_theta_{theta}_D_{D}_S_{S}_v_{v}.png"
    
    visualize_lattice(filepath, N, title_params, output_dir, output_filename)

def viz_final_lattice_chemotaxis_flow(N, L, theta, D, S, v):
    filepath = f"src/microbialMatDirected/outputs/chainChemotaxisFlowLattice/N_{N}_L_{L}_theta_{theta}_D_{D}_v_{v}_S_{S}.tsv"
    title_params = f"N={N} species chain, θ={theta}, D={D}, S={S}, v={v}"
    output_dir = "src/microbialMatDirected/plots/chemotaxisFlowLattice"
    output_filename = f"N_{N}_theta_{theta}_D_{D}_S_{S}_v_{v}.png"
    
    visualize_lattice(filepath, N, title_params, output_dir, output_filename)

def viz_final_rect_lattice_chemotaxis_flow(N, Lx, Ly, theta, D, S, v):
    filepath = f"src/microbialMatDirected/outputs/chainChemotaxisFlowLattice/N_{N}_Lx_{Lx}_Ly_{Ly}_theta_{theta}_D_{D}_v_{v}_S_{S}.tsv"
    title_params = f"N={N} species lattice, θ={theta}, D={D}, S={S}, v={v}"
    output_dir = "src/microbialMatDirected/plots/chemotaxisFlowRectLattice"
    output_filename = f"N_{N}_theta_{theta}_D_{D}_S_{S}_v_{v}.png"
    
    visualize_lattice(filepath, N, title_params, output_dir, output_filename)

if __name__ == "__main__":
    # viz_final_lattice(2, 256, 0.01, 0.25, 100)
    # viz_final_lattice_directedFlow(3, 256, 0.01, 0.25, 100, 0.5)
    # viz_final_lattice_chemotaxis_flow(3, 256, 0.01, 0.25, 1, 1)
    viz_final_rect_lattice_chemotaxis_flow(3, 32, 128, 0.01, 0.25, 100, 0.5)
