import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ast
import os

def parse_lattice(s):
    return ast.literal_eval(s)

def create_combined_colormap(N):
    """Create a colormap for combined nutrients and bacteria lattice"""
    # Define the color scheme from your reference
    light_colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']
    dark_colors = ['green', 'blue', 'purple', 'red', 'darkgoldenrod']
    
    if N > len(light_colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    
    # Structure: white (empty) + light_colors (nutrients 1-N) + dark_colors (bacteria N+1-2N)
    colors = ['white'] + light_colors[:N] + dark_colors[:N]
    
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)
    return cmap, norm

def create_labels(N):
    """Create labels for the colorbar"""
    labels = ['Empty']
    
    # Add nutrient labels
    for i in range(1, N+1):
        labels.append(f'Nutrient {i}')
    
    # Add bacteria labels
    for i in range(1, N+1):
        labels.append(f'Bacteria {i}')
    
    return labels

def visualize_lattice(filepath, N, title_params, output_dir, output_filename):
    """Visualization function for combined nutrients and bacteria lattice"""
    # Create colormap and labels
    cmap, norm = create_combined_colormap(N)
    labels = create_labels(N)
    
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
    
    # Add colorbar with appropriate ticks and labels
    # Only show ticks for states that actually exist (0, 1-N, N+1-2N)
    tick_positions = [0] + list(range(1, N+1)) + list(range(N+1, 2*N+1))
    tick_labels = [labels[i] for i in tick_positions]
    
    cbar = plt.colorbar(img, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{output_filename}', dpi=300, bbox_inches='tight')

def viz_no_nutrient_lattice(N, L, theta):
    """Visualization function for the simplified no-nutrient lattice model"""
    filepath = f"src/microbialMatDirected/noNutrientLattice/outputs/lattice2D/N_{N}_L_{L}_theta_{theta}.tsv"
    title_params = f"N={N} species, θ={theta}"
    output_dir = "src/microbialMatDirected/noNutrientLattice/plots/lattice2D"
    output_filename = f"N_{N}_L_{L}_theta_{theta}.png"
    
    visualize_lattice(filepath, N, title_params, output_dir, output_filename)

if __name__ == "__main__":
    viz_no_nutrient_lattice(2, 256, 0.01)
