import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import networkx as nx
import matplotlib.colors as mcolors
import math

def plot_network(ax, surviving_species, J, S, colors):
    G = nx.DiGraph()

    # Add nodes for surviving species
    for species in surviving_species:
        G.add_node(int(species), size=100)  # Convert to int

    # Add edges based on J and S matrices
    edge_colors = []
    for species in surviving_species:
        species = int(species)  # Ensure species is an int
        if species in S:
            for chemical in S[species]:
                for consumer in J:
                    if chemical in J[consumer] and consumer in surviving_species:
                        G.add_edge(species, consumer, weight=1)
                        edge_colors.append(chemical)

    pos = nx.circular_layout(G)  # Use circular layout
    sizes = [G.nodes[node].get('size', 100) for node in G.nodes]  # Default size to 100 if not set
    weights = [G[u][v]['weight'] for u, v in G.edges]

    cmap = plt.get_cmap('tab20', len(colors))
    norm = mcolors.BoundaryNorm(boundaries=range(len(colors)+1), ncolors=len(colors))

    # Map chemical indices to colors
    edge_colors_mapped = [cmap(norm(chemical)) for chemical in edge_colors]

    node_colors = [colors[node % len(colors)] for node in G.nodes]
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=sizes, width=weights, edge_color=edge_colors_mapped, edge_cmap=cmap, edge_vmin=0, edge_vmax=len(colors)-1, node_color=node_colors, font_size=10, font_color='black')
    ax.set_title('Network of Surviving Species')

def load_matrix(filename):
    matrix = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = int(parts[0])
            values = list(map(int, parts[1:]))
            matrix[key] = values
    return matrix

def main():
    N_species = 200
    N_chemicals = 50
    L = 256
    theta = 0.1
    K_in = 2
    K_out = 4
    D = 0.1
    sigma = 1
    alive_tolerance = 1e-6

    folder_list = glob(f"src/andModelSoil/outputs/timeseries/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}")
    if len(folder_list) == 0:
        print("No folders found matching the specified pattern.")
        return

    folder = folder_list[0]  # Assuming there's only one folder
    timeseries_files = glob(f"{folder}/timeseries_*.csv")
    n_files = len(timeseries_files)

    # Calculate the number of rows and columns for the subplots
    n_cols = 2  # Each file has two plots (timeseries and network)
    n_rows = math.ceil(n_files / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols * 2, figsize=(12 * n_cols, 5 * n_rows))
    axs = np.atleast_2d(axs)  # Ensure axs is always a 2D array

    fig.suptitle(f"{L=}, {N_species=}, {N_chemicals=}, {theta=}, {K_in=}, {K_out=}, {D=}, {sigma=}")

    for fileidx, file in enumerate(timeseries_files):
        row = fileidx // n_cols
        col = (fileidx % n_cols) * 2

        data = pd.read_csv(file)
        colors = plt.cm.rainbow(np.linspace(0, 1, N_species))

        # Plot time series
        for i in range(1, N_species + 1):
            axs[row, col].plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

        alive_species = (data.iloc[-2, 2:N_species+2] > alive_tolerance).sum()
        axs[row, col].set_title(f"{alive_species} survive")
        axs[row, col].set_xlabel(r"Timestep")
        axs[row, col].set_ylabel("Fraction of lattice points")
        axs[row, col].grid()

        # Load J and S matrices
        run_idx = file.split('_')[-1].split('.')[0]
        J = load_matrix(f"{folder}/J_{run_idx}.csv")
        S = load_matrix(f"{folder}/S_{run_idx}.csv")

        # Identify surviving species
        surviving_species = np.where(data.iloc[-2, 2:N_species+2] > alive_tolerance)[0] + 2

        # Plot network diagram
        plot_network(axs[row, col + 1], surviving_species, J, S, colors)

    plt.tight_layout()
    plt.savefig(f'src/andModelSoil/plots/timeseries/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()