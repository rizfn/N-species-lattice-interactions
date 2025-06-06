import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import ast
import networkx as nx
import numpy as np
from tqdm import tqdm

def parse_lattice(s):
    return ast.literal_eval(s)

def load_matrix(filename):
    dictionary = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = int(parts[0])  # The first value is the key
            values = list(map(int, parts[1:]))  # The remaining values are the list of integers
            dictionary[key] = values
    return dictionary

def plot_network(ax, lattice, J, S, species_colors, chemical_colors, species_mapping):
    G = nx.DiGraph()

    # Count the population of each species
    unique, counts = np.unique(lattice, return_counts=True)
    species_population = dict(zip(unique, counts))

    # Exclude empty (0) and soil (1)
    surviving_species = [species for species in species_population if species > 1]


    # Add nodes for surviving species
    for species in surviving_species:
        G.add_node(int(species), size=species_population[species])

    J_survivors = {species: J[species-2] for species in J if species in surviving_species}  # -2 because the J,S matrices start from 0
    S_survivors = {species: S[species-2] for species in S if species in surviving_species}

    # Add edges based on J and S dictionaries
    for secreting_species, secreted_chemical_list in S_survivors.items():
        for chemical in secreted_chemical_list:
            for consuming_species, consuming_chemical_list in J_survivors.items():
                if chemical in consuming_chemical_list:
                    G.add_edge(secreting_species, consuming_species, weight=1, color=chemical_colors[chemical])
    
    pos = nx.circular_layout(G)  # Use circular layout
    sizes = [G.nodes[node]['size'] for node in G.nodes]
    weights = [G[u][v]['weight'] for u, v in G.edges]

    # Use the same colors for the network nodes as the lattice
    node_colors = [species_colors[species_mapping[node]] for node in G.nodes]
    edge_colors = [G[u][v]['color'] for u, v in G.edges]
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=sizes, width=weights, edge_color=edge_colors, node_color=node_colors, font_size=10, font_color='black')
    ax.set_title('Network of Surviving Species')


def update(frame, lines, img, ax_lattice, ax_network, J, S, species_colors, chemical_colours, species_mapping):
    step, lattice_str = lines[frame].split('\t')
    lattice = parse_lattice(lattice_str)

    # Recolor the lattice based on the species mapping
    recolored_lattice = np.copy(lattice)
    for species, mapped_index in species_mapping.items():
        recolored_lattice[lattice == species] = mapped_index

    img.set_array(recolored_lattice)
    ax_lattice.set_title(f'Step: {step}')

    # Clear the network axis and redraw the network
    ax_network.clear()
    plot_network(ax_network, lattice, J, S, species_colors, chemical_colours, species_mapping)

    return img, ax_network


def main():
    N_s = 200  # Number of species
    N_c = 50  # Number of chemicals
    L = 512
    theta = 0.1456
    K_in = 2
    K_out = 4
    D = 0.1
    sigma = 1
    sim_no = 0
    filepath = f"src/andModelSoil/outputs/lattice2D/N_{N_s}-{N_c}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}/lattice_{sim_no}.tsv"
    J_path = f"src/andModelSoil/outputs/lattice2D/N_{N_s}-{N_c}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}/J_{sim_no}.csv"
    S_path = f"src/andModelSoil/outputs/lattice2D/N_{N_s}-{N_c}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}/S_{sim_no}.csv"

    # Load J and S matrices
    J = load_matrix(J_path)
    S = load_matrix(S_path)

    with open(filepath, 'r') as file:
        lines = file.readlines()  # Skip the header
    total_frames = len(lines)
    step, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)

    # Count the number of surviving species
    unique_species = np.unique(lattice)
    surviving_species = [species for species in unique_species if species > 1]
    N_surviving = len(surviving_species)

    cmap_name = 'tab20'  # change to reflect the number of surviving species

    species_cmap = plt.get_cmap(cmap_name)
    species_colors = species_cmap(np.linspace(0, 1, N_surviving))
    combined_species_colors = np.vstack([[1, 1, 1, 1], [0.627, 0.322, 0.176, 1], species_colors])  # Add white and sienna
    species_cmap = mcolors.ListedColormap(combined_species_colors)
    species_norm = mcolors.Normalize(vmin=0, vmax=N_surviving + 1)

    # Create a mapping for surviving species to colormap indices
    species_mapping = {species: idx + 2 for idx, species in enumerate(surviving_species)}

    # Create a colormap for chemicals using HSV
    chemical_cmap = plt.get_cmap('hsv')
    chemical_colors = chemical_cmap(np.linspace(0, 1, N_c))

    # Recolor the lattice for the first frame
    recolored_lattice = np.copy(lattice)
    for species, mapped_index in species_mapping.items():
        recolored_lattice[lattice == species] = mapped_index

    fig, (ax_lattice, ax_network) = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    fig.suptitle(f'{N_surviving} species, $\\theta$={theta}')
    img = ax_lattice.imshow(recolored_lattice, cmap=species_cmap, norm=species_norm, interpolation='nearest')
    ax_lattice.set_title(f'Step: {step}')
    ax_lattice.invert_yaxis()  # Invert the y-axis

    # Initialize the network plot
    plot_network(ax_network, lattice, J, S, combined_species_colors, chemical_colors, species_mapping)
    plt.tight_layout()

    # plt.show(); return

    pbar = tqdm(total=total_frames)
    def update_with_progress(frame):
        pbar.update()
        return update(frame, lines, img, ax_lattice, ax_network, J, S, combined_species_colors, chemical_colors, species_mapping)

    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), blit=False)
    ani.save(f'src/andModelSoil/plots/latticeAnim/N_{N_s}-{N_c}_L_{L}_theta_{theta}_K_IN_{K_in}_K_OUT_{K_out}_D_{D}_sigma_{sigma}.gif', writer='ffmpeg', fps=30, dpi=200)
    pbar.close()

    
if __name__ == "__main__":
    main()