import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import multiprocessing

@njit
def ode_derivatives(S, X, gamma, reaction_rates, connectivity):
    dS = gamma * (1 - S)  # Basic growth rate for resources
    dX = np.zeros_like(X)

    # Iterate over every pair of chemicals
    for j in range(len(X)):
        for k in range(len(X)):
            resource_index = connectivity[j, k]
            if resource_index >= 0:  # Check if there is a reaction using a resource
                reaction_rate = reaction_rates[j, k]
                reaction_term = reaction_rate * X[j] * X[k] * S[resource_index]
                dS[resource_index] -= reaction_term
                dX[k] += reaction_term

    return dS, dX

@njit
def ode_integrate_rk4(N_s, N_c, gamma, reaction_rates, connectivity, S0, X0, stoptime=100, nsteps=1000, dataskip=1):
    dt = stoptime / nsteps
    n_records = nsteps // dataskip + 1

    T = np.zeros(n_records)
    S_vals = np.zeros((N_s, n_records))
    X_vals = np.zeros((N_c, n_records))
    growth_rates = np.zeros(n_records)

    S_vals[:, 0] = S0
    X_vals[:, 0] = X0

    S, X = S0.copy(), X0.copy()
    record_idx = 1

    for i in range(nsteps):
        k1_S, k1_X = ode_derivatives(S, X, gamma, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k1_S
        X_temp = X + 0.5 * dt * k1_X
        k2_S, k2_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k2_S
        X_temp = X + 0.5 * dt * k2_X
        k3_S, k3_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S_temp = S + dt * k3_S
        X_temp = X + dt * k3_X
        k4_S, k4_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S += (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        X += (dt / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Calculate growth rate
        pre_normalization_sum = np.sum(X)
        growth_rate_instant = (pre_normalization_sum - 1) / dt

        # Normalize only the chemicals
        X /= pre_normalization_sum

        if i % dataskip == 0:
            S_vals[:, record_idx] = S
            X_vals[:, record_idx] = X
            T[record_idx] = i * dt
            growth_rates[record_idx] = growth_rate_instant
            record_idx += 1

    return T, S_vals, X_vals, growth_rates

def plot_network(ax, surviving_species, mean_X_vals, connectivity, reaction_rates, N_s, colors):
    G = nx.DiGraph()

    # Add nodes with sizes related to their mean abundance
    for old_species in surviving_species:
        G.add_node(old_species, size=mean_X_vals[old_species])

    # Add edges with widths related to the reaction rates and colors based on connectivity
    edge_colors = []
    for old_j in surviving_species:
        for old_k in surviving_species:
            if connectivity[old_j, old_k] >= 0:
                G.add_edge(old_j, old_k, weight=2 * (reaction_rates[old_j, old_k])**2)
                edge_colors.append(connectivity[old_j, old_k])

    pos = nx.circular_layout(G)  # Use circular layout
    sizes = [G.nodes[node]['size'] * 5000 for node in G.nodes]  # Increase node size
    weights = [G[u][v]['weight'] for u, v in G.edges]  # Increase edge width

    # Create a categorical colormap with N_s discrete values
    cmap = plt.get_cmap('tab20', N_s)
    norm = mcolors.BoundaryNorm(boundaries=range(N_s+1), ncolors=N_s)

    # Draw the network with the new edge colors and node outlines
    node_edge_colors = [colors[node] for node in G.nodes]
    node_colors = [(r, g, b, 0.2) for r, g, b, _ in node_edge_colors]
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=sizes, width=weights, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=0, edge_vmax=N_s-1, node_color=node_colors, edgecolors=node_edge_colors, font_size=10, font_color='black')
    ax.set_title('Network of Surviving Species')

def run_simulation(args):
    gamma, stoptime, nsteps, connectivity, reaction_rates, S0, X0, N_s, N_c, dataskip = args
    T, S_vals, X_vals, growth_rates = ode_integrate_rk4(N_s, N_c, gamma, reaction_rates, connectivity, S0, X0, stoptime, nsteps, dataskip)

    # Calculate the mean for the last 10% of the timesteps
    last_10_percent_idx = int(0.9 * len(T))
    mean_X_vals = np.mean(X_vals[:, last_10_percent_idx:], axis=1)
    mean_growth_rate = np.mean(growth_rates[last_10_percent_idx:])

    N_surviving_species = np.sum(mean_X_vals > 0.01 * 1 / N_c)

    return T, X_vals, growth_rates, mean_X_vals, N_surviving_species

def main():
    N_s = 10
    N_c = 100
    alpha_min = 0.5
    alpha_max = 1
    sparsity = 0.9  # Fraction of the connectivity matrix that should be set to -1
    gamma = 100
    stoptime = 10_000_000
    nsteps = 10_000_000
    dataskip = 100
    N_simulations = 12

    connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
    mask = np.random.rand(N_c, N_c) < sparsity
    connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
    for i in range(N_c):
        connectivity[i, i] = -1  # prevent self-catalyzation

    reaction_rates = np.random.uniform(alpha_min, alpha_max, (N_c, N_c))  # Random reaction rates in the interval (0.5, 1)

    S0 = np.zeros(N_s)

    initial_conditions = [np.random.rand(N_c) for _ in range(N_simulations)]
    for X0 in initial_conditions:
        X0 /= np.sum(X0)  # Normalize the initial chemical values

    args_list = [(gamma, stoptime, nsteps, connectivity, reaction_rates, S0, X0, N_s, N_c, dataskip) for X0 in initial_conditions]

    with multiprocessing.Pool() as pool:
        results = pool.map(run_simulation, args_list)

    n_rows = (N_simulations + 2) // 3
    fig, axs = plt.subplots(n_rows, 6, figsize=(36, 6 * n_rows))

    colors = plt.cm.rainbow(np.linspace(0, 1, N_c))

    for i, (T, X_vals, growth_rates, mean_X_vals, N_surviving_species) in enumerate(results):
        row = i // 3
        col = (i % 3) * 2

        # Plot time series
        for j in range(N_c):
            axs[row, col].plot(T, X_vals[j, :], label=f'X{j+1}', c=colors[j % len(colors)])
        axs[row, col].plot(T, growth_rates, label='Growth Rate', linestyle='--', color='grey')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Chemical Values')
        axs[row, col].set_title(f'Simulation {i+1}: Chemicals')
        axs[row, col].grid(True)

        # Plot network of surviving species
        surviving_species = np.where(mean_X_vals > 0.01 * 1 / N_c)[0]
        plot_network(axs[row, col + 1], surviving_species, mean_X_vals, connectivity, reaction_rates, N_s, colors)

    plt.tight_layout()
    plt.savefig(f'src/diversityTransition/plots/differentICs/N_{N_s}-{N_c}_sparsity_{sparsity}_gamma_{gamma}.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()