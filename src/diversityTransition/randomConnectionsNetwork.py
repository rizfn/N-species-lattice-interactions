import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

@njit
def ode_derivatives(S, X, growth_rate, reaction_rates, connectivity):
    dS = growth_rate * (1 - S)  # Basic growth rate for resources
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
def ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime=100, nsteps=1000, dataskip=1):
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
        k1_S, k1_X = ode_derivatives(S, X, growth_rate, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k1_S
        X_temp = X + 0.5 * dt * k1_X
        k2_S, k2_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k2_S
        X_temp = X + 0.5 * dt * k2_X
        k3_S, k3_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rates, connectivity)

        S_temp = S + dt * k3_S
        X_temp = X + dt * k3_X
        k4_S, k4_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rates, connectivity)

        S += (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        X += (dt / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Calculate growth rate
        pre_normalization_sum = np.sum(X)
        growth_rate_instant = pre_normalization_sum - 1

        # Normalize only the chemicals
        X /= pre_normalization_sum

        if i % dataskip == 0:
            S_vals[:, record_idx] = S
            X_vals[:, record_idx] = X
            T[record_idx] = i * dt
            growth_rates[record_idx] = growth_rate_instant
            record_idx += 1

    return T, S_vals, X_vals, growth_rates

def plot_network(ax, surviving_species, X_vals, connectivity, reaction_rates, N_s):
    G = nx.DiGraph()

    # Renumber the surviving species
    renumbered_species = {old: new for new, old in enumerate(surviving_species)}

    # Add nodes with sizes related to their abundance
    for old_species in surviving_species:
        new_species = renumbered_species[old_species]
        G.add_node(new_species, size=X_vals[old_species, -1])

    # Add edges with widths related to the reaction rates and colors based on connectivity
    edge_colors = []
    for old_j in surviving_species:
        for old_k in surviving_species:
            if connectivity[old_j, old_k] >= 0:
                new_j = renumbered_species[old_j]
                new_k = renumbered_species[old_k]
                G.add_edge(new_j, new_k, weight=2 * (reaction_rates[old_j, old_k])**2)
                edge_colors.append(connectivity[old_j, old_k])

    pos = nx.circular_layout(G)  # Use circular layout
    sizes = [G.nodes[node]['size'] * 5000 for node in G.nodes]  # Increase node size
    weights = [G[u][v]['weight'] for u, v in G.edges]  # Increase edge width

    # Create a categorical colormap with N_s discrete values
    cmap = plt.get_cmap('tab20', N_s)
    norm = mcolors.BoundaryNorm(boundaries=range(N_s+1), ncolors=N_s)

    # Draw the network with the new edge colors
    nx.draw(G, pos, with_labels=True, node_size=sizes, width=weights, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=0, edge_vmax=N_s-1, node_color='skyblue', font_size=10, font_color='black')
    ax.set_title('Network of Surviving Species')

def main():
    N_s = 10
    N_c = 100
    growth_rate = 100
    alpha_min = 0.5
    alpha_max = 1
    sparsity = 0.99  # Fraction of the connectivity matrix that should be set to -1
    survival_threshold = 0.01 * 1 / N_c

    connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
    mask = np.random.rand(N_c, N_c) < sparsity
    connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
    for i in range(N_c):
        connectivity[i, i] = -1  # prevent self-catalyzation

    reaction_rates = np.random.uniform(alpha_min, alpha_max, (N_c, N_c))  # Random reaction rates in the interval (0.5, 1)

    S0 = np.random.rand(N_s)
    X0 = np.random.rand(N_c)
    X0 /= np.sum(X0)  # Normalize the initial chemical values

    stoptime = 1_00_000
    nsteps = 10_000_000
    dataskip = 100

    T, S_vals, X_vals, growth_rates = ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime, nsteps, dataskip)

    N_surviving_species = np.sum(X_vals[:, -1] > survival_threshold)

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, N_c))

    for i in range(N_c):
        axs[0].plot(T, X_vals[i, :], label=f'X{i+1}', c=colors[i % len(colors)])
    axs[0].plot(T, growth_rates, label='Growth Rate', linestyle='--', color='grey')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Chemical Values')
    axs[0].set_title('Chemicals')
    axs[0].grid(True)

    plt.suptitle(f'{N_s} resources, {N_c} chemicals, $\gamma$={growth_rate}, $\\alpha\in[${alpha_min},{alpha_max}$]$, sparsity={sparsity}\n{N_surviving_species} survive')

    # Plot the network of surviving species
    surviving_species = np.where(X_vals[:, -1] > survival_threshold)[0]
    plot_network(axs[1], surviving_species, X_vals, connectivity, reaction_rates, N_s)

    plt.tight_layout()
    plt.savefig(f'src/diversityTransition/plots/network(1-S)/N_{N_s}-{N_c}_gamma_{growth_rate}_alpha_{alpha_min}-{alpha_max}_sparsity_{sparsity}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()