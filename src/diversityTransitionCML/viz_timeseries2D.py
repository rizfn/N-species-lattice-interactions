import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_one(N_resources, N_chemicals, L, gamma, diffusion_coeff, sparsity):

    alive_tolerance = 1e-3 / N_chemicals

    filepath = f"src/diversityTransitionCML/outputs/timeseries2D/N_{N_resources}-{N_chemicals}_L_{L}_gamma_{gamma}_D_{diffusion_coeff}.csv"

    data = pd.read_csv(filepath)

    colors_resources = plt.cm.rainbow(np.linspace(0, 1, N_resources))
    colors_chemicals = plt.cm.rainbow(np.linspace(0, 1, N_chemicals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(N_resources):
        ax1.plot(data['time'], data[f'resource{i}'], label=f"resource{i}", c=colors_resources[i % len(colors_resources)])

    for i in range(N_chemicals):
        ax2.plot(data['time'], data[f'chemical{i}'], label=f"chemical{i}", c=colors_chemicals[i % len(colors_chemicals)])

    alive_species = (data.iloc[-1, 1:N_resources+1] > alive_tolerance).sum()

    plt.suptitle(f"{N_resources} resource, {N_chemicals} chemicals, $\gamma$={gamma}; sparsity={sparsity}\n{alive_species} survive")

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"Resource concentration")
    ax1.grid()
    ax1.legend()
    ax2.set_xlabel(r"Time")
    ax2.set_ylabel(r"Chemical concentration")
    ax2.grid()
    ax2.legend()


    plt.tight_layout()

    # plt.savefig(f'src/diversityTransitionCML/plots/timeseries2D/N_{N_resources}-{N_chemicals}_gamma_{gamma}_sparsity_{sparsity}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    # plot_one(10, 100, 100, 0.01, 0.1, 0.9)
    # plot_one(10, 100, 100, 0, 0, 0.9)
    plot_one(1, 2, 100, 0, 0, 0.9)
    