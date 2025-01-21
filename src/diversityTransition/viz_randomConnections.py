import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

def main():
    N_resources = 10
    N_chemicals = 100
    gamma = 10
    sparsity = 0.8
    alive_tolerance = 1e-3 / N_chemicals

    file_list = glob(f"src/diversityTransition/outputs/randomConnections/N_{N_resources}-{N_chemicals}_gamma_{gamma}_sparsity_{sparsity}_*.csv")

    n_files = len(file_list)

    fig, axs = plt.subplots(n_files // (n_files//2), n_files//2, figsize=(15, 10))
    axs = axs.flatten()
    
    for fileidx, file in enumerate(file_list):

        data = pd.read_csv(file)

        colors = plt.cm.rainbow(np.linspace(0, 1, N_chemicals))

        for i in range(1, N_chemicals + 1):
            axs[fileidx].plot(data['time'], data[f'chemical{i}'], label=f"chemical{i}", c=colors[i % len(colors)])

        alive_species = (data.iloc[-1, 1:N_resources+1] > alive_tolerance).sum()

        axs[fileidx].set_title(f"{N_resources} resource, {N_chemicals} chemicals, $\gamma$={gamma}; sparsity={sparsity}\n{alive_species} survive")
        axs[fileidx].set_xlabel(r"Time")
        axs[fileidx].set_ylabel("Population")

        # axs[fileidx].legend()
        axs[fileidx].grid()

    plt.tight_layout()

    plt.savefig(f'src/diversityTransition/plots/randomConnectionsCPP/N_{N_resources}-{N_chemicals}_gamma_{gamma}_sparsity_{sparsity}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    main()