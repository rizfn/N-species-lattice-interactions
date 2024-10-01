import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

def main():
    N = 50
    L = 1024
    theta = 0.1
    K = 2
    alive_tolerance = 1e-6

    file_list = glob(f"src/multipleNutrientsNecessary/outputs/populationTimeseriesLattice/N_{N}_L_{L}_theta_{theta}_K_{K}_*.csv")

    n_files = len(file_list)

    fig, axs = plt.subplots(n_files // (n_files//2), n_files//2, figsize=(12, 10))
    axs = axs.flatten()
    
    for fileidx, file in enumerate(file_list):

        data = pd.read_csv(file)

        colors = plt.cm.rainbow(np.linspace(0, 1, N))

        axs[fileidx].plot(data['step'], data['empty'], label="empty", c='grey', linestyle='--')
        for i in range(1, N + 1):
            axs[fileidx].plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

        alive_species = (data.iloc[-1, 2:N+2] > alive_tolerance).sum()

        axs[fileidx].set_title(f"{L=}, {N=}, {theta=}; {alive_species} survive")
        axs[fileidx].set_xlabel(r"Timestep")
        axs[fileidx].set_ylabel("Fraction of lattice points")
        # axs[fileidx].legend()
        axs[fileidx].grid()

    plt.tight_layout()

    plt.savefig(f'src/multipleNutrientsNecessary/plots/populationTimeseriesLattice/N_{N}_L_{L}_theta_{theta}_K_{K}.png', dpi=300)
    plt.show()


def viz_shimadaAnd():
    N_species = 200
    N_chemicals = 50
    L = 1024
    theta = 0.1
    K = 2
    alive_tolerance = 1e-6

    file_list = glob(f"src/multipleNutrientsNecessary/outputs/populationTimeseriesLatticeDiffchem/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}_*.csv")

    n_files = len(file_list)

    fig, axs = plt.subplots(n_files // (n_files//2), n_files//2, figsize=(12, 10))
    axs = axs.flatten()

    fig.suptitle(f"{L=}, {N_species=}, {N_chemicals=}, {theta=}, {K=}")
    
    for fileidx, file in enumerate(file_list):

        data = pd.read_csv(file)

        colors = plt.cm.rainbow(np.linspace(0, 1, N_species))

        # axs[fileidx].plot(data['step'], data['empty'], label="empty", c='grey', linestyle='--')
        for i in range(1, N_species + 1):
            axs[fileidx].plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

        alive_species = (data.iloc[-1, 2:N_species+2] > alive_tolerance).sum()

        axs[fileidx].set_title(f"{alive_species} survive")
        axs[fileidx].set_xlabel(r"Timestep")
        axs[fileidx].set_ylabel("Fraction of lattice points")
        # axs[fileidx].legend()
        axs[fileidx].grid()

    plt.tight_layout()

    plt.savefig(f'src/multipleNutrientsNecessary/plots/populationTimeseriesLatticeDiffchem/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    # main()
    viz_shimadaAnd()