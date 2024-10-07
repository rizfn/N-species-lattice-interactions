import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

def main():
    N_species = 200
    N_chemicals = 50
    L = 1024
    theta = 0.1
    K = 2
    D = 1
    alive_tolerance = 1e-6

    file_list = glob(f"src/shimadaModel2D/outputs/timeseriesChemicalDiffusion/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}_*.csv")

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
    # plt.savefig(f'src/shimadaModel2D/plots/timeseriesChemicalDiffusion/CPU_N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}_D_{D}.png', dpi=300)
    plt.show()


def cuda():
    N_species = 200
    N_chemicals = 50
    L = 1024
    theta = 0.01
    K = 2
    D = 0
    alive_tolerance = 1e-6

    file = f"src/shimadaModel2D/outputs/timeseriesChemicalDiffusionCUDA/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}_D_{D}.csv"

    data = pd.read_csv(file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"L={L}, N_species={N_species}, N_chemicals={N_chemicals}, theta={theta}, K={K}")

    colors = plt.cm.rainbow(np.linspace(0, 1, N_species))

    # Plot the fraction of lattice points occupied by each species
    for i in range(1, N_species + 1):
        ax1.plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

    alive_species = (data.iloc[-1, 2:N_species+2] > alive_tolerance).sum()

    ax1.set_title(f"{alive_species} species survive")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Fraction of lattice points")
    ax1.grid()

    # Plot the number of surviving species as a function of time
    surviving_species = (data.iloc[:, 2:N_species+2] > alive_tolerance).sum(axis=1)
    ax2.plot(data['step'], surviving_species, label="Surviving species", c='blue')

    ax2.set_title("Number of surviving species over time")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Number of surviving species")
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f'src/shimadaModel2D/plots/timeseriesChemicalDiffusion/N_{N_species}-{N_chemicals}_L_{L}_theta_{theta}_K_{K}_D_{D}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # main()
    cuda()