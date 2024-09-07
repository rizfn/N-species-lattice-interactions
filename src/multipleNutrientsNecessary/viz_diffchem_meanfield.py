import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

def main():
    N_species = 200
    N_chemicals = 70
    theta = 0.01
    K = 2
    alive_tolerance = 1e-6

    file_list = glob(f"src/multipleNutrientsNecessary/outputs/diffchemTimeseries/N_{N_species}-{N_chemicals}_theta_{theta}_K_{K}_*.csv")

    n_files = len(file_list)

    fig, axs = plt.subplots(n_files // (n_files//2), n_files//2, figsize=(50, 20))
    axs = axs.flatten()
    
    for fileidx, file in enumerate(file_list):

        data = pd.read_csv(file)

        colors = plt.cm.rainbow(np.linspace(0, 1, N_species))

        for i in range(1, N_species + 1):
            axs[fileidx].plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)], marker='.', linestyle='')

        alive_species = (data.iloc[-1, 1:N_species+1] > alive_tolerance).sum()

        # axs[fileidx].set_title(f"{N_species} species, {N_chemicals} chemicals, {theta=}; {alive_species} survive")
        # axs[fileidx].set_xlabel(r"Timestep")
        # axs[fileidx].set_ylabel("Fraction of lattice points")
        # axs[fileidx].legend()
        axs[fileidx].grid()

    plt.tight_layout()

    plt.savefig(f'src/multipleNutrientsNecessary/plots/diffchemTimeseries/differentICsStatistics/N_{N_species}-{N_chemicals}_theta_{theta}_K_{K}.png', dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()