import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    N = 500
    L = 4096
    theta = 0.1
    alive_tolerance = 1e-6

    data = pd.read_csv(f"src/randomMatrix2D/outputs/populationTimeseries/cell2/N_{N}_L_{L}_theta_{theta}.csv")

    fig, axs = plt.subplots(figsize=(10, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, N))

    for i in range(1, N + 1):
        axs.plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

    alive_species = (data.iloc[-1, 1:N+1] > alive_tolerance).sum()

    axs.set_title(f"{L=}, {N=}, {theta=}; {alive_species} survive")
    axs.set_xlabel(r"Timestep")
    axs.set_ylabel("Fraction of lattice points")
    # axs.legend()
    axs.grid()

    plt.savefig(f'src/randomMatrix2D/plots/populationTimeseries/N_{N}_L_{L}_theta_{theta}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()