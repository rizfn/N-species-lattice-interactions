import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    N = 50
    L = 512
    theta = 0.2

    data = pd.read_csv(f"src/randomMatrix2D/outputs/populationTimeseries/N_{N}_L_{L}_theta_{theta}.csv")

    fig, axs = plt.subplots(figsize=(10, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, N))

    for i in range(1, N + 1):
        axs.plot(data['step'], data[f'bacteria{i}'], label=f"bacteria{i}", c=colors[i % len(colors)])

    axs.set_title(f"{L=}, {N=}, {theta=}")
    axs.set_xlabel(r"Timestep")
    axs.set_ylabel("Fraction of lattice points")
    # axs.legend()
    axs.grid()

    plt.savefig(f'src/randomMatrix2D/plots/populationTimeseries/N_{N}_L_{L}_theta_{theta}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()