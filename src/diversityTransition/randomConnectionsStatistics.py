import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm

@njit
def ode_derivatives(S, X, growth_rate, reaction_rate, connectivity):
    dS = np.zeros_like(S)
    dX = np.zeros_like(X)

    # Update resources
    for i in range(len(S)):
        resource_consumption = 0.0
        for j in range(len(X)):
            for k in range(len(X)):
                if connectivity[j, k] == i:  # Check if there is a reaction using resource i
                    resource_consumption += reaction_rate * X[j] * X[k]
        dS[i] = growth_rate * S[i] - resource_consumption * S[i]

    # Update chemicals
    for i in range(len(X)):
        reaction_sum = 0.0
        for j in range(len(X)):
            if connectivity[j, i] >= 0:  # Check if there is a reaction
                resource_index = connectivity[j, i]  # Get the resource index (0-based)
                reaction_sum += reaction_rate * X[j] * X[i] * S[resource_index]
        dX[i] = reaction_sum

    return dS, dX

@njit
def ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rate, connectivity, S0, X0, stoptime=100, nsteps=1000, dataskip=1):
    dt = stoptime / nsteps
    n_records = nsteps // dataskip + 1

    T = np.zeros(n_records)
    S_vals = np.zeros((N_s, n_records))
    X_vals = np.zeros((N_c, n_records))

    S_vals[:, 0] = S0
    X_vals[:, 0] = X0

    S, X = S0.copy(), X0.copy()
    record_idx = 1

    for i in range(nsteps):
        k1_S, k1_X = ode_derivatives(S, X, growth_rate, reaction_rate, connectivity)

        S_temp = S + 0.5 * dt * k1_S
        X_temp = X + 0.5 * dt * k1_X
        k2_S, k2_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rate, connectivity)

        S_temp = S + 0.5 * dt * k2_S
        X_temp = X + 0.5 * dt * k2_X
        k3_S, k3_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rate, connectivity)

        S_temp = S + dt * k3_S
        X_temp = X + dt * k3_X
        k4_S, k4_X = ode_derivatives(S_temp, X_temp, growth_rate, reaction_rate, connectivity)

        S += (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        X += (dt / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Enforce carrying capacity
        total = np.sum(S) + np.sum(X)
        S /= total
        X /= total

        if i % dataskip == 0:
            S_vals[:, record_idx] = S
            X_vals[:, record_idx] = X
            T[record_idx] = i * dt
            record_idx += 1

    return T, S_vals, X_vals

# @njit
def run_one_simulation(N_s, N_c, growth_rate, reaction_rate, sparsity, survival_threshold, stoptime, nsteps, dataskip):
    connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
    mask = np.random.rand(N_c, N_c) < sparsity
    connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
    for i in range(N_c):
        connectivity[i, i] = -1  # prevent self-catalyzation

    S0 = np.random.rand(N_s)
    X0 = np.random.rand(N_c)
    total = np.sum(S0) + np.sum(X0)
    S0 /= total
    X0 /= total

    T, S_vals, X_vals = ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rate, connectivity, S0, X0, stoptime, nsteps, dataskip)

    N_surviving_species = np.sum(X_vals[:, -1] > survival_threshold)

    return N_surviving_species

def main():
    N_simulations = 100

    N_s = 10
    N_c = 100
    reaction_rate = 1
    sparsity = 0.8  # Fraction of the connectivity matrix that should be set to -1
    survival_threshold = 0.01 * 1 / (N_s + N_c)
    stoptime = 10000000
    nsteps = 10000
    dataskip = 10

    growth_rate_list = np.geomspace(1e-4, 1e-1, 4)
    N_surviving_species_mean = np.zeros_like(growth_rate_list)
    N_surviving_species_std = np.zeros_like(growth_rate_list)

    for i in range(len(growth_rate_list)):
        N_surviving_species = np.zeros(N_simulations)
        for sim in tqdm(range(N_simulations)):
            N_surviving_species[sim] = run_one_simulation(N_s, N_c, growth_rate_list[i], reaction_rate, sparsity, survival_threshold, stoptime, nsteps, dataskip)
        N_surviving_species_mean[i] = np.mean(N_surviving_species)
        N_surviving_species_std[i] = np.std(N_surviving_species) / np.sqrt(N_simulations)

    plt.errorbar(growth_rate_list, N_surviving_species_mean, N_surviving_species_std, capsize=10)
    plt.xlabel('Growth rate')
    plt.ylabel('Number of surviving species')
    plt.xscale('log')
    plt.title(f"{N_s} species, {N_c} chemicals")
    plt.grid()
    plt.savefig(f'src/diversityTransition/plots/diversityVsGrowthrate_{N_s}-{N_c}_stoptime_{stoptime}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()