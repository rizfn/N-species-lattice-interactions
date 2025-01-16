import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

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
def ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime=100, nsteps=1000):
    dt = stoptime / nsteps

    S, X = S0.copy(), X0.copy()

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

        # Normalize only the chemicals
        X /= np.sum(X)

    return S, X

def run_one_simulation(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, survival_threshold, stoptime, nsteps):
    S, X = ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime, nsteps)

    N_surviving_species = np.sum(X > survival_threshold)

    return N_surviving_species

def worker(args):
    return run_one_simulation(*args)

def main():
    N_simulations = 100
    N_s = 10
    N_c = 100
    alpha_min = 0.5
    alpha_max = 1
    sparsity = 0.8  # Fraction of the connectivity matrix that should be set to -1
    survival_threshold = 0.01 * 1 / N_c
    nsteps = 10000000

    growth_rate_list = np.geomspace(1e-6, 1e3, 10)
    N_surviving_species_mean = np.zeros_like(growth_rate_list)
    N_surviving_species_std = np.zeros_like(growth_rate_list)

    # Generate connectivity, reaction rates, and initial conditions for each simulation
    connectivity_list = []
    reaction_rates_list = []
    S0_list = []
    X0_list = []
    for _ in range(N_simulations):
        connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
        mask = np.random.rand(N_c, N_c) < sparsity
        connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
        for i in range(N_c):
            connectivity[i, i] = -1  # prevent self-catalyzation

        reaction_rates = np.random.uniform(alpha_min, alpha_max, (N_c, N_c))  # Random reaction rates in the interval (0.5, 1)
        connectivity_list.append(connectivity)
        reaction_rates_list.append(reaction_rates)

        S0 = np.random.rand(N_s)
        X0 = np.random.rand(N_c)
        X0 /= np.sum(X0)  # Normalize only the chemicals
        S0_list.append(S0)
        X0_list.append(X0)

    # Prepare arguments for parallel processing
    args_list = []
    for growth_rate in growth_rate_list:
        stoptime = 100000 / growth_rate  # Adjust stoptime based on growth rate
        for i in range(N_simulations):
            args_list.append((N_s, N_c, growth_rate, reaction_rates_list[i], connectivity_list[i], S0_list[i], X0_list[i], survival_threshold, stoptime, nsteps))

    # Run simulations in parallel
    with mp.Pool(mp.cpu_count() - 2) as pool:
        results = list(tqdm(pool.imap(worker, args_list), total=len(args_list)))

    # Collect results
    for i, growth_rate in enumerate(growth_rate_list):
        N_surviving_species = results[i * N_simulations:(i + 1) * N_simulations]
        N_surviving_species_mean[i] = np.mean(N_surviving_species)
        N_surviving_species_std[i] = np.std(N_surviving_species) / np.sqrt(N_simulations)

    # Plot results
    plt.errorbar(growth_rate_list, N_surviving_species_mean, N_surviving_species_std, capsize=10)
    plt.xlabel('Growth rate')
    plt.ylabel('Number of surviving species')
    plt.xscale('log')
    plt.title(f"{N_s} species, {N_c} chemicals, sparsity={sparsity}")
    plt.grid()
    plt.savefig(f'src/diversityTransition/plots/diversityVsGrowthrate_{N_s}-{N_c}_sparsity_{sparsity}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()