import numpy as np
from numba import njit
import csv
import os

@njit
def ode_derivatives(S, X, gamma, reaction_rates, connectivity):
    dS = gamma * (1 - S)  # Basic growth rate for resources
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
def ode_integrate_rk4(N_s, N_c, gamma, reaction_rates, connectivity, S0, X0, stoptime=100, nsteps=1000, dataskip=1):
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
        k1_S, k1_X = ode_derivatives(S, X, gamma, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k1_S
        X_temp = X + 0.5 * dt * k1_X
        k2_S, k2_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S_temp = S + 0.5 * dt * k2_S
        X_temp = X + 0.5 * dt * k2_X
        k3_S, k3_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S_temp = S + dt * k3_S
        X_temp = X + dt * k3_X
        k4_S, k4_X = ode_derivatives(S_temp, X_temp, gamma, reaction_rates, connectivity)

        S += (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        X += (dt / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Calculate growth rate
        pre_normalization_sum = np.sum(X)
        growth_rate_instant = (pre_normalization_sum - 1) / dt

        # Normalize only the chemicals
        X /= pre_normalization_sum

        if i % dataskip == 0:
            S_vals[:, record_idx] = S
            X_vals[:, record_idx] = X
            T[record_idx] = i * dt
            growth_rates[record_idx] = growth_rate_instant
            record_idx += 1

    return T, S_vals, X_vals, growth_rates

def save_timeseries(T, X_vals, growth_rates, output_dir):
    output_file = os.path.join(output_dir, 'timeseries.csv')
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['time', 'growthRate'] + [f'chemical{i}' for i in range(X_vals.shape[0])]
        writer.writerow(header)
        for i, t in enumerate(T):
            row = [t, growth_rates[i]] + list(X_vals[:, i])
            writer.writerow(row)

def save_matrix(matrix, filename, output_dir):
    output_file = os.path.join(output_dir, filename)
    np.savetxt(output_file, matrix, delimiter=',')

def main():
    N_s = 10
    N_c = 100
    alpha_min = 0.5
    alpha_max = 1
    sparsity = 0.9  # Fraction of the connectivity matrix that should be set to -1
    gamma = 0.1
    stoptime = 10_000_0
    nsteps = 10_000_0
    dataskip = 100

    connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
    reaction_rates = np.random.uniform(alpha_min, alpha_max, (N_c, N_c))  # Random reaction rates in the interval (0.5, 1)
    mask = np.random.rand(N_c, N_c) < sparsity
    connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
    reaction_rates[mask] = 0  # Set the reaction rates to 0 where the connectivity matrix is -1
    for i in range(N_c):
        connectivity[i, i] = -1  # prevent self-catalyzation
        reaction_rates[i, i] = 0  # prevent self-catalyzation


    S0 = np.zeros(N_s)
    X0 = np.random.rand(N_c)
    X0 /= np.sum(X0)  # Normalize the initial chemical values

    T, S_vals, X_vals, growth_rates = ode_integrate_rk4(N_s, N_c, gamma, reaction_rates, connectivity, S0, X0, stoptime, nsteps, dataskip)

    output_dir = f'docs/data/diversityTransition/timeseries/N_{N_s}-{N_c}_sparsity_{sparsity}_gamma_{gamma}'
    os.makedirs(output_dir, exist_ok=True)

    # Save timeseries data
    save_timeseries(T, X_vals, growth_rates, output_dir)

    # Save connectivity matrix
    save_matrix(connectivity, 'connectivity_matrix.csv', output_dir)

    # Save reaction rates matrix
    save_matrix(reaction_rates, 'reaction_rates_matrix.csv', output_dir)

if __name__ == '__main__':
    main()