import numpy as np
from numba import njit
import matplotlib.pyplot as plt

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

# @njit
# def ode_derivatives_minusS(S, X, growth_rate, reaction_rates, connectivity):
#     dS = growth_rate - S  # Basic growth rate for resources
#     dX = np.zeros_like(X)

#     # Iterate over every pair of chemicals
#     for j in range(len(X)):
#         for k in range(len(X)):
#             resource_index = connectivity[j, k]
#             if resource_index >= 0:  # Check if there is a reaction using a resource
#                 reaction_rate = reaction_rates[j, k]
#                 reaction_term = reaction_rate * X[j] * X[k] * S[resource_index]
#                 dS[resource_index] -= reaction_term
#                 dX[k] += reaction_term

#     return dS, dX


@njit
def ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime=100, nsteps=1000, dataskip=1):
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

        if i % dataskip == 0:
            S_vals[:, record_idx] = S
            X_vals[:, record_idx] = X
            T[record_idx] = i * dt
            record_idx += 1

    return T, S_vals, X_vals

def main():
    N_s = 10
    N_c = 100
    # growth_rate = 0.01 / N_s
    growth_rate = 0.01
    alpha_min = 0.5
    alpha_max = 1
    sparsity = 0.9  # Fraction of the connectivity matrix that should be set to -1
    survival_threshold = 0.01 * 1 / N_c

    connectivity = np.random.randint(0, N_s, (N_c, N_c))  # Random connectivity matrix with resource indices
    mask = np.random.rand(N_c, N_c) < sparsity
    connectivity[mask] = -1  # Set a fraction of the matrix to -1 based on sparsity
    for i in range(N_c):
        connectivity[i, i] = -1  # prevent self-catalyzation

    reaction_rates = np.random.uniform(alpha_min, alpha_max, (N_c, N_c))  # Random reaction rates in the interval (0.5, 1)

    S0 = np.random.rand(N_s)
    X0 = np.random.rand(N_c)
    total = np.sum(S0) + np.sum(X0)
    S0 /= total
    X0 /= total

    stoptime = 1000000
    nsteps = 10000000
    dataskip = 100

    T, S_vals, X_vals = ode_integrate_rk4(N_s, N_c, growth_rate, reaction_rates, connectivity, S0, X0, stoptime, nsteps, dataskip)

    N_surviving_species = np.sum(X_vals[:, -1] > survival_threshold)

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(N_s):
        axs[0].plot(T, S_vals[i, :], label=f'S{i+1}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Resource Values')
    axs[0].set_title('Resources')
    axs[0].grid(True)

    for i in range(N_c):
        axs[1].plot(T, X_vals[i, :], label=f'X{i+1}')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Chemical Values')
    axs[1].set_title('Chemicals')
    axs[1].grid(True)

    plt.suptitle(f'{N_s} resources, {N_c} chemicals, $\gamma$={growth_rate}, $\\alpha\in[${alpha_min},{alpha_max}$]$, sparsity={sparsity}\n{N_surviving_species} survive')

    plt.tight_layout()
    plt.savefig(f'src/diversityTransition/plots/randomConnections(1-S)/N_{N_s}-{N_c}_gamma_{growth_rate}_alpha_{alpha_min}-{alpha_max}_sparsity_{sparsity}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()