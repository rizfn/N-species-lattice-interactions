import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time


@njit
def set_random_ICs(N_species, N_chemicals, n_records):
    Os = np.zeros((N_species, n_records))
    Ns = np.zeros((N_chemicals, n_records))
    randomICs = np.random.rand(N_species)
    randomICs /= np.sum(randomICs)
    Os[:, 0] = randomICs
    randomICsN = np.random.rand(N_chemicals)
    Ns[:, 0] = randomICsN
    return Os, Ns


@njit
def ode_derivatives(O_vals, N_vals, J, S, theta):
    d_Os = np.zeros_like(O_vals)
    d_Ns = np.zeros_like(N_vals)

    empty_frac = 1 - np.sum(O_vals)

    for i in range(len(O_vals)):
        # Compute the sum of nutrient concentrations consumed by species O_i
        nutrient_sum_O = 0
        for j in range(len(N_vals)):
            nutrient_sum_O += N_vals[j] * J[i, j]

        nutrient_sum_O /= len(N_vals)  # Divide by the number of chemicals

        # Compute dO_i/dt
        d_Os[i] = O_vals[i] * empty_frac * nutrient_sum_O - theta * O_vals[i]

        for j in range(len(N_vals)):
            d_Ns[j] -= O_vals[i] * empty_frac * nutrient_sum_O * J[i, j]

    for i in range(len(O_vals)):
        for j in range(len(N_vals)):
            d_Ns[j] += S[i, j] * O_vals[i] * (1 - N_vals[j])

    return d_Os, d_Ns

@njit
def ode_integrate_rk4(N_species, N_chemicals, J, S, theta, stoptime=100_000, nsteps=100_000, dataskip=1):

    dt = stoptime / nsteps
    n_records = nsteps // dataskip + 1

    Os, Ns = set_random_ICs(N_species, N_chemicals, n_records)
    T = np.zeros(n_records)
    T[0] = 0

    current_Os, current_Ns = Os[:, 0], Ns[:, 0]
    record_idx = 1

    for i in range(nsteps):
        k1_Os, k1_Ns = ode_derivatives(current_Os, current_Ns, J, S, theta)

        O_temp = current_Os + 0.5 * dt * k1_Os
        N_temp = current_Ns + 0.5 * dt * k1_Ns
        k2_Os, k2_Ns = ode_derivatives(O_temp, N_temp, J, S, theta)

        O_temp = current_Os + 0.5 * dt * k2_Os
        N_temp = current_Ns + 0.5 * dt * k2_Ns
        k3_Os, k3_Ns = ode_derivatives(O_temp, N_temp, J, S, theta)

        O_temp = current_Os + dt * k3_Os
        N_temp = current_Ns + dt * k3_Ns
        k4_Os, k4_Ns = ode_derivatives(O_temp, N_temp, J, S, theta)

        current_Os = current_Os + (dt / 6) * (k1_Os + 2 * k2_Os + 2 * k3_Os + k4_Os)
        current_Ns = current_Ns + (dt / 6) * (k1_Ns + 2 * k2_Ns + 2 * k3_Ns + k4_Ns)

        if i % dataskip == 0:
            Os[:, record_idx] = current_Os
            Ns[:, record_idx] = current_Ns
            T[record_idx] = T[record_idx-1] + dataskip*dt
            record_idx += 1

    return T, Os, Ns


def main():

    N_species = 200
    N_chemicals = 5
    N_steps = 100_000
    N_sims = 16
    dataskip = 100

    theta = 0.1
    survival_threshold = 1e-3

    nrows = int(np.ceil(np.sqrt(N_sims)))
    ncols = int(np.ceil(N_sims / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
    axs = axs.flatten()

    for sim in range(N_sims):
        J = np.random.rand(N_species, N_chemicals)
        S = np.random.rand(N_species, N_chemicals)
        T, Os, Ns = ode_integrate_rk4(N_species, N_chemicals, J, S, theta, N_steps, N_steps, dataskip)
        surviving_counts = np.sum(Os > survival_threshold, axis=0)
        for i in range(N_species):
            axs[sim].plot(T, Os[i, :], label=f'Species {i+1}')
        axs[sim].set_xlabel('Time')
        axs[sim].set_ylabel('Prevalence of N species')
        axs[sim].grid(True)

        # Perform comparative advantage test
        surviving_Js = []
        surviving_Ss = []
        for i in range(N_species):
            if Os[i, -1] > survival_threshold:
                surviving_Js.append(J[i, :])
                surviving_Ss.append(S[i, :])

        has_comparative_advantage_J = False
        has_comparative_advantage_S = False

        for i in range(len(surviving_Js)):
            for j in range(i + 1, len(surviving_Js)):
                if np.all(surviving_Js[i] > surviving_Js[j]):
                    has_comparative_advantage_J = True
                if np.all(surviving_Ss[i] > surviving_Ss[j]):
                    has_comparative_advantage_S = True

        title = f'{surviving_counts[-1]} survive'
        if has_comparative_advantage_J and has_comparative_advantage_S:
            title += ' (†‡)'
        elif has_comparative_advantage_J:
            title += ' (†)'
        elif has_comparative_advantage_S:
            title += ' (‡)'

        axs[sim].set_title(title)

    plt.suptitle(f'Random J and S matrices, {N_species} species, {N_chemicals} chemicals, {theta=}')
    plt.tight_layout()
    plt.savefig(f'src/randomMatrixMeanField/plots/diffChemTimeseries/comparitiveAdvantage/N_{N_species}-{N_chemicals}_theta_{theta}_{time.time()}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
