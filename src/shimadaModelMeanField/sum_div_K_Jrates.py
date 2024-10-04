import numpy as np
from numba import njit, typed, types
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


@njit
def set_random_ICs(N_SPECIES, N_CHEMICALS, n_records):
    Os = np.zeros((N_SPECIES, n_records))
    randomICs = np.random.rand(N_SPECIES + 1)  # (+1 for empty sites)
    Os[:, 0] = randomICs[1:] / np.sum(randomICs)
    Ns = np.zeros((N_CHEMICALS, n_records))  
    Ns[:, 0] = 1.0   # All nutrients are initially present

    return Os, Ns


@njit
def ode_derivatives(O_vals, N_vals, J, S, theta, k):
    d_Os = np.zeros_like(O_vals)
    d_Ns = np.zeros_like(N_vals)

    empty_frac = 1 - np.sum(O_vals)

    for i in range(len(O_vals)):
        # Compute the sum of nutrient concentrations consumed by species O_i
        nutrient_sum_O = 0
        for j, rate in J[i]:
            nutrient_sum_O += N_vals[j] * rate

        nutrient_sum_O /= k  # special to the sum_div_K model: Having all of everything is better than all of one particular thing.

        # Compute dO_i/dt
        d_Os[i] = O_vals[i] * empty_frac * nutrient_sum_O - theta * O_vals[i]
        for j, rate in J[i]:
            d_Ns[j] -= O_vals[i] * empty_frac * nutrient_sum_O * rate

        for j in S[i]:
            d_Ns[j] += O_vals[i] * (1 - N_vals[j])

    return d_Os, d_Ns


@njit
def ode_integrate_rk4(N_species, N_chemicals, J, S, theta, k, stoptime=100_000, nsteps=100_000, dataskip=1):

    dt = stoptime / nsteps
    n_records = nsteps // dataskip + 1

    Os, Ns = set_random_ICs(N_species, N_chemicals, n_records)
    T = np.zeros(n_records)
    T[0] = 0

    current_Os, current_Ns = Os[:, 0], Ns[:, 0]
    record_idx = 1

    for i in range(nsteps):
        k1_Os, k1_Ns = ode_derivatives(current_Os, current_Ns, J, S, theta, k)

        O_temp = current_Os + 0.5 * dt * k1_Os
        N_temp = current_Ns + 0.5 * dt * k1_Ns
        k2_Os, k2_Ns = ode_derivatives(O_temp, N_temp, J, S, theta, k)

        O_temp = current_Os + 0.5 * dt * k2_Os
        N_temp = current_Ns + 0.5 * dt * k2_Ns
        k3_Os, k3_Ns = ode_derivatives(O_temp, N_temp, J, S, theta, k)

        O_temp = current_Os + dt * k3_Os
        N_temp = current_Ns + dt * k3_Ns
        k4_Os, k4_Ns = ode_derivatives(O_temp, N_temp, J, S, theta, k)

        current_Os = current_Os + (dt / 6) * (k1_Os + 2 * k2_Os + 2 * k3_Os + k4_Os)
        current_Ns = current_Ns + (dt / 6) * (k1_Ns + 2 * k2_Ns + 2 * k3_Ns + k4_Ns)

        if i % dataskip == 0:
            Os[:, record_idx] = current_Os
            Ns[:, record_idx] = current_Ns
            T[record_idx] = T[record_idx-1] + dataskip*dt
            record_idx += 1

    return T, Os, Ns


def init_J_S_maps(N_species, N_chemicals, k):
    J = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.ListType(types.Tuple((types.int64, types.float64)))
    )
    S = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )
    for i in range(N_species):
        possible_indices = np.arange(N_chemicals)
        np.random.shuffle(possible_indices)
        J[i] = typed.List.empty_list(types.Tuple((types.int64, types.float64)))
        for idx in possible_indices[:k]:
            rate = np.random.rand()
            J[i].append((idx, rate))
        S[i] = np.array(possible_indices[k:2*k], dtype=np.int64)
    return J, S

def main():

    N_species = 200
    N_chemicals = 50
    k = 2
    N_steps = 1_000_000
    N_sims = 4
    N_datapoints = 1000
    dataskip = N_steps // N_datapoints

    theta = 0.01
    survival_threshold = 1e-6

    J, S = init_J_S_maps(N_species, N_chemicals, k)

    fig, axs = plt.subplots(N_sims // (N_sims//2), N_sims//2, figsize=(16, 12))
    axs = axs.flatten()

    colors = plt.cm.rainbow(np.linspace(0, 1, N_species))

    for sim in tqdm(range(N_sims)):
        T, Os, Ns = ode_integrate_rk4(N_species, N_chemicals, J, S, theta, k, N_steps, N_steps, dataskip)
        surviving_counts = np.sum(Os > survival_threshold, axis=0)
        for i in range(N_species):
            axs[sim].plot(T, Os[i, :], label=f'Species {i+1}', c=colors[i % len(colors)])  # , marker='.', linestyle=''
        axs[sim].set_xlabel('Time')
        axs[sim].set_ylabel('Prevalence of N species')
        # axs[sim].set_xscale('log')
        # axs[sim].set_yscale('log')
        axs[sim].grid(True)
        axs[sim].set_title(f'{surviving_counts[-1]} survive')

    plt.suptitle(f'{N_species=}, {N_chemicals}, {k=} {theta=}')
    # plt.legend()
    plt.savefig(f'src/shimadaModelMeanField/plots/sum_div_K_Jrates/N_{N_species}-{N_chemicals}_k_{k}_theta_{theta}_{time.time()}.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()