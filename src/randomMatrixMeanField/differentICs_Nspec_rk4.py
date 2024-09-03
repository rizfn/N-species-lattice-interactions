import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time


@njit
def set_random_ICs(N, n_records):
    Os = np.zeros((N, n_records))
    Ns = np.zeros((N, n_records))
    randomICs = np.random.rand(2*N)
    randomICs /= np.sum(randomICs)
    for i in range(N):
        Os[i, 0] = randomICs[i]
        Ns[i, 0] = randomICs[i+N]

    return Os, Ns

@njit
def ode_derivatives(O_vals, N_vals, J, theta):
    d_Os = np.zeros_like(O_vals)
    d_Ns = np.zeros_like(N_vals)

    for i in range(len(O_vals)):
        for j in range(len(N_vals)):
            d_Os[i] += J[i,j]*O_vals[i]*N_vals[j]
            d_Ns[i] += (1-J[i,j])*O_vals[i]*N_vals[j] - O_vals[j]*N_vals[i]
        d_Os[i] -= theta * O_vals[i]
        d_Ns[i] += theta * O_vals[i]

    return d_Os, d_Ns

@njit
def ode_integrate_rk4(N, J, theta, stoptime=100_000, nsteps=100_000, dataskip=1):

    dt = stoptime / nsteps
    n_records = nsteps // dataskip + 1

    Os, Ns = set_random_ICs(N, n_records)
    T = np.zeros(n_records)
    T[0] = 0

    current_Os, current_Ns = Os[:, 0], Ns[:, 0]
    record_idx = 1

    for i in range(nsteps):
        k1_Os, k1_Ns = ode_derivatives(current_Os, current_Ns, J, theta)

        O_temp = current_Os + 0.5 * dt * k1_Os
        N_temp = current_Ns + 0.5 * dt * k1_Ns
        k2_Os, k2_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        O_temp = current_Os + 0.5 * dt * k2_Os
        N_temp = current_Ns + 0.5 * dt * k2_Ns
        k3_Os, k3_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        O_temp = current_Os + dt * k3_Os
        N_temp = current_Ns + dt * k3_Ns
        k4_Os, k4_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        current_Os = current_Os + (dt / 6) * (k1_Os + 2 * k2_Os + 2 * k3_Os + k4_Os)
        current_Ns = current_Ns + (dt / 6) * (k1_Ns + 2 * k2_Ns + 2 * k3_Ns + k4_Ns)

        if i % dataskip == 0:
            Os[:, record_idx] = current_Os
            Ns[:, record_idx] = current_Ns
            T[record_idx] = T[record_idx-1] + dataskip*dt
            record_idx += 1

    return T, Os, Ns


def main():

    N = 100
    N_steps = 1_000_000
    N_sims = 4
    dataskip = 100

    J = np.random.rand(N, N)
    # set the diagonal elements of J to 0
    np.fill_diagonal(J, 0)

    theta = 0.1
    survival_threshold = 1e-3


    fig, axs = plt.subplots(N_sims // (N_sims//2), N_sims//2, figsize=(12, 12))
    axs = axs.flatten()

    for sim in range(N_sims):
        T, Os, Ns = ode_integrate_rk4(N, J, theta, N_steps, N_steps, dataskip)
        surviving_counts = np.sum(Os > survival_threshold, axis=0)
        for i in range(N):
            axs[sim].plot(T, Os[i, :], label=f'Species {i+1}')
        axs[sim].set_xlabel('Time')
        axs[sim].set_ylabel('Prevalence of N species')
        axs[sim].grid(True)
        axs[sim].set_title(f'{surviving_counts[-1]} survive')

    plt.suptitle(f'Random J (0 diagonals), {N=}, {theta=}')
    # plt.legend()
    plt.savefig(f'src/randomMatrixMeanField/plots/differentICTimeseries/N_{N}_theta_{theta}_{time.time()}.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()
