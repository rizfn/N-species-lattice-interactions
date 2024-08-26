import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

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
def ode_integrate_rk4(N, J, theta, stoptime=100_000, nsteps=100_000):

    dt = stoptime / nsteps
    Os = np.zeros((N, nsteps+1))
    Ns = np.zeros((N, nsteps+1))

    Os[:, 0] = 1/(2*N)
    Ns[:, 0] = 1/(2*N)

    T = np.zeros(nsteps+1)
    T[0] = 0

    for i in range(nsteps):
        k1_Os, k1_Ns = ode_derivatives(Os[:, i], Ns[:, i], J, theta)

        O_temp = Os[:, i] + 0.5 * dt * k1_Os
        N_temp = Ns[:, i] + 0.5 * dt * k1_Ns
        k2_Os, k2_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        O_temp = Os[:, i] + 0.5 * dt * k2_Os
        N_temp = Ns[:, i] + 0.5 * dt * k2_Ns
        k3_Os, k3_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        O_temp = Os[:, i] + dt * k3_Os
        N_temp = Ns[:, i] + dt * k3_Ns
        k4_Os, k4_Ns = ode_derivatives(O_temp, N_temp, J, theta)

        Os[:, i+1] = Os[:, i] + (dt / 6) * (k1_Os + 2 * k2_Os + 2 * k3_Os + k4_Os)
        Ns[:, i+1] = Ns[:, i] + (dt / 6) * (k1_Ns + 2 * k2_Ns + 2 * k3_Ns + k4_Ns)
        T[i+1] = T[i] + dt

    return T, Os, Ns


def main():

    N = 100
    J = np.random.rand(N, N)

    theta = 0.1

    T, Os, Ns = ode_integrate_rk4(N, J, theta, 100_000, 100_000)

    for i in range(N):
        plt.plot(T, Os[i, :], label=f'Species {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Prevalence of N species')
    plt.title(f'Random J, {N=}, {theta=}')
    plt.grid(True)
    # plt.legend()
    plt.savefig(f'src/randomMatrixMeanField/plots/timeseries/N_{N}_theta_{theta}_{time.time()}.png', dpi=300)
    plt.show()


def no_diag():

    N = 100
    N_steps = 100_000
    J = np.random.rand(N, N)
    # set the diagonal elements of J to 0
    np.fill_diagonal(J, 0)

    theta = 0.1
    survival_threshold = 1e-3

    T, Os, Ns = ode_integrate_rk4(N, J, theta, N_steps, N_steps)

    surviving_counts = np.sum(Os > survival_threshold, axis=0)
    change_indices = np.where(np.diff(surviving_counts) != 0)[0] + 1

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(T, Os[i, :], label=f'Species {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Prevalence of N species')
    plt.title(f'Random J (0 diagonals), {N=}, {theta=}, {surviving_counts[-1]} survive')
    plt.grid(True)
    # Plot vertical dashed lines and add text annotations
    for idx in range(len(change_indices)):
        if T[change_indices[idx]] > T[N_steps // 10]:
            plt.axvline(x=T[change_indices[idx]], color='k', linestyle='--', alpha=0.5)
            if idx < len(change_indices) - 1:
                mid_point = (T[change_indices[idx]] + T[change_indices[idx + 1]]) / 2
                plt.text(mid_point, np.max(Os), f'{surviving_counts[change_indices[idx]]}', 
                         verticalalignment='bottom', horizontalalignment='center', alpha=0.7)
    if change_indices.size > 0 and T[change_indices[-1]] > T[N_steps // 4]:
        plt.text((T[change_indices[-1]] + N_steps) / 2, np.max(Os), f'{surviving_counts[-1]}', 
                 verticalalignment='bottom', horizontalalignment='center', alpha=0.7)

    # plt.legend()
    plt.savefig(f'src/randomMatrixMeanField/plots/NoDiagTimeseries/N_{N}_theta_{theta}_{time.time()}.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    # main()
    no_diag()
