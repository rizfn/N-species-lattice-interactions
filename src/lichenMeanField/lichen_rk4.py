import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

@njit
def ode_derivatives(O_vals, J):
    d_Os = np.zeros_like(O_vals)

    for i in range(len(O_vals)):
        for j in range(len(O_vals)):
            if i != j:
                d_Os[i] += J[i,j] * O_vals[i] * O_vals[j]
                d_Os[i] -= J[j,i] * O_vals[j] * O_vals[i]
            

    return d_Os

@njit
def ode_integrate_rk4(N, J, stoptime=100_000, nsteps=100_000):

    dt = stoptime / nsteps
    Os = np.zeros((N, nsteps+1))

    Os[:, 0] = 1/N

    T = np.zeros(nsteps+1)
    T[0] = 0

    for i in range(nsteps):
        k1_Os = ode_derivatives(Os[:, i], J)

        O_temp = Os[:, i] + 0.5 * dt * k1_Os
        k2_Os = ode_derivatives(O_temp, J)

        O_temp = Os[:, i] + 0.5 * dt * k2_Os
        k3_Os = ode_derivatives(O_temp, J)

        O_temp = Os[:, i] + dt * k3_Os
        k4_Os = ode_derivatives(O_temp, J)

        Os[:, i+1] = Os[:, i] + (dt / 6) * (k1_Os + 2 * k2_Os + 2 * k3_Os + k4_Os)
        T[i+1] = T[i] + dt

    return T, Os


# def main():

#     N = 100
#     J = np.random.rand(N, N)
#     np.fill_diagonal(J, 0)

#     T, Os = ode_integrate_rk4(N, J, 100_000, 100_000)
#     N_surviving = np.sum(Os[:, -1] > 1e-12)

#     for i in range(N):
#         plt.plot(T, Os[i, :], label=f'Species {i+1}')
#     plt.xlabel('Time')
#     plt.ylabel('Prevalence of N species')
#     plt.title(f'Random J, {N=}, {N_surviving} survive')
#     plt.grid(True)
#     # plt.legend()
#     plt.show()


def main():
    N = 20
    N_steps = 100_000
    J = np.random.rand(N, N)
    np.fill_diagonal(J, 0)

    T, Os = ode_integrate_rk4(N, J, N_steps, N_steps)
    survival_threshold = 1e-12
    surviving_counts = np.sum(Os > survival_threshold, axis=0)
    
    # Find the points where the number of surviving species changes
    change_indices = np.where(np.diff(surviving_counts) != 0)[0] + 1
    
    for i in range(N):
        plt.plot(T, Os[i, :], label=f'Species {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Prevalence of N species')
    plt.title(f'Random J, {N=}, {surviving_counts[-1]} survive')
    plt.grid(True)
    
    # Plot vertical dashed lines and add text annotations
    for idx in range(len(change_indices)):
        if T[change_indices[idx]] > T[N_steps // 4]:
            plt.axvline(x=T[change_indices[idx]], color='k', linestyle='--', alpha=0.5)
            if idx < len(change_indices) - 1:
                mid_point = (T[change_indices[idx]] + T[change_indices[idx + 1]]) / 2
                plt.text(mid_point, np.max(Os), f'{surviving_counts[change_indices[idx]]}', 
                         verticalalignment='bottom', horizontalalignment='center', alpha=0.7)
    if change_indices.size > 0 and T[change_indices[-1]] > T[N_steps // 4]:
        plt.text((T[change_indices[-1]] + T[-1]) / 2, np.max(Os), f'{surviving_counts[-1]}', 
                 verticalalignment='bottom', horizontalalignment='center', alpha=0.7)
    
    plt.show()

if __name__ == '__main__':
    main()
