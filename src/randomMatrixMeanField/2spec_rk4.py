import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def ode_derivatives(O1, N1, O2, N2, J, theta):

    dO1 = J[0,0]*O1*N1 + J[0,1]*O1*N2 - theta*O1
    dN1 = theta*O1 + (1-J[0,0])*O1*N1 + (1-J[0,1])*O1*N2 - O1*N1 - O2*N1
    dO2 = J[1,0]*O2*N1 + J[1,1]*O2*N2 - theta*O2
    dN2 = theta*O2 + (1-J[1,0])*O2*N1 + (1-J[1,1])*O2*N2 - O1*N2 - O2*N2

    return dO1, dN1, dO2, dN2


@njit
def ode_integrate_rk4(J, theta, stoptime=100_000, nsteps=100_000):

    O1_0 = 0.25
    N1_0 = 0.25
    O2_0 = 0.25
    N2_0 = 0.25

    dt = stoptime / nsteps

    O1 = np.zeros(nsteps+1)
    O2 = np.zeros(nsteps+1)
    N1 = np.zeros(nsteps+1)
    N2 = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    O1[0] = O1_0
    N1[0] = N1_0
    O2[0] = O2_0
    N2[0] = N2_0
    T[0] = 0

    for i in range(nsteps):
        k1_O1, k1_N1, k1_O2, k1_N2 = ode_derivatives(O1[i], N1[i], O2[i], N2[i], J, theta)

        O1_temp = O1[i] + 0.5 * dt * k1_O1
        N1_temp = N1[i] + 0.5 * dt * k1_N1
        O2_temp = O2[i] + 0.5 * dt * k1_O2
        N2_temp = N2[i] + 0.5 * dt * k1_N2
        k2_O1, k2_N1, k2_O2, k2_N2 = ode_derivatives(O1_temp, N1_temp, O2_temp, N2_temp, J, theta)

        O1_temp = O1[i] + 0.5 * dt * k2_O1
        N1_temp = N1[i] + 0.5 * dt * k2_N1
        O2_temp = O2[i] + 0.5 * dt * k2_O2
        N2_temp = N2[i] + 0.5 * dt * k2_N2
        k3_O1, k3_N1, k3_O2, k3_N2 = ode_derivatives(O1_temp, N1_temp, O2_temp, N2_temp, J, theta)

        O1_temp = O1[i] + dt * k3_O1
        N1_temp = N1[i] + dt * k3_N1
        O2_temp = O2[i] + dt * k3_O2
        N2_temp = N2[i] + dt * k3_N2
        k4_O1, k4_N1, k4_O2, k4_N2 = ode_derivatives(O1_temp, N1_temp, O2_temp, N2_temp, J, theta)

        O1[i+1] = O1[i] + (dt / 6) * (k1_O1 + 2 * k2_O1 + 2 * k3_O1 + k4_O1)
        N1[i+1] = N1[i] + (dt / 6) * (k1_N1 + 2 * k2_N1 + 2 * k3_N1 + k4_N1)
        O2[i+1] = O2[i] + (dt / 6) * (k1_O2 + 2 * k2_O2 + 2 * k3_O2 + k4_O2)
        N2[i+1] = N2[i] + (dt / 6) * (k1_N2 + 2 * k2_N2 + 2 * k3_N2 + k4_N2)
        T[i+1] = T[i] + dt

    return T, O1, N1, O2, N2


def main():
    J = np.array([[1, 1], [0.1, 0.1]])
    theta = 0.1

    T, O1, N1, O2, N2 = ode_integrate_rk4(J, theta, 1000, 10000)

    plt.plot(T, O1, label='Species 1')
    plt.plot(T, O2, label='Species 2')
    plt.xlabel('Time')
    plt.ylabel('Prevalence of N species')
    plt.title(f'{J=}, {theta=}')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
