# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# def odes(t, y, J, theta, N):
#     O = y[:N]
#     N_vec = y[N:]
    
#     dO_dt = J @ O * N_vec - theta * O
#     dN_dt = theta * O + (np.eye(N) - J) @ O * N_vec - O * N_vec
    
#     return np.concatenate([dO_dt, dN_dt])

# N = 2
# theta = 0.1
# # J = np.random.rand(N, N)
# J = np.array([[1, 1], [0.1, 0.1]])

# O0 = np.random.rand(N)
# O0 /= O0.sum() * 2
# N0 = np.random.rand(N)
# N0 /= N0.sum() * 2
# y0 = np.concatenate([O0, N0])

# t_span = (0, 1000)
# t_eval = np.linspace(t_span[0], t_span[1], 10000)

# solution = solve_ivp(odes, t_span, y0, args=(J, theta, N), t_eval=t_eval, method='RK45')

# t = solution.t
# O_sol = solution.y[:N, :]
# N_sol = solution.y[N:, :]

# plt.figure(figsize=(10, 6))
# for i in range(N):
#     plt.plot(t, O_sol[i, :], label=f'Species {i+1}')

# plt.xlabel('Time')
# plt.ylabel('Prevalence of N species')
# plt.title(f'{N=}, {theta=}')
# plt.grid(True)
# plt.savefig('src/randomMatrixMeanField/plots/test.png')
# plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def odes(t, y, J, theta, N):
    O = y[:N]
    N_vec = y[N:]
    
    dO_dt = J @ (O * N_vec) - theta * O
    dN_dt = theta * O + (np.ones((N, N)) - J) @ (O * N_vec) - O * N_vec - (O * N_vec).sum(axis=0)
    
    return np.concatenate([dO_dt, dN_dt])

# Parameters
N = 2  # Number of species
theta = 0.1
J = np.array([[1, 1], [0.1, 0.1]])

# Initial conditions
O0 = np.array([0.25, 0.25])
N0 = np.array([0.25, 0.25])
y0 = np.concatenate([O0, N0])

t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(odes, t_span, y0, args=(J, theta, N), t_eval=t_eval, method='RK45')

t = solution.t
O_sol = solution.y[:N, :]
N_sol = solution.y[N:, :]

plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, O_sol[i, :], label=f'Species {i+1}')

plt.xlabel('Time')
plt.ylabel('Prevalence of N species')
plt.title(f'{J=}, {theta=}')
plt.grid(True)
plt.legend()
plt.show()