We have $N$ species of organisms, $O_i$, each with its own nutrient $N_i$.

The organisms die at a constant rate $\theta$. The organisms consume different nutrients, and each one reproduces depending on the nutrient it consumes. The rate at which organism $O_i$ reproduces when consuming nutrient $N_j$ is given by $J_{ij}$, where $J$ is a random $N\times N$ matrix. 

For a simple case for 2 species, we can write equations as

$$
\begin{align}
\frac{dO_1}{dt} &= J_{11}O_1N_1 + J_{12}O_1N_2  -\theta O_1 \\
\frac{dO_2}{dt} &= J_{21}O_2N_1 + J_{22}O_2N_2  -\theta O_2 \\
\frac{dN_1}{dt} &= \theta O_1 + (1-J_{11})O_1N_1 + (1-J_{12})O_1N_2 - O_1N_1 - O_2N_1 \\
\frac{dN_2}{dt} &= \theta O_2 + (1-J_{21})O_2N_1 + (1-J_{22})O_2N_2 - O_1N_2 - O_2N_2
\end{align}
$$