## Shimada Model

An equivalent model to the one developed by Takashi Shimada and Kunihiko Kaneko.

- Each species secretes $K$ chemicals, given by the secretion matrix $S$
- Species can eat $K$ chemicals, given by the consumption matrix $J$
- If a species can eat chemicals A and B, it means it needs *either* A or B to grow. In other words, eating each can help the species grow.

We can write down mean-field equations such as:

$$
\begin{align}
\frac{dO_i}{dt} &= O_i \left(1- \sum_j O_j\right) \sum_j \left(J_{ij} N_j \right) - \theta O_i \\
\frac{dN_i}{dt} &= (1-N_i)\left(\sum_j O_j S_{ji} \right) - \sum_{k} \left[ O_k \left(1-\sum_mO_m\right) J_{ki} N_i \right]
\end{align}
$$


### `sum_div_K` case:

In the `sum_div_K` case, we say that each chemical is bounded by 0 to 1, and a species's reproduction rate depends on the sum of the chemicals it consumes divided by $K$. In other words, if both species are at the maximum concentration, it will be consumed and reproduced at the maximum rate. If only half are at the maximum and the other half are 0, it'll grow at *half* the rate as the former case.

$$
\begin{align}
\frac{dO_i}{dt} &= O_i \left(1- \sum_j O_j\right) \sum_j \left(\frac{J_{ij} N_j}{K} \right) - \theta O_i \\
\frac{dN_i}{dt} &= (1-N_i)\left(\sum_j O_j S_{ji} \right) - \sum_{k} \left[ O_k \left(1-\sum_mO_m\right) \frac{J_{ki} N_i}{K} \right]
\end{align}
$$


K = sum(J_ij)N_j