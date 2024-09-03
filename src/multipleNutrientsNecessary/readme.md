## AND logic

Set a parameter $k$ which controls the number of nutrients a certain organism needs to replicate.

Now, $J$ is a **boolean** matrix, and $J_{ij}$ has information on what species $i$ needs to grow. 

$$ \sum_jJ_{ij} = k \quad ; \forall \;i $$

(Each species needs $k$ nutrients to grow).

- If a species finds **all** of the $k$ nutrients it needs, it reproduces
- It leaves behind it's nutrient, unless it's already present
- It dies with a constant rate $\theta$.

For a simple 2-species system, we have


$$
\begin{align}
\frac{dO_i}{dt} &= O_i (1-O_i) \prod_j \left({N_j}^{J_{ij}} \right) - \theta O_i \\
\frac{dN_i}{dt} &= O_i(1-N_i) - \sum_{k} \left( O_k (1-O_k) J_{ki} \prod_{j} \left( N_j^{J_{kj}} \right) \right)
\end{align}
$$
