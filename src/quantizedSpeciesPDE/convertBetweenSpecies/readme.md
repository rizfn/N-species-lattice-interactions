### Single species able to change tradeoff

From the old model, we had

$$ R_i(N) = \frac{g_i N}{10^{(g_i-1)/0.3} + N}  $$

Where $g_i\in [0,1]$ is a random number, unique to species $i$.


$$ \frac{\partial N}{\partial t} = D_N \nabla^2 N - \sum_i R_i(N) S_i $$

$$\frac{\partial S_i}{\partial t} = D_S \nabla^2 S_i + R_i(N) S_i - S_i \sum_j S_j R_j(N)$$

Now, we allow species to 'transform' or mutate into other species at a constant rate $\mu$. We assume it's symmetric: half goes one way, the other half goes the other way. Then, we get

$$ \frac{\partial N}{\partial t} = D_N \nabla^2 N - \sum_i R_i(N) S_i $$

$$\frac{\partial S_i}{\partial t} = D_S \nabla^2 S_i + \frac\mu2 \left(S_{i-1}+S_{i+1}\right) + R_i(N) S_i - \mu S_i - S_i \sum_j S_j R_j(N)$$

