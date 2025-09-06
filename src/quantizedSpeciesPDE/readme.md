### Quantum Biology

The goal is to generate a system with "quantized" species populations: i.e, sharp transitions and plateaus. Furthermore, is stratification and layering possible with only a single resource?

A standard model for bacterial growth rates is Monod's law. The growth rate $R$ varies as a function of nutrient concentration $N$ by

$$ R(N) = \frac{\mu N}{k + N}  $$

We consider a tradeoff: some species could grow very quickly in excess nutrients but slowly in their absence, but other "generalist" species could grow faster in low-nutrient conditions but not as fast as the previous species in nutrient-rich conditions. We use the tradeoff described by [Masaki Tachikawa (2008)](https://doi.org/10.1371/journal.pone.0003925):

$$ R_i(N) = \frac{g_i N}{10^{(g_i-1)/0.3} + N}  $$

Where $g_i\in [0,1]$ is a random number, unique to species $i$.

The equations are now

$$ \frac{\partial N}{\partial t} = D_N \nabla^2 N - \sum_i R_i(N) S_i $$

$$\frac{\partial S_i}{\partial t} = D_S \nabla^2 S_i + R_i(N) S_i - S_i \sum_j S_j R_j(N)$$

Where $D_N$ and $D_S$ are diffusion coefficients for nutrients and species respectively. The rightmost term represents **local dilution**: at each local space, the growth is diluted through normalization.

### Old PDEs that didn't work:


To start, we have one nutrient, $N$, and multiple species $S_i$. Each species has an "efficiency" of growing when it converts a single nutrient, $r_i$

$$ \frac{dN}{dt} = D_N \nabla^2 N - \sum_i r_i S_i N $$

$$ \frac{dS_i}{dt} = D_S \nabla^2 S + r_i S_i N \left(1 - \int_x \frac{S_i}{\sum_j S_j} dx\right) - \gamma S_i $$

Tried, the competition term didn't work.

Instead, we'll add a tradeoff.

The growth of a species in the presence of nutrients with concentration $N$ can be given in terms of a single value $g_i$

$$ R_i(N) = \frac{g_i N}{10^{(g_i-1)/0.3} + N}  $$

$$ \frac{dN}{dt} = D_N \nabla^2 N - \sum_i R_i(N) S_i $$

$$ \frac{dS_i}{dt} = D_S \nabla^2 S + R_i(N) S_i - \gamma S_i $$
