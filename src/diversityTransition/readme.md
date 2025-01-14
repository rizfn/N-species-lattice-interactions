## Random Connections:

Two vectors, $S$, an array with abundances of 'resources' (externally supplied at a fixed rate $\gamma$).

Reaction rate for the chemical network is $a$ (fixed, for now). The network telling you who interacts with whom is given by a $J$ matrix.

Every time step, grow $S$ and $X$.

$$
\begin{align}
\frac{dS_i}{dt} &= \gamma (1- S_i) - \sum_{j,k} \alpha X_j X_k \delta_{i, \text{connectivity}[j,k]} \\
\frac{dX_i}{dt} &= \sum_{j} \alpha X_j X_i S_{\text{connectivity}[j,i]}
\end{align}
$$

Or, the first equation can also be

$$ 
\begin{equation}
\frac{dS_i}{dt} = \gamma - S_i - \sum_{j,k} \alpha X_j X_k \delta_{i, \text{connectivity}[j,k]}     
\end{equation}
$$