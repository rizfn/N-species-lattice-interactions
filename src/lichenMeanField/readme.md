This is inspired by the work of [Mitarai, Mathiesen and Sneppen (2012)](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.86.011929).

First, in a simplified model of this, we consider the case where new species are **not** added.

There are **no** nutrients here, and it's just a random matrix predator-prey interaction. Once again, you have a $J$ matrix, and the rate at which species $i$ eats species $j$ is given by $J_{ij}$.

The equations for 2 species will be

$$
\begin{align}
\frac{dO_1}{dt} &= J_{12}O_1O_2 - J_{21}O_2O_1  \\
\frac{dO_2}{dt} &= J_{21}O_2O_1 - J_{12}O_1O_2
\end{align}
$$

and for 3 species, 

$$
\begin{align}
\frac{dO_1}{dt} &= J_{12}O_1O_2 + J_{13}O_1O_3 - J_{21}O_2O_1 - J_{31}O_3O_1 \\
\frac{dO_2}{dt} &= J_{21}O_2O_1 + J_{23}O_2O_3 - J_{12}O_1O_2 - J_{32}O_3O_2 \\
\frac{dO_3}{dt} &= J_{31}O_3O_1 + J_{32}O_3O_2 - J_{13}O_1O_3 - J_{23}O_2O_3
\end{align}
$$