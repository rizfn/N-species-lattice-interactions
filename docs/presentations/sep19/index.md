---
marp: true
theme: uncover
math: mathjax
paginate: true
_paginate: skip
backgroundSize: contain
style: |
        .columns {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.6rem;
        }
        h1, h2, h3, h4, h5, h6, strong {
          color: #400000;
        }
        .caption {
          font-size: 0.5em;
          line-height: 10%;
          letter-spacing: 0.01em;
          margin-top: -100px;
          margin-bottom: -100px;
        }

---

![bg right](fig/Mangrove.jpg)

# Multi-species ecosystems

$\\$

Riz Fernando Noronha

---

### Predator-Prey Models

$\\$

$$
\begin{align}
\frac{\mathrm{d}x}{\mathrm{d}t} &= \alpha x - \beta xy \\
\frac{\mathrm{d}y}{\mathrm{d}t} &= \delta xy - \gamma y
\end{align}
$$

---

### How do we model many species??

$\\$

Every species (possibly) interacts with every other species

$N$ species implies $\sim N^2$ interactions!!


---

### Random Matrix Models

Each species can consume every other species.

The rate at which species $i$ consumes species $j$ is given by $J_{ij}$

$J$ is a matrix with random (uniform) entries

---

![bg](fig/mathiesen_mitarai.png)
![bg vertical right:60%](fig/sneppen_trusina.png)

### On a lattice?

$\\$

Done in a simple, elegant way!

---

![bg right](fig/lichen_model.png)

#### Lichen Model

2011

Introduce **new species** at a fixed rate

---

### On the Micro-scale?

<div class="columns">

<video src="fig/t6ss_vibriocyclitrophicus_ordalii.mp4" autoplay muted loop></video>

- Type 6 Secretion System (**T6SS**)

- <span style="color:#EA4DC0">Vibrio cyclitrophicus</span>

- <span style="color:#00A9DC">Vibrio ordalii</span>

- Damages cell membrane (becomes circular)

</div>


---

### Real Microbial Communities

$\\$

T6SS is relatively **uncommon**

Microbes interact through chemicals

**Secretion** and **uptake**


---

### Consumption $\rightarrow$ Catalysis

$\\$

- Every species has it's own chemical

- Species $i$ constantly produces chemical $i$ 

- Species $i$ reproduces with chemical $j$ at rate $J_{ij}$

---


### Mean-Field Equations

$$
\begin{align}
\frac{\mathrm{d}\color{darkgreen}{O_1}}{\mathrm{d}t} &= J_{\color{darkgreen}{1}\color{green}{1}}\color{darkgreen}{O_1}\color{green}{N_1} \color{black}{+} J_{\color{darkgreen}{1}\color{blue}{2}}\color{darkgreen}{O_1}\color{blue}{N_2}  \color{black}{- \theta} \color{darkgreen}{O_1} \\
\frac{\mathrm{d}\color{darkblue}{O_2}}{\mathrm{d}t} &= J_{\color{darkblue}{2}\color{green}{1}}\color{darkblue}{O_2}\color{green}{N_1} \color{black}{+} J_{\color{darkblue}{2}\color{blue}{2}}\color{darkblue}{O_2}\color{blue}{N_2}  \color{black}{- \theta} \color{darkblue}{O_2} \\
\frac{\mathrm{d}\color{green}{N_1}}{\mathrm{d}t} &= \color{black}{\theta} \color{darkgreen}{O_1} \color{black}{+ (1-} J_{\color{darkgreen}{1}\color{green}{1}} \color{black}{)} \color{darkgreen}{O_1}\color{green}{N_1} \color{black}{+ (1-} J_{\color{darkgreen}{1}\color{blue}{2}} \color{black}{)} \color{darkgreen}{O_1}\color{blue}{N_2} \color{black}{-} \color{darkgreen}{O_1}\color{green}{N_1} \color{black}{-} \color{darkblue}{O_2}\color{green}{N_1} \\
\frac{\mathrm{d}\color{blue}{N_2}}{\mathrm{d}t} &= \color{black}{\theta} \color{darkblue}{O_2} \color{black}{+ (1-} J_{\color{darkblue}{2}\color{green}{1}} \color{black}{)} \color{darkblue}{O_2}\color{green}{N_1} \color{black}{+ (1-} J_{\color{darkblue}{2}\color{blue}{2}} \color{black}{)} \color{darkblue}{O_2}\color{blue}{N_2} \color{black}{-} \color{darkgreen}{O_1}\color{blue}{N_2} \color{black}{-} \color{darkblue}{O_2}\color{blue}{N_2}
\end{align}
$$


---

![bg width:75%](fig/random_matrix.png)

---

![width:600px](fig/random_matrix_why_diagonals_matter.png)

Switching is **interesting**, but self-interactions?

Turn off the diagonal!

$$ J_{ii} = 0 \quad \forall \;i $$

---

![bg width:90%](fig/random_matrix_0diag.png)


---

<div class="columns">

<video src="fig/lattice_anim_N_50_theta_0.1_1.mp4" autoplay muted loop style="max-width:100%;"></video>

<div>

### Lattice model
<br>

- Spatial patterning present

- Invasion between ecosystems?


</div>


</div>

---

**More species survive!**

![width:900px](fig/random_matrix_lattice_timeseries.png)


---

### 'Problem' 1: Long-Term Stability

![w:1000px](fig/random_matrix_longtime.png)

---

![bg left:55% fit](fig/randommatrix_differentICs.png)

Same $J$ matrix, different ICs

$\\$

**Different Final States!**

$\\$
1. Life is transient
2. Add noise

---

### Problem 2: King Midas

<div class="columns">

<video src="fig/midas.mp4" autoplay muted loop style="max-width:100%;"></video>

![width:500px](fig/midas_schematic.png)

</div>


---

## More species than chemicals?

$\\$

![bg left:45%](fig/takashi_shimada.jpg)

Inspired by the work of **Takashi Shimada** and Kuni.

_Spherical Cow Prize Winner_

---

![bg left:40% fit](fig/shimada_schematic.png)

### Two matrices!

Consume chemicals from **$J_{ij}$** to reproduce

Secrete chemicals from **$S_{ij}$**

$S_{ij}$ and $J_{ij}$ are **sparse**!

---

![bg left:40% fit](fig/and_model_schematic.png)

### 'AND' model

- Each species needs **2** chemicals to reproduce!

- Each species also secretes 2

- You **cannot** secrete what you need!

---

200 species, 50 chemicals

![width:1100px](fig/shimadaAnd_200-50.png)


---

200 species, 70 chemicals

![width:1100px](fig/shimadaAnd_200-70.png)

---

200 species, 100 chemicals

![width:1100px](fig/shimadaAnd_200-100.png)

---

For 200 species:

- 50 chemicals: **Fixed point**
  - Many survive! (~100+)
- 70 chemicals: **Switching**
  - Fewer survive (~20)
- 90 chemicals: **Chaos**
  - Almost none survive (~10)

**Lack of resource variety drives diversity??!!**

---

![bg left:37%](fig/mikhail_tikhonov.jpg)

### "Emergent Simplicity"

$\\$

Inspired by **Mikhail Tikhonov**, Washington University St. Louis.

---

Coarse-graining makes things look **simpler!**

$\\$

![width:1100px](fig/tikhonov_emergent_simplicity.png)

---

### Coarse-graining Mean-Field?

![width:800px](fig/coarse_graining_switching.png)

Reindex matrix by abundances, and look for patterns!

---

### Coarse-graining Lattice?

<div class="columns">

![width:500px](fig/coarse_graining_lattice.png)


- Divide into sectors
<br>
- Each sector is it's own ecosystem
<br>
- Low dimensional manifold?

</div>

---

## Conclusion

- Multi-species modelling is arbitrary!

- More work needed!

  - Check coarse-graining

  - Evolve $J_{ij}$?

  - Introduce new species?
