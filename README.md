<p align="center">
  <img src="images/title.png" width="100%" style="margin: 2px;">
</p>

[![Python badge](https://img.shields.io/badge/Python-3.11.11-0066cc?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-31111/)
[![Pylint badge](https://img.shields.io/badge/Linting-pylint-brightgreen?style=for-the-badge)](https://pylint.pycqa.org/en/latest/)
[![Ruff format badge](https://img.shields.io/badge/Formatter-Ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/formatter/)

This code allows to play with the process mentioned in the paper [*Long time asymptotic behavior of a self-similar fragmentation equation*](https://hal.science/hal-04477123v1).

## Installation
A script is available for an easy creation of the conda environment and compilation of auxiliary functions:

```bash
$ source install.bash
```

## How to use ? 

A toy example can be ran with:

```bash
$ python main.py
```
## Fragmention process simulation

<div style="text-align: center;">
    <img src="images/schema_frag.png" alt="Description de l'image" width="600" />
</div>


We consider a process in which a particle of mass $M$ lives a random time (depending on $M$) and split in $\lambda$ particle of mass $M/\lambda$ when dying. The process is then repeated. It is straightforward to simulate such a process. 

## Theoretical behaviour - concentration function
If we assume the random life times to follow an exponential distribution of parameter $\alpha$, the equation of a such process is given by:
```math
    \begin{cases}
    \partial_t c(t,x) = \lambda^{2+\alpha}x^\alpha c(t,\lambda x) - x^\alpha c(t,x) \\
    % c(0,x) = \delta_M(x)
    c(0,x) = u_0(x)
    \end{cases}

```

where $c(t,x)$ denotes the number of particle of mass $x$ at time $t$. We prove that the solution can be obtained with a series expansion of operators applied to the initial condition. More precisely:
```math
    \forall (t,x)\in \mathbb{R}_+^2,\quad c(t,x) = \left[\sum_{k=0}^\infty\frac{t^k}{k!}\mathscr{L}_k\mathcal{F}_\alpha^k\right]u_0(x)
```
with:
```math
        \mathscr{L}_k := \sum_{i=0}^k (-1)^{k-i} \binom{k}{k-i}_{\lambda^{-\alpha}}
        \lambda^{2i+\alpha i (i-1)/2 - \alpha} \mathscr{D}_\lambda^i
```

and

```math
        \mathscr{D}_\lambda f(x) :=f(\lambda x)  \quad\mathrm{and}\quad \mathscr{F}_\alpha f(x) :=x^\alpha f(x).
```

## Theoretical behaviour - moment evolution

We also focus our attention on the the Mellin transform:
```math
    C(t,\sigma) = \int_0^\infty c(t,x)x^{\sigma-1} \mathrm{d}x.
```

We prove in the article that this exact expansion holds:
```math
    C(t,\sigma) = \sum_{k=0}^\infty \frac{(\lambda^{-\alpha},\lambda^{2-\sigma})_k}{k!} (-t)^k
```
where the Pochhamer symbol is defined as:
```math
    \forall n\geq 0,\quad (a,q)_n := \prod_{i=0}^{n-1}(1-aq^i).
```
**This code simulates the process and compute the exact series expansion for the population (case $\sigma=1$).**
Here are some comparisons between experimental results and our series expansion for the population evolution ($\sigma=1$).

<p align="center">
  <img src="images/experiments/POP_10.png" width="40%" style="margin: 2px;">
  <img src="images/experiments/POP_50.png" width="40%" style="margin: 2px;">
</p>
<p align="center">
  <img src="images/experiments/POP_100.png" width="40%" style="margin: 2px;">
  <img src="images/experiments/POP_500.png" width="40%" style="margin: 2px;">
</p>

## Estimation of $\lambda$

We are also interested in computing the inverse problem. From an observation (or several), can we recover the parameter $\lambda$ ? 
Here are the estimations depending on the order of summation.

<p align="center">
  <img src="images/experiments/EST_10.png" width="40%" style="margin: 2px;">
  <img src="images/experiments/EST_50.png" width="40%" style="margin: 2px;">
</p>
<p align="center">
  <img src="images/experiments/EST_100.png" width="40%" style="margin: 2px;">
  <img src="images/experiments/EST_500.png" width="40%" style="margin: 2px;">
</p>


## Interesting ? 

If you have any questions, feel free to contact us. We will be more than happy to answer ! 😀

If you use it, a reference to the paper would be highly appreciated.

```
@article{agazzotti2024long,
  title={Long time asymptotic behavior of a self-similar fragmentation equation},
  author={Agazzotti, Gaetano and Deaconu, Madalina and Lejay, Antoine},
  year={2024}
}
```

## Tested on

[![Ubuntu badge](https://img.shields.io/badge/Ubuntu-24.04-cc3300?style=for-the-badge&logo=ubuntu)](https://www.releases.ubuntu.com/24.04/)
[![Conda badge](https://img.shields.io/badge/conda-24.9.2-339933?style=for-the-badge&logo=anaconda)](https://docs.conda.io/projects/conda/en/24.9.x/)
[![Intel badge](https://img.shields.io/badge/CPU-%20i5_10210U%201.60GHZ-blue?style=for-the-badge&logo=intel)](https://www.intel.com/content/www/us/en/products/sku/195436/intel-core-i510210u-processor-6m-cache-up-to-4-20-ghz/specifications.html)

