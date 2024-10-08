# Long time asymptotic behavior of a self-similar fragmentation equation

This code allows to play with the process mentioned in the paper [*Long time asymptotic behavior of a self-similar fragmentation equation*](https://hal.science/hal-04477123v1) written by Gaetano Agazzotti, Madalina Deaconu and Antoine Lejay.

## Fragmention process simulation

<div style="text-align: center;">
    <img src="images/schema_frag.png" alt="Description de l'image" width="600" />
</div>


We consider a process in which a particle of masse $M$ lives a random time (depending on $M$) and split in $\lambda$ particle of mass $M/\lambda$ when dying. The process is then repeated. If we assume the random life times to follow a exponential distribution of parameter $\alpha$, the equation of a such process is given by:
```math
    \begin{cases}
    \partial_t c(t,x) = \lambda^{2+\alpha}x^\alpha c(t,\lambda x) - x^\alpha c(t,x) \\
    c(0,x) = \delta_M(x)
    \end{cases}

```

where $c(t,x)$ denotes the number of particle of mass $x$ at time $t$, we are interested in the Mellin transform:
```math
    C(t,\sigma) = \int_0^\infty c(t,x)x^{\sigma-1} \mathrm{d}x.
```

This codes simulates the process and compute the exact serie expansion given in the paper. 

## Estimation of $\lambda$

We are also interested in computing the inverse problem. From an observation (or several), can we recover the parameter $\lambda$ ? 

The output presents the estimation of $\lambda$ depending on the number of estimation and on the order of the expansion. 







