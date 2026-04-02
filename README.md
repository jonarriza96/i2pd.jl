# i²PD: Implicit Interior Primal-Dual Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2604.00364-b31b1b.svg)](https://arxiv.org/abs/2604.00364)

<!-- For the algorithmic details see [this manuscript](https://www.overleaf.com/read/gmkksfqnqkff#87d1ef). -->

**i2pd.jl** is a Julia implementation of a primal-dual interior point solver for convex optimization problems using a novel implicit representation for complementarity.

**For more information on the algorithmic details see our paper ([preprint](https://arxiv.org/pdf/2604.00364)).**

## Installation

Requires `Julia 1.11.6`

Install dependencies with the following commands

```
cd i2pd.jl/
]
activate .
instantiate
```

## Features

**i2pd.jl** is mainly characterized for having a linear system that is well-conditioned, even near the solution (see this [example](examples/hello_world.jl)). This implies that we can:

- Solve the linear system with **iterative-methods** (without requiring expensive preconditioners) ([example](examples/iterative_methods.jl))
- **Reuse factorizations** across iterations ([example](examples/inexact_newton.jl))
- Reach **high-accuracy in low-precision** arithmetic (float32) ([example](examples/precision.jl))

## Examples

All the examples can be found in [here](examples/).

## Citing

```
@misc{arrizabalaga2026implicitprimaldualinteriorpointmethods,
      title={Implicit Primal-Dual Interior-Point Methods for Quadratic Programming},
      author={Jon Arrizabalaga and Zachary Manchester},
      year={2026},
      eprint={2604.00364},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2604.00364},
}
```

<!-- ## License

TBD -->
