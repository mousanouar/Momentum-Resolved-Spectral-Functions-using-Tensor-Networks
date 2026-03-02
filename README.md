# Momentum-Resolved-Spectral-Functions-using-Tensor-Networks

Code and data accompanying the manuscript: **[2512.18397](https://arxiv.org/abs/2512.18397)**. This repository provides a reference implementation of tensor network methods for momentum-resolved spectral functions in tight-binding systems.

This repository aims to:
- Provide core algorithms used in the paper
- Include example workflows showing basic usage
- Offer minimal reproducible examples at small system sizes

---

## Repository structure

### `Main_Modules`

Core implementations of the methods presented in the manuscript:
- `Hamiltonians.jl`: Used to construct Hamiltonians as a matrix product operator (MPO)
- `QuantumKPM.jl`: Kernel Polynomial Algorithms and routines for observables
- `kin_builders.jl`: helper functions to build kinetic operators in MPO format for Hamiltonians

### `Examples notebook`

Examples (same as in the manuscript) illustrating methodology at tractable system sizes:
- `Examples_TNMSF.ipynb`: Step-by-step notebook showing how to build Hamiltonians, compute observables, and visualize results

---

## Installation

The code is written in **Julia**. The required packages are shown below.

### Required packages

Run the following in the Julia REPL before using the repository:
```julia
using Pkg

Install all dependencies via:

```julia
using Pkg

Pkg.add([
    "ITensors",
    "ITensorMPS",
    "Quantics",
    "QuanticsTCI",
    "TensorCrossInterpolation",
    "TCIITensorConversion",
    "ProgressMeter",
    "Plots",
    "LaTeXStrings"
])
