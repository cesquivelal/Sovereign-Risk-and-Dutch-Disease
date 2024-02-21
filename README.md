# Sovereign-Risk-and-Dutch-Disease

Replication code for "Sovereign Risk and Dutch Disease" by Carlos Esquivel

February, 2024:
https://cesquivelal.github.io/Esquivel_SRDD.pdf

# Data

The folder Data contains the panel data used in Section 4 in the file Panel.csv and STATA code for all regression exercises in the file Regressions.do.

# Quantitative solution of model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The files ResultsForPaper.jl and OptimalSimpleSubsidy.jl use Primitives.jl to solve the model and produce all the results reported in Section 3 of the paper.
