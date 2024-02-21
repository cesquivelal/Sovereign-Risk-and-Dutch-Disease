
using Distributed
addprocs(5)

@everywhere include("Primitives.jl")

β=0.95
knk=0.9234166070189569
d1=0.4111111136723473
d0=-knk*d1
yC=1.45
max_b_y=0.35
par, GRIDS=Setup(max_b_y,β,d0,d1,yC)

################################################################
################# Optimal simple subsidy rule ##################
################################################################
#test different values for τ0 and τ1
#make sure the function SimpleSubsidy in Primitives.jl is
#defined accordingly, the function below only tries values for
#either τ0 or τ1 leaving the other fixed at some value
#(set the fixed value inside the function SimpleSubsidy) 
Nτ=10
τ_GRID=collect(range(0.0,stop=0.3,length=Nτ))
TryDifferentSubsidies(τ_GRID,GRIDS,par)
