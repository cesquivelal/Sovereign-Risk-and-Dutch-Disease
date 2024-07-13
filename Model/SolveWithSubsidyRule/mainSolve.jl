
using Distributed
addprocs(39)
@everywhere include("Primitives.jl")

################################################################
######################## Benchmark case ########################
################################################################
#Decentralized
@everywhere MOD_DEC_0=UnpackModel_Vector("Decentralized.csv"," ")

@everywhere par_τ=Pars(MOD_DEC_0.par,WithInvTax=true)
@everywhere par_τ=Pars(par_τ,τsN0=0.635/100)
@everywhere par_τ=Pars(par_τ,τsN1=-0.053/100)
@everywhere par_τ=Pars(par_τ,τsT0=4.401/100)
@everywhere par_τ=Pars(par_τ,τsT1=0.535/100)

@everywhere NAME_DEC="Decentralized_rules.csv"
SolveAndSaveModel_DEC(NAME_DEC,MOD_DEC_0.SOLUTION,MOD_DEC_0.GRIDS,par_τ)
