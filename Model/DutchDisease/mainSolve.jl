
using Distributed
addprocs(39)
@everywhere include("Primitives.jl")

@everywhere XX=readdlm("Setup.csv",',')

################################################################
######################## Benchmark case ########################
################################################################
#Decentralized
@everywhere NAME_DEC="Decentralized.csv"
@everywhere NAME_PLA="Planner.csv"
@everywhere VEC=XX[2:end,2]*1.0
@everywhere par, GRIDS=Setup_From_Vector(VEC)
SolveAndSaveModel_BOTH(NAME_DEC,NAME_PLA,GRIDS,par)

################################################################
#################### Benchmark case, yC=0 ######################
################################################################
#Decentralized
@everywhere NAME_DEC="Decentralized_0.csv"
@everywhere NAME_PLA="Planner_0.csv"
@everywhere VEC=XX[2:end,3]*1.0
@everywhere par, GRIDS=Setup_From_Vector(VEC)
SolveAndSaveModel_BOTH(NAME_DEC,NAME_PLA,GRIDS,par)

################################################################
######################### Low eta ##############################
################################################################
#Decentralized
@everywhere NAME_DEC="Decentralized_eta.csv"
@everywhere NAME_PLA="Planner_eta.csv"
@everywhere VEC=XX[2:end,4]*1.0
@everywhere par, GRIDS=Setup_From_Vector(VEC)
SolveAndSaveModel_BOTH(NAME_DEC,NAME_PLA,GRIDS,par)
