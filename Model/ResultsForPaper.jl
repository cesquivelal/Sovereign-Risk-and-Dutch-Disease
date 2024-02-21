
using Distributed, BenchmarkTools
include("Primitives.jl")

################################################################
#################### Benchmark calibration #####################
################################################################
β=0.95
knk=0.9234166070189569
d1=0.4111111136723473
d0=-knk*d1
yC=1.45
max_b_y=0.35
par, GRIDS=Setup(max_b_y,β,d0,d1,yC)

################################################################
############# Planner and decentralized solutions ##############
################################################################
SOL_DEC=SolveModel_VFI_DEC(true,GRIDS,par)
SOL_PLA=SolveModel_VFI(true,GRIDS,par)

################################################################
############### Targeted and untargeted moments ################
################################################################
MOM_DEC=AverageMomentsManySamples(SOL_DEC,GRIDS,par)
MOM_PLA=AverageMomentsManySamples(SOL_PLA,GRIDS,par)
#Targeted
MOM_DEC.MeanSpreads
MOM_DEC.Debt_GDP
MOM_DEC.ρ_GDP
MOM_DEC.σϵ_GDP
MOM_DEC.yC_GDP

MOM_PLA.MeanSpreads
MOM_PLA.Debt_GDP
MOM_PLA.ρ_GDP
MOM_PLA.σϵ_GDP
MOM_PLA.yC_GDP

#Untargeted
MOM_DEC.StdSpreads
MOM_DEC.σ_GDP
MOM_DEC.σ_con/MOM_DEC.σ_GDP
MOM_DEC.σ_TB_y
MOM_DEC.σ_RER
MOM_DEC.Corr_Spreads_GDP
MOM_DEC.Corr_TB_GDP
MOM_DEC.Corr_RER_GDP

MOM_PLA.StdSpreads
MOM_PLA.σ_GDP
MOM_PLA.σ_con/MOM_PLA.σ_GDP
MOM_PLA.σ_TB_y
MOM_PLA.σ_RER
MOM_PLA.Corr_Spreads_GDP
MOM_PLA.Corr_TB_GDP
MOM_PLA.Corr_RER_GDP

################################################################
##################### Plot price schedule ######################
################################################################
include("FunctionsForResults.jl")
plt=Plot_q1(SOL_PLA,GRIDS,par)
savefig(plt,"PriceSchedule.pdf")
################################################################
############## Compute subsidy and welfare gains ###############
################################################################
N=10000
T=100000
wg=AverageWelfareGains(N,SOL_DEC,SOL_PLA,GRIDS,par)
TS_DEC=SimulatePaths(T,SOL_DEC,GRIDS,par)
TS_PLA=SimulatePaths(T,SOL_PLA,GRIDS,par)
τs_TS=TimeSeriesOfSubsidies(TS_PLA,SOL_PLA,par)
mean(τs_TS)
std(τs_TS)
cor(τs_TS,TS_PLA.nGDP)
cor(τs_TS,TS_PLA.p)
cor(τs_TS,TS_PLA.Spreads)

################################################################
## Planner and decentralized with no commodities, same economy #
################################################################
knk_0=knk
d1_0=d1
d0_0=-knk_0*d1_0
yC_0=0.0
max_b_y=0.35
par_0, GRIDS_0=Setup(max_b_y,β,d0_0,d1_0,yC_0)
SOL_DEC_0=SolveModel_VFI_DEC(true,GRIDS_0,par_0)
SOL_PLA_0=SolveModel_VFI(true,GRIDS_0,par_0)

MOM_DEC_0=AverageMomentsManySamples(SOL_DEC_0,GRIDS_0,par_0)
MOM_PLA_0=AverageMomentsManySamples(SOL_PLA_0,GRIDS_0,par_0)

wg_0=AverageWelfareGains(N,SOL_DEC_0,SOL_PLA_0,GRIDS_0,par_0)
TS_DEC_0=SimulatePaths(T,SOL_DEC_0,GRIDS_0,par_0)
TS_PLA_0=SimulatePaths(T,SOL_PLA_0,GRIDS_0,par_0)
τs_TS_0=TimeSeriesOfSubsidies(TS_PLA_0,SOL_PLA_0,par_0)
mean(τs_TS_0)
std(τs_TS_0)
cor(τs_TS_0,TS_PLA_0.nGDP)
cor(τs_TS_0,TS_PLA_0.p)
cor(τs_TS_0,TS_PLA_0.Spreads)

################################################################
##################### Paths of windfalls #######################
################################################################
N=10000
plt=Plot_TS_Λ_PvsD(N,SOL_DEC,SOL_PLA,GRIDS,par)
savefig(plt,"TimeSeries_CommWind.pdf")

################################################################
################### Welfare gains from rules ###################
################################################################
include("FunctionsForResults.jl")
FIXED="TAX_GAINS_Fixed.csv"
SPREADS="TAX_GAINS_Spreads.csv"
BOTH="TAX_GAINS_SpreadsFix.csv"
plt=PlotSubsidyRules(FIXED,SPREADS,BOTH)
savefig(plt,"GainsSubsidyRules.pdf")

################################################################
########### Planner and decentralized with low eta #############
################################################################
#Set bounds Λlow=0.001 and Λhigh=0.6 for η=0.5 in definition of Parameters
include("Primitives.jl")
max_b_y=1.5
par_η, GRIDS_η=Setup(max_b_y,β,d0,d1,yC,η=0.5)
SOL_DEC_η=SolveModel_VFI_DEC(true,GRIDS_η,par_η)
SOL_PLA_η=SolveModel_VFI(true,GRIDS_η,par_η)

MOM_DEC_η=AverageMomentsManySamples(SOL_DEC_η,GRIDS_η,par_η)
MOM_PLA_η=AverageMomentsManySamples(SOL_PLA_η,GRIDS_η,par_η)

plt=Plot_q1(SOL_PLA_η,GRIDS_η,par_η)

N=10000
T=100000
wg_η=AverageWelfareGains(N,SOL_DEC_η,SOL_PLA_η,GRIDS_η,par_η)
TS_DEC_η=SimulatePaths(T,SOL_DEC_η,GRIDS_η,par_η)
TS_PLA_η=SimulatePaths(T,SOL_PLA_η,GRIDS_η,par_η)
τs_TS_η=TimeSeriesOfSubsidies(TS_PLA_η,SOL_PLA_η,par_η)
mean(τs_TS_η)
std(τs_TS_η)
cor(τs_TS_η,TS_PLA_η.nGDP)
cor(τs_TS_η,TS_PLA_η.p)
cor(τs_TS_η,TS_PLA_η.Spreads)

plot(TS_PLA_η.Λ)

################################################################
######## Planner and decentralized with low eta, yC=0 ##########
################################################################
include("Primitives.jl")
max_b_y=0.35
par_η0, GRIDS_η0=Setup(max_b_y,β,d0,d1,0.0,η=0.5)
SOL_DEC_η0=SolveModel_VFI_DEC(true,GRIDS_η0,par_η0)
SOL_PLA_η0=SolveModel_VFI(true,GRIDS_η0,par_η0)

MOM_DEC_η0=AverageMomentsManySamples(SOL_DEC_η0,GRIDS_η0,par_η0)
MOM_PLA_η0=AverageMomentsManySamples(SOL_PLA_η0,GRIDS_η0,par_η0)

plt=Plot_q1(SOL_PLA_η0,GRIDS_η0,par_η0)

N=10000
T=100000
wg_η0=AverageWelfareGains(N,SOL_DEC_η0,SOL_PLA_η0,GRIDS_η0,par_η0)
TS_DEC_η0=SimulatePaths(T,SOL_DEC_η0,GRIDS_η0,par_η0)
TS_PLA_η0=SimulatePaths(T,SOL_PLA_η0,GRIDS_η0,par_η0)
τs_TS_η0=TimeSeriesOfSubsidies(TS_PLA_η0,SOL_PLA_η0,par_η0)
mean(τs_TS_η0)
std(τs_TS_η0)
cor(τs_TS_η0,TS_PLA_η0.nGDP)
cor(τs_TS_η0,TS_PLA_η0.p)
cor(τs_TS_η0,TS_PLA_η0.Spreads)

plot(TS_PLA_η0.Λ)
