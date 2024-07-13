
using Distributed, BenchmarkTools
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                    markersize=9,size=(650,500))

include("Primitives.jl")
include("FunctionsForResults.jl")

################################################################
#################### Benchmark calibration #####################
################################################################
FOLDER="DutchDisease"
MOD_PLA=UnpackModel_Vector("Planner.csv",FOLDER)
MOD_DEC=UnpackModel_Vector("Decentralized.csv",FOLDER)

################################################################
############### Targeted and untargeted moments ################
################################################################
MOM_DEC=AverageMomentsManySamples(MOD_DEC)
MOM_PLA=AverageMomentsManySamples(MOD_PLA)
#Table 2, Targeted
MOM_DEC.MeanSpreads
MOM_DEC.Debt_GDP
MOM_DEC.ρ_GDP
MOM_DEC.σϵ_GDP
MOM_DEC.yC_GDP
MOM_DEC.σ_inv/MOM_DEC.σ_GDP

#Table 3, decentralized vs planner
MOM_DEC.DefaultPr
MOM_PLA.DefaultPr

MOM_DEC.MeanSpreads
MOM_PLA.MeanSpreads

MOM_DEC.StdSpreads
MOM_PLA.StdSpreads

MOM_DEC.Debt_GDP
MOM_PLA.Debt_GDP

MOM_DEC.σ_GDP
MOM_PLA.σ_GDP

MOM_DEC.σ_con/MOM_DEC.σ_GDP
MOM_PLA.σ_con/MOM_PLA.σ_GDP

MOM_DEC.σ_inv/MOM_DEC.σ_GDP
MOM_PLA.σ_inv/MOM_PLA.σ_GDP

MOM_DEC.σ_TB_y
MOM_PLA.σ_TB_y

MOM_DEC.Corr_Spreads_GDP
MOM_PLA.Corr_Spreads_GDP

MOM_DEC.Corr_TB_GDP
MOM_PLA.Corr_TB_GDP

################################################################
##################### Plot price schedule ######################
################################################################
plt_sprSch=Plot_SpreadSchedule(MOD_DEC)
savefig(plt_sprSch,"Graphs\\SpreadsSchedule.pdf")

################################################################
##################### Paths of windfalls #######################
################################################################
include("FunctionsForResults.jl")
N=10000
plt_k=Plot_TS_K_PvsD(N,MOD_DEC,MOD_PLA)
savefig(plt_k,"Graphs\\TimeSeries_CommWind.pdf")


################################################################
############## Compute subsidy and welfare gains ###############
################################################################
#Table 4
N=10000
wg=AverageWelfareGains_VF(N,MOD_DEC,MOD_PLA)

T=100000
TS_DEC=SimulatePaths(1.0,T,MOD_DEC;FixSeed=true)
τsN_TS, τsT_TS=TimeSeriesOfSubsidies(TS_DEC,MOD_DEC)

round(mean(τsN_TS),digits=2), round(mean(τsT_TS),digits=2)
round(std(τsN_TS),digits=2), round(std(τsT_TS),digits=2)
round(cor(τsN_TS,TS_DEC.GDP),digits=2), round(cor(τsT_TS,TS_DEC.GDP),digits=2)
round(cor(τsN_TS,TS_DEC.p),digits=2), round(cor(τsT_TS,TS_DEC.p),digits=2)
round(cor(τsN_TS,TS_DEC.Spreads),digits=2), round(cor(τsT_TS,TS_DEC.Spreads),digits=2)

FOLDER_RULES="SolveWithSubsidyRule"
BETA_N, BETA_T=SubsidyRuleRegression(TS_DEC,MOD_DEC)
MOD_DEC_RULE=UnpackModel_Vector("Decentralized_rules.csv",FOLDER_RULES)
wg_rule=AverageWelfareGains_VF(N,MOD_DEC,MOD_DEC_RULE)

MOD_PLA_η=UnpackModel_Vector("Planner_eta.csv",FOLDER)
MOD_DEC_η=UnpackModel_Vector("Decentralized_eta.csv",FOLDER)
wg_η=AverageWelfareGains_VF(N,MOD_DEC_η,MOD_PLA_η)
TS_DEC=SimulatePaths(1.0,T,MOD_DEC_η;FixSeed=true)
τsN_TS, τsT_TS=TimeSeriesOfSubsidies(TS_DEC,MOD_DEC_η)
mean(τsN_TS), mean(τsT_TS)
round(std(τsN_TS),digits=3), round(std(τsT_TS),digits=3)
round(cor(τsN_TS,TS_DEC.GDP),digits=2), round(cor(τsT_TS,TS_DEC.GDP),digits=2)
round(cor(τsN_TS,TS_DEC.p),digits=2), round(cor(τsT_TS,TS_DEC.p),digits=2)
round(cor(τsN_TS,TS_DEC.Spreads),digits=2), round(cor(τsT_TS,TS_DEC.Spreads),digits=2)

#Decentralized with no commodities
MOD_PLA_0=UnpackModel_Vector("Planner_0.csv",FOLDER)
MOD_DEC_0=UnpackModel_Vector("Decentralized_0.csv",FOLDER)

MOM_DEC_0=AverageMomentsManySamples(MOD_DEC_0)
MOM_PLA_0=AverageMomentsManySamples(MOD_PLA_0)

MOM_DEC_0.MeanSpreads
