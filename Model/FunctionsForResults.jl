
# using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
#                     background_color_legend=nothing,foreground_color_legend=nothing,
#                     legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
#                     markersize=9,size=(650,500))

###############################################################################
#Functions to compute Euler Equations errors in state space
###############################################################################
function EulerErrors(Default::Bool,I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Compute sum of squared Euler Equation residuals
    @unpack cmin, WithInvTax = par
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    if Default
        (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
        @unpack kNprime_D, kTprime_D = SOLUTION
        @unpack itp_ERN_Def, itp_ERT_Def = HH_OBJ
        #Compute present consumption
        z=GR_z[z_ind]; p=GR_p[p_ind]
        zD=zDefault(z,par)
        kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]

        kNprime=kNprime_D[I]
        kTprime=kTprime_D[I]
        T=0.0
        y=FinalOutput(zD,p,kN,kT,T,par)
        cons=ConsNet(y,kN,kT,kNprime,kTprime,par)

        #compute expectation over z and yC
        EvN=itp_ERN_Def(kTprime,kNprime,z,p_ind)
        EvT=itp_ERT_Def(kTprime,kNprime,z,p_ind)
        #compute extra term and return FOC
        ψ1N=dΨ_dkprime(kNprime,kN,par)
        ψ1T=dΨ_dkprime(kTprime,kT,par)

        #Compute Euler equation residuals
        Euler_N=EvN-U_c(cons,par)*(1+ψ1N)
        Euler_T=EvT-U_c(cons,par)*(1+ψ1T)

        return Euler_N, Euler_T
    else
        @unpack GR_b = GRIDS
        (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
        @unpack itp_q1 = SOLUTION
        @unpack itp_ERN_Rep, itp_ERT_Rep = HH_OBJ
        #Compute present consumption
        z=GR_z[z_ind]; p=GR_p[p_ind]
        kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]

        kNprime=SOLUTION.kNprime[I]
        kTprime=SOLUTION.kTprime[I]
        bprime=SOLUTION.bprime[I]


        #Compute subsidy
        if WithInvTax
            τsN, τsT=SimpleSubsidy(qq,par)
        else
            τsN=0.0
            τsT=0.0
        end
        T=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
        y=FinalOutput(z,p,kN,kT,T,par)
        cons=ConsNet(y,kN,kT,kNprime,kTprime,par)

        #compute expectation over z and yC
        EvN=itp_ERN_Rep(bprime,kTprime,kNprime,z,p_ind)
        EvT=itp_ERT_Rep(bprime,kTprime,kNprime,z,p_ind)
        #compute extra term and return FOC
        ψ1N=dΨ_dkprime(kNprime,kN,par)
        ψ1T=dΨ_dkprime(kTprime,kT,par)

        #Compute Euler equation residuals
        Euler_N=EvN-U_c(cons,par)*(1+ψ1N-τsN)
        Euler_T=EvT-U_c(cons,par)*(1+ψ1T-τsT)

        return Euler_N, Euler_T
    end
end

function MatricesOfEulerErrors(MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack Nb, NkT, NkN, Nz, Np = par
    HH_OBJ=InitiateHH_Obj(GRIDS,par)
    IncludeRepayment=true
    SHARED_AUX=InitiateSharedAux(GRIDS,par)
    UpdateHH_Obj!(IncludeRepayment,SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par)
    Euler_N=Array{Float64,5}(undef,Nb,NkT,NkN,Nz,Np)
    Euler_T=Array{Float64,5}(undef,Nb,NkT,NkN,Nz,Np)

    Euler_Ndef=Array{Float64,4}(undef,NkT,NkN,Nz,Np)
    Euler_Tdef=Array{Float64,4}(undef,NkT,NkN,Nz,Np)

    Default=true
    for I in CartesianIndices(SOLUTION.VD)
        Euler_Ndef[I], Euler_Tdef[I]=EulerErrors(true,I,HH_OBJ,SOLUTION,GRIDS,par)
    end

    for I in CartesianIndices(SOLUTION.V)
        (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
        if SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]>=SOLUTION.VP[I]
            Id=CartesianIndex(kT_ind,kN_ind,z_ind,p_ind)
            Euler_N[I], Euler_T[I]=EulerErrors(true,Id,HH_OBJ,SOLUTION,GRIDS,par)
        else
            Euler_N[I], Euler_T[I]=EulerErrors(false,I,HH_OBJ,SOLUTION,GRIDS,par)
        end
    end

    return Euler_N, Euler_T, Euler_Ndef, Euler_Tdef
end

function DistributionOfEulerErrors(MODEL::Model)
    Euler_N, Euler_T, Euler_Ndef, Euler_Tdef=MatricesOfEulerErrors(MODEL)
    logErr_N=log10.(abs.(Euler_N))
    logErr_T=log10.(abs.(Euler_T))

    logErr_Ndef=log10.(abs.(Euler_Ndef))
    logErr_Tdef=log10.(abs.(Euler_Tdef))
    all_err_K=vcat(reshape(logErr_N,(:)),reshape(logErr_T,(:)))
    all_err_rep=vcat(reshape(logErr_N,(:)),reshape(logErr_T,(:)))
    all_err_def=vcat(reshape(logErr_Ndef,(:)),reshape(logErr_Tdef,(:)))
    all_err=vcat(all_err_rep,all_err_def)

    μ_err=mean(all_err)
    std_err=std(all_err)
    max_err=maximum(all_err)
    min_err=minimum(all_err)

    pct_1=1-(sum(all_err .> -1.0)/length(all_err))
    pct_2=1-(sum(all_err .> -2.0)/length(all_err))
    pct_3=1-(sum(all_err .> -3.0)/length(all_err))

    hst=histogram(all_err,label=false,xlabel="log10(e)")
    return hst, μ_err, std_err, max_err, min_err, pct_1, pct_2, pct_3
end

################################################################
################### Figure 1. Price schedule ###################
################################################################
function Plot_SpreadSchedule(MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack GRIDS_Q = GRIDS
    #Compute where to plot price schedule
    T=100000
    TS=SimulatePaths(0.0,T,MODEL;FixSeed=true)
    zat=1.0#mean(TS.z)
    pat=1

    KN_L=mean(TS.KN)-std(TS.KN)
    KN_M=mean(TS.KN)
    KN_H=mean(TS.KN)+std(TS.KN)

    KT_L=mean(TS.KT)-std(TS.KT)
    KT_M=mean(TS.KT)
    KT_H=mean(TS.KT)+std(TS.KT)

    fstd=0.85
    b_L=mean(TS.B)-fstd*std(TS.B)
    b_H=mean(TS.B)+fstd*std(TS.B)

    avGDP=mean(TS.nGDP)

    YLABEL="r-r*"
    COLORS=[:blue :darkorange]
    LINESTYLES=[:solid :dash]
    YLIMS=[0.0,20.0]

    #Plot as a function of b
    TITLE="as a function of B'"
    LABELS=["low pC" "high pC"]
    xx=collect(range(50.0,stop=80.0,length=100))
    # xx=GRIDS.GR_b
    XLABEL="percentage of av(GDP)"
    foob(by::Float64)=0.01*4*avGDP*by
    fooLb(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(foob(x),KT_M,KN_M,zat,1),par)
    fooHb(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(foob(x),KT_M,KN_M,zat,2),par)
    yyL=fooLb.(xx)
    yyH=fooHb.(xx)
    plt_b=plot(xx,[yyL yyH],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:bottomright,ylims=YLIMS)

    #Plot as a function of KN
    TITLE="as a function of KN'"
    LABELS=["low B'" "high B'"]
    xx=GRIDS_Q.GR_kN[2:end]
    XLABEL="KN'"
    fooLkN(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_L,KT_M,x,zat,1),par)
    fooHkN(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_H,KT_M,x,zat,1),par)
    yyLL=fooLkN.(xx)
    yyHL=fooHkN.(xx)
    plt_N=plot(xx,[yyLL yyHL],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:topleft,ylims=YLIMS)

    #Plot as a function of KT
    TITLE="as a function of KT'"
    LABELS=["low B'" "high B'"]
    xx=GRIDS_Q.GR_kT[2:end]
    XLABEL="KT'"
    fooLkT(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_L,x,KN_M,zat,1),par)
    fooHkT(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_H,x,KN_M,zat,1),par)
    yyLL=fooLkT.(xx)
    yyHL=fooHkT.(xx)
    plt_T=plot(xx,[yyLL yyHL],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:topright,ylims=YLIMS)

    #Plot as a function of Λ
    Λ_TS=TS.KT ./ (TS.KT .+ TS.KN)
    Λll=0.05#0.05
    Λhh=0.10#0.35
    KK=mean(TS.KT .+ TS.KN)
    TITLE="as a function of Λ'"
    LABELS=["low B'" "high B'"]
    xx=collect(range(Λll,stop=Λhh,length=100))
    XLABEL="Λ'"
    fooLΛ(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_L,x*KK,(1-x)*KK,zat,1),par)
    fooHΛ(x::Float64)=100*ComputeSpreadWithQ(SOLUTION.itp_q1(b_H,x*KK,(1-x)*KK,zat,1),par)
    yyLL=fooLΛ.(xx)
    yyHL=fooHΛ.(xx)
    plt_Λ=plot(xx,[yyLL yyHL],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:topright,ylims=YLIMS)

    #Create layout
    size_width=650#400
    size_height=500#350
    l = @layout([a b; c d])
    plt=plot(plt_b,plt_Λ,plt_N,plt_T,
             layout=l,size=(size_width*2,size_height*2))
    return plt
end

################################################################
################## Figure 2. Time series of Λ ##################
################################################################
function Simulate2PathsSameWindfall(Tbefore::Int64,TWind::Int64,Tafter::Int64,
                                    MOD_DEC::Model,MOD_PLA::Model)
    @unpack drp, pCL = MOD_DEC.par
    T=Tbefore+1+TWind+Tafter
    z_TS, p_ind_TS, p_TS=Simulate_AllShocks(drp+T,MOD_DEC.GRIDS,MOD_DEC.par)
    cnt=0
    while maximum(p_ind_TS[drp+1:drp+Tbefore])>1
        #Draw another sample
        z_TS, p_ind_TS, p_TS=Simulate_AllShocks(drp+T,MOD_DEC.GRIDS,MOD_DEC.par)
        cnt=cnt+1
        if cnt>20
            println("could not find p_ind=1 as initial")
            break
        end
    end

    for t in -drp+1:T
        if t<=Tbefore
            # p_ind_TS[drp+t]=1
        else
            if t>Tbefore && t<=Tbefore+1+TWind
                p_ind_TS[drp+t]=2
            else
                p_ind_TS[drp+t]=1
            end
        end
        p_TS[drp+t]=MOD_DEC.GRIDS.GR_p[p_ind_TS[drp+t]]
    end
    KN0=0.5*(MOD_DEC.par.kNlow+MOD_DEC.par.kNhigh)
    KT0=0.5*(MOD_DEC.par.kTlow+MOD_DEC.par.kThigh)
    B0=0.0
    Def0=0.0
    X0=[Def0; KN0; KT0; B0]

    #Simulate decentralized first
    P_DEC=SimulatePathsOfStates(drp,z_TS,p_ind_TS,p_TS,X0,MOD_DEC)
    UpdateOtherVariablesInPaths!(P_DEC,MOD_DEC)

    #Simulate planner
    P_PLA=SimulatePathsOfStates(drp,z_TS,p_ind_TS,p_TS,X0,MOD_PLA)
    UpdateOtherVariablesInPaths!(P_PLA,MOD_PLA)
    return P_DEC, P_PLA
end

function TimeSeries_WelfareGains(P_DEC::Paths,MOD_DEC::Model,MOD_PLA::Model)
    T=length(P_DEC.z)
    TS_WG=Array{Float64,1}(undef,T)
    for t in 1:T
        p_ind=P_DEC.p_ind[t]; z=P_DEC.z[t]
        kN=P_DEC.KN[t]; kT=P_DEC.KT[t]; b=P_DEC.B[t]
        TS_WG[t]=WelfareGains_VF(z,p_ind,kN,kT,b,MOD_DEC,MOD_PLA)
    end
    return TS_WG
end

function AddNewPathToAveragePaths!(N::Int64,PATHS::Paths,NEW_PATH::Paths)
    #Paths of shocks
    PATHS.z=PATHS.z .+ (NEW_PATH.z ./ N)
    PATHS.p=PATHS.p .+ (NEW_PATH.p ./ N)

    #Paths of chosen states
    PATHS.Def=PATHS.Def .+ (NEW_PATH.Def ./ N)
    PATHS.KN=PATHS.KN .+ (NEW_PATH.KN ./ N)
    PATHS.KT=PATHS.KT .+ (NEW_PATH.KT ./ N)

    #Path of relevant variables
    if sum(NEW_PATH.Def)>0.0
        # Ignore spreads and debt average if default along the way
        # just add the current average
        PATHS.B=PATHS.B .+ (PATHS.B ./ N)
        PATHS.Spreads=PATHS.Spreads .+ (PATHS.Spreads ./ N)
    else
        PATHS.B=PATHS.B .+ (NEW_PATH.B ./ N)
        PATHS.Spreads=PATHS.Spreads .+ (NEW_PATH.Spreads ./ N)
    end
    PATHS.GDP=PATHS.GDP .+ (NEW_PATH.GDP ./ N)
    PATHS.nGDP=PATHS.nGDP .+ (NEW_PATH.nGDP ./ N)
    PATHS.P=PATHS.P .+ (NEW_PATH.P ./ N)
    PATHS.GDPdef=PATHS.GDPdef .+ (NEW_PATH.GDPdef ./ N)
    PATHS.Cons=PATHS.Cons .+ (NEW_PATH.Cons ./ N)
    PATHS.Inv=PATHS.Inv .+ (NEW_PATH.Inv ./ N)
    PATHS.InvN=PATHS.InvN .+ (NEW_PATH.InvN ./ N)
    PATHS.InvT=PATHS.InvT .+ (NEW_PATH.InvT ./ N)
    PATHS.TB=PATHS.TB .+ (NEW_PATH.TB ./ N)
    PATHS.CA=PATHS.CA .+ (NEW_PATH.CA ./ N)
    PATHS.RER=PATHS.RER .+ (NEW_PATH.RER ./ N)
end

function Average_2_WindfallPaths(N::Int64,Tbefore::Int64,TWind::Int64,Tafter::Int64,
                                 MOD_DEC::Model,MOD_PLA::Model)
    #Initiate empty paths
    T=Tbefore+1+TWind+Tafter
    P_DEC=InitiateEmptyPaths(T)
    P_PLA=InitiateEmptyPaths(T)
    TS_WG=zeros(Float64,T)
    τsN=zeros(Float64,T)
    τsT=zeros(Float64,T)
    for i in 1:N
        println(i)
        TS_DEC, TS_PLA=Simulate2PathsSameWindfall(Tbefore,TWind,Tafter,MOD_DEC,MOD_PLA)
        # Do not start in default or with high commodity price
        while sum(TS_DEC.Def[1:Tbefore])>0.0 || sum(TS_PLA.Def[1:Tbefore])>0.0
            #Get a new pair
            TS_DEC, TS_PLA=Simulate2PathsSameWindfall(Tbefore,TWind,Tafter,MOD_DEC,MOD_PLA)
        end
        AddNewPathToAveragePaths!(N,P_DEC,TS_DEC)
        AddNewPathToAveragePaths!(N,P_PLA,TS_PLA)

        WG=TimeSeries_WelfareGains(TS_DEC,MOD_DEC,MOD_PLA)
        TS_WG .= TS_WG .+ (WG ./ N)

        τsNa, τsTa=TimeSeriesOfSubsidies(TS_DEC,MOD_DEC)
        τsN=τsN .+ (τsNa ./ N)
        τsT=τsT .+ (τsTa ./ N)
    end
    return P_DEC, P_PLA, TS_WG, τsN, τsT
end

function Plot_TS_K_PvsD(N::Int64,MOD_DEC::Model,MOD_PLA::Model)
    rs=1233
    Tbefore=1
    TWind=15
    Tafter=8
    T=Tbefore+1+TWind+Tafter
    #Simlate decentralized and planner
    Random.seed!(rs)
    TS_D, TS_P, TS_WG, τsN, τsT=Average_2_WindfallPaths(N,Tbefore,TWind,Tafter,MOD_DEC,MOD_PLA)
    # τsN, τsT=TimeSeriesOfSubsidies(TS_D,MOD_DEC)

    XLABEL="quarters"
    tt=[-Tbefore:TWind+Tafter]

    #Make plot of price of commodities
    TITLE="Commodity price"
    plt_p=plot(tt,TS_P.p,title=TITLE,ylabel="pC",
               legend=false,linecolor=:black,xlabel=XLABEL)

    #Make plot of invN
    TITLE="non-traded investment"
    YLABEL="percentage change"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*((TS_P.InvN ./ TS_P.InvN[1]) .- 1)
    yy_D=100*((TS_D.InvN ./ TS_D.InvN[1]) .- 1)
    plt_iN=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:topright,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of invR
    TITLE="traded investment"
    YLABEL="percentage change"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*((TS_P.InvT ./ TS_P.InvT[1]) .- 1)
    yy_D=100*((TS_D.InvT ./ TS_D.InvT[1]) .- 1)
    plt_iT=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:topright,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of investment
    TITLE="investment"
    YLABEL="percentage change"
    LABELS=["planner, N" "decentralized, N" "planner, T" "decentralized, T"]
    COLORS=[:green :darkorange :green :darkorange]
    LINESTYLES=[:solid :dash :dot :dashdot]
    yy_PN=100*((TS_P.InvN ./ TS_P.InvN[1]) .- 1)
    yy_DN=100*((TS_D.InvN ./ TS_D.InvN[1]) .- 1)
    yy_PT=100*((TS_P.InvT ./ TS_P.InvT[1]) .- 1)
    yy_DT=100*((TS_D.InvT ./ TS_D.InvT[1]) .- 1)
    YLIMS=[-0.5,19.0]
    plt_i=plot(tt,[yy_PN yy_DN yy_PT yy_DT],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:topleft,
               linestyle=LINESTYLES,title=TITLE,ylims=YLIMS)

    #Make plot of kN
    TITLE="non-traded capital"
    YLABEL="kN"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.KN
    yy_D=TS_D.KN
    plt_N=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:bottom,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of kN %
    TITLE="non-traded capital"
    YLABEL="percentage change"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*((TS_P.KN ./ TS_P.KN[1]) .- 1)
    yy_D=100*((TS_D.KN ./ TS_D.KN[1]) .- 1)
    plt_Npp=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:bottom,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of kT
    TITLE="traded capital"
    YLABEL="kT"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.KT
    yy_D=TS_D.KT
    plt_T=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:best,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of kT %
    TITLE="traded capital"
    YLABEL="kT"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*((TS_P.KT ./ TS_P.KT[1]) .- 1)
    yy_D=100*((TS_D.KT ./ TS_D.KT[1]) .- 1)
    plt_Tpp=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:best,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of total capital
    TITLE="Total capital"
    YLABEL="kT+kN"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.KT .+ TS_P.KN
    yy_D=TS_D.KT .+ TS_D.KN
    plt_K=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:best,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of share in traded
    TITLE="share of capital in T"
    YLABEL="percentage"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*TS_P.KT ./ (TS_P.KT .+ TS_P.KN)
    yy_D=100*TS_D.KT ./ (TS_D.KT .+ TS_D.KN)
    plt_Λ=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:left,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of optimal subsidy
    TITLE="Sectoral capital wedge"
    YLABEL="τT-τN"
    LABELS=["τT-τN"]
    COLORS=[:black]
    LINESTYLES=[:solid]
    plt_τ=plot(tt,[τsT .- τsN],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=false,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of spreads
    TITLE="Spreads"
    YLABEL="percentage points"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.Spreads
    yy_D=TS_D.Spreads
    plt_spr=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:topright,
               linestyle=LINESTYLES,title=TITLE)#,ylims=[0.0,50.0])

    #Make plot of b
    TITLE="debt"
    YLABEL="percentage of average GDP"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*TS_P.B ./ (4*TS_P.nGDP[1])
    yy_D=100*TS_D.B ./ (4*TS_D.nGDP[1])
    plt_b=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:top,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of P
    TITLE="price of final good"
    YLABEL="P"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.P
    yy_D=TS_D.P
    plt_P=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:best,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of default state
    TITLE="fraction in default"
    YLABEL="percentage"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.Def
    yy_D=TS_D.Def
    plt_def=plot(tt,100*[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:topleft,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of consumption
    TITLE="consumption difference"
    YLABEL="percentage points"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.Cons
    yy_D=TS_D.Cons
    yy=100*((yy_P ./ yy_D) .-1)
    plt_c=plot(tt,yy,label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=false,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of trade balance
    TITLE="trade balance"
    YLABEL="percentage of GDP"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=100*TS_P.TB ./ TS_P.nGDP
    yy_D=100*TS_D.TB ./ TS_D.nGDP
    plt_tb=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=false,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of welfare gains
    TITLE="welfare gains"
    YLABEL="consumption equivalent units"
    COLORS=[:black]
    LINESTYLES=[:solid]
    plt_wg=plot(tt,TS_WG,label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=false,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of welfare losses
    TITLE="welfare losses"
    YLABEL="consumption equivalent units"
    COLORS=[:black]
    LINESTYLES=[:solid]
    TS_WL=100*((1 ./ (1 .+ 0.01*TS_WG)) .- 1)
    plt_wl=plot(tt,-TS_WL,label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=false,
               linestyle=LINESTYLES,title=TITLE)

    #Create layout
    size_width=650
    size_height=500
    l = @layout([a b; c d; e f])
    plt=plot(plt_p,plt_spr,
             plt_i,plt_Λ,
             plt_wl,plt_τ,
             layout=l,size=(size_width*2,size_height*3))
    return plt
end
