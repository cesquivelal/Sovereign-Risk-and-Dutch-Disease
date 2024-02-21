
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                    markersize=9,size=(650,500))


################################################################
################### Figure 1. Price schedule ###################
################################################################
function Plot_q1(SOL::Solution,GRIDS::Grids,par::Pars)
    #Compute where to plot price schedule
    T=100000
    TS=SimulatePaths(T,SOL,GRIDS,par)
    zat=1.0#mean(TS.z)
    pat=2
    Λ_L=mean(TS.Λ)-std(TS.Λ)
    Λ_M=mean(TS.Λ)
    Λ_H=mean(TS.Λ)+std(TS.Λ)
    b_L=mean(TS.B)-2*std(TS.B)
    b_H=mean(TS.B)+2*std(TS.B)

    YLABEL="q(x',s)"
    COLORS=[:blue :darkorange]
    LINESTYLES=[:solid :dash]
    YLIMS=[0.0,1.35]

    #Plot as a function of b
    TITLE="as a function of B'"
    LABELS=["low pC" "high pC"]
    xx=GRIDS.GR_b
    XLABEL="B'"
    fooLb(x::Float64)=SOL.itp_q1(x,Λ_M,zat,1)
    fooHb(x::Float64)=SOL.itp_q1(x,Λ_M,zat,2)
    yyL=fooLb.(xx)
    yyH=fooHb.(xx)
    plt_b=plot(xx,[yyL yyH],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:topright)#,ylims=YLIMS)

    #Plot as a function of Λ
    TITLE="as a function of Λ'"
    LABELS=["low B'" "high B'"]
    xx=GRIDS.GR_Λ
    XLABEL="Λ'"
    fooLΛpL(x::Float64)=SOL.itp_q1(b_L,x,zat,1)
    fooHΛpL(x::Float64)=SOL.itp_q1(b_H,x,zat,1)
    yyLL=fooLΛpL.(xx)
    yyHL=fooHΛpL.(xx)
    plt_Λ=plot(xx,[yyLL yyHL],title=TITLE,label=LABELS,
               xlabel=XLABEL,ylabel=YLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               legend=:bottomright)#,ylims=[0.7,1.25])

    #Create layout
    size_width=650
    size_height=500
    l = @layout([a b])
    plt=plot(plt_b,plt_Λ,
             layout=l,size=(size_width*2,size_height*1))
    return plt
end

################################################################
################## Figure 2. Time series of Λ ##################
################################################################
function Simulate2PathsSameWindfall(Tbefore::Int64,TWind::Int64,Tafter::Int64,
                                    SOL::Solution,GRIDS::Grids,par::Pars,
                                    SOL_0::Solution,GRIDS_0::Grids,par_0::Pars)
    @unpack drp, pCL = par
    T=Tbefore+TWind+Tafter
    z_TS, p_ind_TS, p_TS=Simulate_AllShocks(drp+T,GRIDS,par)
    for t in -11:T
        if t<=Tbefore
            p_ind_TS[drp+t]=1
        else
            if t>Tbefore && t<=Tbefore+TWind
                p_ind_TS[drp+t]=2
            else
                p_ind_TS[drp+t]=1
            end
        end
        p_TS[drp+t]=GRIDS.GR_p[p_ind_TS[drp+t]]
    end
    Λ0=Λsteady_state(pCL,par)
    B0=0.0
    Def0=0.0
    X0=[Def0; Λ0; B0]

    PATHS=SimulatePathsOfStates(drp,z_TS,p_ind_TS,p_TS,X0,SOL,GRIDS,par)
    UpdateOtherVariablesInPaths!(PATHS,SOL,GRIDS,par)

    PATHS_0=SimulatePathsOfStates(drp,z_TS,p_ind_TS,p_TS,X0,SOL_0,GRIDS_0,par_0)
    UpdateOtherVariablesInPaths!(PATHS_0,SOL_0,GRIDS_0,par_0)
    return PATHS, PATHS_0
end

function AddNewPathToAveragePaths!(N::Int64,PATHS::Paths,NEW_PATH::Paths)
    #Paths of shocks
    PATHS.z=PATHS.z .+ (NEW_PATH.z ./ N)
    PATHS.p=PATHS.p .+ (NEW_PATH.p ./ N)

    #Paths of chosen states
    PATHS.Def=PATHS.Def .+ (NEW_PATH.Def ./ N)
    PATHS.Λ=PATHS.Λ .+ (NEW_PATH.Λ ./ N)
    PATHS.B=PATHS.B .+ (NEW_PATH.B ./ N)

    #Path of relevant variables
    PATHS.Spreads=PATHS.Spreads .+ (NEW_PATH.Spreads ./ N)
    PATHS.GDP=PATHS.GDP .+ (NEW_PATH.GDP ./ N)
    PATHS.nGDP=PATHS.nGDP .+ (NEW_PATH.nGDP ./ N)
    PATHS.P=PATHS.P .+ (NEW_PATH.P ./ N)
    PATHS.GDPdef=PATHS.GDPdef .+ (NEW_PATH.GDPdef ./ N)
    PATHS.Cons=PATHS.Cons .+ (NEW_PATH.Cons ./ N)
    PATHS.TB=PATHS.TB .+ (NEW_PATH.TB ./ N)
    PATHS.CA=PATHS.CA .+ (NEW_PATH.CA ./ N)
    PATHS.RER=PATHS.RER .+ (NEW_PATH.RER ./ N)
end

function Average_2_WindfallPaths(N::Int64,Tbefore::Int64,TWind::Int64,Tafter::Int64,
                                 SOL::Solution,GRIDS::Grids,par::Pars,
                                 SOL_0::Solution,GRIDS_0::Grids,par_0::Pars)
    #Initiate empty paths
    T=Tbefore+TWind+Tafter
    PATHS=InitiateEmptyPaths(T)
    PATHS_0=InitiateEmptyPaths(T)

    for i in 1:N
        TS, TS_0=Simulate2PathsSameWindfall(Tbefore,TWind,Tafter,SOL,GRIDS,par,SOL_0,GRIDS_0,par_0)
        AddNewPathToAveragePaths!(N,PATHS,TS)
        AddNewPathToAveragePaths!(N,PATHS_0,TS_0)
    end
    return PATHS, PATHS_0
end

function Plot_TS_Λ_PvsD(N::Int64,SOL_DEC::Solution,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    rs=12333
    Tbefore=5
    TWind=13
    Tafter=10
    T=Tbefore+TWind+Tafter
    #Simlate decentralized and planner
    Random.seed!(rs)
    TS_P, TS_D=Average_2_WindfallPaths(N,Tbefore,TWind,Tafter,SOL_PLA,GRIDS,par,SOL_DEC,GRIDS,par)
    τs=TimeSeriesOfSubsidies(TS_P,SOL_PLA,par)

    XLABEL="quarters"
    tt=[-Tbefore:TWind+Tafter-1]

    #Make plot of price of commodities
    TITLE="Commodity price"
    plt_p=plot(tt,TS_P.p,title=TITLE,ylabel="pC",
               legend=false,linecolor=:blue,xlabel=XLABEL)

    #Make plot of Λ
    TITLE="Capital allocation"
    YLABEL="Λ"
    LABELS=["planner" "decentralized"]
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    yy_P=TS_P.Λ
    yy_D=TS_D.Λ
    plt_Λ=plot(tt,[yy_P yy_D],label=LABELS,linecolor=COLORS,
               xlabel=XLABEL,ylabel=YLABEL,legend=:best,
               linestyle=LINESTYLES,title=TITLE)

    #Make plot of optimal subsidy
    TITLE="Optimal subsidy"
    YLABEL="percentage points"
    COLOR=[:green]
    LINESTYLES=[:solid]
    yy_P=τs
    plt_τ=plot(tt,yy_P,label=LABELS,linecolor=COLORS,
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
               xlabel=XLABEL,ylabel=YLABEL,legend=:topleft,
               linestyle=LINESTYLES,title=TITLE)

    #Create layout
    size_width=650
    size_height=500
    l = @layout([a b; c d])
    plt=plot(plt_p,plt_Λ,
             plt_τ,plt_spr,
             layout=l,size=(size_width*2,size_height*2))
    return plt
end

################################################################
###################### Plot subsidy rules ######################
################################################################
function PlotSubsidyRules(FIXED::String,SPREADS::String,BOTH::String)
    MAT_FIXED=readdlm(FIXED,',')
    MAT_SPREADS=readdlm(SPREADS,',')
    τ_GRID=MAT_FIXED[:,1]
    wg_f=MAT_FIXED[:,2]
    wg_spr=MAT_SPREADS[:,2]
    YLIMS=[0.0,0.04]

    XLABEL="τ0, τ1"
    TITLE="Fixed rule and linear rule with τ0=0"
    COLORS=[:green :darkorange]
    LINESTYLES=[:solid :dash]
    LABELS=["fixed" "linear rule"]
    YLABEL="percentage points, c.e. units"
    plt_1=plot(τ_GRID,[wg_f wg_spr],ylims=YLIMS,
               xlabel=XLABEL,ylabel=YLABEL,
               label=LABELS,legend=:topleft,
               linecolor=COLORS,title=TITLE,
               linestyle=LINESTYLES)

    MAT_BOTH=readdlm(BOTH,',')
    wg_both=MAT_BOTH[:,2]
    τ_GRID=MAT_BOTH[:,1]
    XLABEL="τ1"
    TITLE="Linear rule with τ0=0.05"
    COLORS=[:blue]
    LINESTYLES=[:solid]
    YLABEL="percentage points, c.e. units"
    plt_2=plot(τ_GRID,wg_both,ylims=YLIMS,
               xlabel=XLABEL,ylabel=YLABEL,
               label=LABELS,legend=false,
               linecolor=COLORS,title=TITLE,
               linestyle=LINESTYLES)

    #Create layout
    size_width=650
    size_height=500
    l = @layout([a b])
    plt=plot(plt_1,plt_2,
             layout=l,size=(size_width*2,size_height*1))
    return plt
end
