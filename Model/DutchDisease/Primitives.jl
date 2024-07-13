
using Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Roots, Distributed

################################################################
#### Defining parameters and other structures for the model ####
################################################################
#Define parameter and grid structure, use quarterly calibration
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.95         #Discount factor consistent with annualized r=8.4%
    r_star::Float64 = 0.01    #Risk-fr  ee interest rate
    #Debt parameters
    γ::Float64 = 0.024         #Reciprocal of average maturity
    κ::Float64 = 0.00         #Coupon payments
    Covenants::Bool = false
    #Default cost and debt parameters
    θ::Float64 = 1/(mean([12 1 6])*4)    #Probability of re-admission
    d0::Float64 = -0.18819#-0.2914       #income default cost
    d1::Float64 = 0.24558#0.4162        #income default cost
    Arellano::Bool = false
    knk::Float64 = 0.90
    #Capital allocations
    WithInvTax::Bool = false #Consider distortionary tax
    τsN0::Float64 = 0.0       #Subsidy for kN, intercept
    τsN1::Float64 = 0.0       #Subsidy for kN, slope
    τsT0::Float64 = 0.0       #Subsidy for kT, intercept
    τsT1::Float64 = 0.0       #Subsidy for kT, slope
    φ::Float64 = 5.5        #Adjustment cost
    δ::Float64 = 0.05       #Capital depreciation rate
    #Production functions
    #Final consumption good
    η::Float64 = 0.83
    ωN::Float64 = 0.66
    ωT::Float64 = 1-ωN#0.30
    ωC::Float64 = 0.0
    #Value added of intermediates
    αN::Float64 = 0.36#0.66          #Capital share in non-traded sector
    αT::Float64 = 0.36#0.57          #Capital share in manufacturing sector
    #Commodity endowment
    yC::Float64 = 5.1
    #Stochastic process
    #parameters for productivity shock
    σ_ϵz::Float64 = 0.017
    μ_ϵz::Float64 = 0.0-0.5*(σ_ϵz^2)
    # dist_ϵz::UnivariateDistribution = truncated(Normal(μ_ϵz,σ_ϵz),-3.0*σ_ϵz,3.0*σ_ϵz)
    dist_ϵz::UnivariateDistribution = Normal(μ_ϵz,σ_ϵz)
    ρ_z::Float64 = 0.95
    μ_z::Float64 = exp(μ_ϵz+0.5*(σ_ϵz^2))
    zlow::Float64 = exp(log(μ_z)-3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #parameters for commodity price shock
    AvDevFromMean::Float64 = 0.406
    pCL::Float64 = 0.8186          #Normal times
    pCH::Float64 = 1.2215          #Commodity windfall
    L_dur::Float64 = 12.8          #Average quarters low price
    H_dur::Float64 = 16.0          #Average quarters high price
    πp::Float64 = 0.855
    #Quadrature parameter
    N_GLz::Int64 = 50
    #Grids
    Nz::Int64 = 11#31
    Np::Int64 = 2
    NkN::Int64 = 11#21
    NkT::Int64 = 11#21
    Nb::Int64 = 21#31
    kNlow::Float64 = 0.01#0.01
    kNhigh::Float64 = 25.0#25.0
    kTlow::Float64 = 0.01#0.01
    kThigh::Float64 = 5.0#5.0
    blow::Float64 = 0.0
    bhigh::Float64 = 20.0#1.5
    #Parameters for solution algorithm
    cmin::Float64 = 1e-4
    Tol_V::Float64 = 1e-6       #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-6       #Tolerance for absolute distance for q
    Tolpct_q::Float64 = 1.0        #Tolerance for % of states for which q has not converged
    cnt_max::Int64 = 300           #Maximum number of iterations on VFI
    kNlowOpt::Float64 = 0.9*kNlow             #Minimum level of capital for optimization
    kNhighOpt::Float64 = 1.1*kNhigh           #Maximum level of capital for optimization
    kTlowOpt::Float64 = 0.9*kTlow             #Minimum level of capital for optimization
    kThighOpt::Float64 = 1.1*kThigh           #Maximum level of capital for optimization
    blowOpt::Float64 = blow-0.01             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    #Simulation parameters
    Tsim::Int64 = 10000
    drp::Int64 = 5000
    Tmom::Int64 = 400
    NSamplesMoments::Int64 = 100
    Npaths::Int64 = 100
    HPFilter_Par::Float64 = 1600.0

    #Grid sizes for price q1
    q1_N_GLz::Int64 = 50
    #Grids
    q1_Nz::Int64 = 11#31
    q1_Np::Int64 = 2
    q1_NkN::Int64 = 11#21
    q1_NkT::Int64 = 11#21
    q1_Nb::Int64 = 21#31
end

@with_kw struct Grids_q
    #Size of grids for q
    N_GLz::Int64
    Nz::Int64
    Np::Int64
    NkN::Int64
    NkT::Int64
    Nb::Int64
    #Grids of states
    GR_z::Array{Float64,1}
    GR_p::Array{Float64,1}
    GR_kN::Array{Float64,1}
    GR_kT::Array{Float64,1}
    GR_b::Array{Float64,1}
    #Quadrature vectors for z integrals
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}
    #Factor to correct quadrature underestimation
    FacQz::Float64
    #Quadrature matrices for z integrals
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,1}
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64,1}
    GR_p::Array{Float64,1}
    GR_kN::Array{Float64,1}
    GR_kT::Array{Float64,1}
    GR_b::Array{Float64,1}
    #Quadrature vectors for z integrals
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}
    #Factor to correct quadrature underestimation
    FacQz::Float64
    #Quadrature matrices for z integrals
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,1}
    #Transition matrix for pC
    PI_pC::Array{Float64,2}

    #Grids for q
    GRIDS_Q::Grids_q
end

function CreateGrids_q(par::Pars)
    #Size of grids for q
    N_GLz=par.q1_N_GLz
    Nz=par.q1_Nz
    Np=par.q1_Np
    NkN=par.q1_NkN
    NkT=par.q1_NkT
    Nb=par.q1_Nb

    #Grid for z, same
    @unpack zlow, zhigh = par
    lnGR_z=collect(range(log(zlow),stop=log(zhigh),length=Nz))
    GR_z=exp.(lnGR_z)

    #Grid for pC, same
    @unpack pCL, pCH = par
    GR_p=Array{Float64,1}(undef,2)
    GR_p[1]=pCL
    GR_p[2]=pCH

    #Grid of capital allocation, go to very lower values too
    @unpack kNlow, kNhigh = par
    GR_kN=collect(range(kNlow,stop=kNhigh,length=NkN))

    #Grid of capital allocation, go to very lower values too
    @unpack kTlow, kThigh = par
    GR_kT=collect(range(kTlow,stop=kThigh,length=NkT))

    #Grid of debt, same
    @unpack blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    #Gauss-Legendre vectors for z
    @unpack σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    GL_nodes, GL_weights = gausslegendre(N_GLz)
    ϵzlow=-3.0*σ_ϵz
    ϵzhigh=3.0*σ_ϵz
    ϵz_nodes=0.5*(ϵzhigh-ϵzlow).*GL_nodes .+ 0.5*(ϵzhigh+ϵzlow)
    ϵz_weights=GL_weights .* 0.5*(ϵzhigh-ϵzlow)
    #Matrices for integration over z
    ZPRIME=Array{Float64,2}(undef,Nz,N_GLz)
    PDFz=pdf.(dist_ϵz,ϵz_nodes)
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ_z)*log(μ_z)+ρ_z*log(z) .+ ϵz_nodes)
    end
    FacQz=dot(ϵz_weights,PDFz)

    return Grids_q(N_GLz,Nz,Np,NkN,NkT,Nb,GR_z,GR_p,GR_kN,GR_kT,GR_b,ϵz_weights,ϵz_nodes,FacQz,ZPRIME,PDFz)
end

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    lnGR_z=collect(range(log(zlow),stop=log(zhigh),length=Nz))
    GR_z=exp.(lnGR_z)

    #Gauss-Legendre vectors for z
    @unpack N_GLz, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    GL_nodes, GL_weights = gausslegendre(N_GLz)
    ϵzlow=-3.0*σ_ϵz
    ϵzhigh=3.0*σ_ϵz
    ϵz_nodes=0.5*(ϵzhigh-ϵzlow).*GL_nodes .+ 0.5*(ϵzhigh+ϵzlow)
    ϵz_weights=GL_weights .* 0.5*(ϵzhigh-ϵzlow)
    #Matrices for integration over z
    ZPRIME=Array{Float64,2}(undef,Nz,N_GLz)
    PDFz=pdf.(dist_ϵz,ϵz_nodes)
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ_z)*log(μ_z)+ρ_z*log(z) .+ ϵz_nodes)
    end
    FacQz=dot(ϵz_weights,PDFz)

    #Grid for pC and transition matrix
    @unpack pCL, pCH, L_dur, H_dur, πp = par
    GR_p=Array{Float64,1}(undef,2)
    GR_p[1]=pCL
    GR_p[2]=pCH
    PI_pC=Array{Float64,2}(undef,2,2)
    πW=1/L_dur
    PI_pC[1,1]=πp#1.0-PI_pC[1,2]
    PI_pC[2,2]=πp#1.0-PI_pC[2,1]
    PI_pC[1,2]=1-πp#πW
    PI_pC[2,1]=1-πp#1.0/H_dur

    #Grid of capital allocation
    @unpack NkN, kNlow, kNhigh = par
    GR_kN=collect(range(kNlow,stop=kNhigh,length=NkN))

    #Grid of capital allocation
    @unpack NkT, kTlow, kThigh = par
    GR_kT=collect(range(kTlow,stop=kThigh,length=NkT))

    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    #Grids for q
    GRIDS_Q=CreateGrids_q(par)

    return Grids(GR_z,GR_p,GR_kN,GR_kT,GR_b,ϵz_weights,ϵz_nodes,FacQz,ZPRIME,PDFz,PI_pC,GRIDS_Q)
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6,T7}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    #Policy functions
    kNprime_D::T1
    kTprime_D::T1
    kNprime::T2
    kTprime::T2
    bprime::T2
    Tr::T2
    ### Interpolation objects
    #Value Functions
    itp_VD::T3
    itp_VP::T4
    itp_V::T4
    #Expectations and price
    itp_EVD::T3
    itp_EV::T4
    itp_q1::T5
    #Policy functions
    itp_kNprime_D::T6
    itp_kTprime_D::T6
    itp_kNprime::T7
    itp_kTprime::T7
    itp_bprime::T7
end

@with_kw struct Model
    SOLUTION::Solution
    GRIDS::Grids
    par::Pars
end

@with_kw mutable struct SharedAux{T1,T2}
    sVD::T1
    skNprime_D::T1
    skTprime_D::T1

    sVP::T2
    sV::T2
    skNprime::T2
    skTprime::T2
    sbprime::T2
    sTr::T2

    sq1::T2

    sdRN_Def::T1
    sdRT_Def::T1
    sdRN_Rep::T2
    sdRT_Rep::T2

    sSMOOTHED::T2
end

function InitiateSharedAux(GRIDS::Grids,par::Pars)
    @unpack Nz, Np, NkN, NkT, Nb = par
    @unpack GRIDS_Q = GRIDS
    ### Allocate all values to object
    sVD=SharedArray{Float64}(NkT,NkN,Nz,Np)
    skNprime_D=SharedArray{Float64}(NkT,NkN,Nz,Np)
    skTprime_D=SharedArray{Float64}(NkT,NkN,Nz,Np)

    sVP=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    sV=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    skNprime=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    skTprime=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    sbprime=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    sTr=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)

    sq1=SharedArray{Float64}(GRIDS_Q.Nb,GRIDS_Q.NkT,GRIDS_Q.NkN,GRIDS_Q.Nz,GRIDS_Q.Np)

    sRN_D=SharedArray{Float64}(NkT,NkN,Nz,Np)
    sRT_D=SharedArray{Float64}(NkT,NkN,Nz,Np)
    sRN_P=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)
    sRT_P=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)

    sSMOOTHED=SharedArray{Float64}(Nb,NkT,NkN,Nz,Np)

    for I in CartesianIndices(sVD)
        sVD[I]=0.0
        skNprime_D[I]=0.0
        skTprime_D[I]=0.0
        sRN_D[I]=0.0
        sRT_D[I]=0.0
    end

    for I in CartesianIndices(sV)
        sVP[I]=0.0
        sV[I]=0.0
        skNprime[I]=0.0
        skTprime[I]=0.0
        sbprime[I]=0.0
        sTr[I]=0.0
        sRN_P[I]=0.0
        sRT_P[I]=0.0
        sSMOOTHED[I]=0.0
    end

    return SharedAux(sVD,skNprime_D,skTprime_D,sVP,sV,skNprime,skTprime,sbprime,sTr,sq1,sRN_D,sRT_D,sRN_P,sRT_P,sSMOOTHED)
end

function Setup(β::Float64,d0::Float64,d1::Float64,φ::Float64;η::Float64=0.83)
    par=Pars(β=β,d0=d0,d1=d1,φ=φ,η=η)
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

################################################################
#################### Auxiliary functions #######################
################################################################
function MyBisection(foo,a::Float64,b::Float64;xatol::Float64=1e-8)
    s=sign(foo(a))
    x=0.5*(a+b)
    d=0.5*(b-a)
    while d>xatol
        d=0.5*d
        if s==sign(foo(x))
            x=x+d
        else
            x=x-d
        end
    end
    return x
end

function TransformIntoBounds(x::Float64,min::Float64,max::Float64)
    (max - min) * (1.0/(1.0 + exp(-x))) + min
end

function TransformIntoReals(x::Float64,min::Float64,max::Float64)
    log((x - min)/(max - x))
end

################################################################
############# Functions to interpolate matrices ################
################################################################
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()

    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs,Ps),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        ORDER_B_STATES=Linear()
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs,Ps),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GRIDS_Q = GRIDS
    @unpack GR_z, GR_p, GR_kN, GR_kT, GR_b = GRIDS_Q
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Reflect(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs,Ps),Interpolations.Flat())
end

function CreateInterpolation_Policies(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs,Ps),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs,Ps),Interpolations.Flat())
    end
end

function CreateInterpolation_ForExpectations(MAT::Array{Float64,2},GRIDS::Grids)
    @unpack GR_z, GR_p = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs,Ps),Interpolations.Line())
end

function CreateInterpolation_HouseholdObjects(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()
    # ORDER_K_STATES=Cubic(Line(OnGrid()))
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs,Ps),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        ORDER_B_STATES=Linear()
        # ORDER_B_STATES=Cubic(Line(OnGrid()))
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs,Ps),Interpolations.Line())
    end
end

################################################################################
### Functions to save solution in CSV
################################################################################
function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nz, Np, NkN, NkT, Nb = par
    @unpack GRIDS_Q = GRIDS
    ### Allocate all values to object
    VD=zeros(Float64,NkT,NkN,Nz,Np)
    VP=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    V=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,NkT,NkN,Nz,Np)
    EV=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    q1=zeros(Float64,GRIDS_Q.Nb,GRIDS_Q.NkT,GRIDS_Q.NkN,GRIDS_Q.Nz,GRIDS_Q.Np)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    #Policy functions
    kNprime_D=zeros(Float64,NkT,NkN,Nz,Np)
    kTprime_D=zeros(Float64,NkT,NkN,Nz,Np)
    kNprime=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    kTprime=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    bprime=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    Tr=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    itp_kNprime_D=CreateInterpolation_Policies(kNprime_D,true,GRIDS)
    itp_kTprime_D=CreateInterpolation_Policies(kTprime_D,true,GRIDS)
    itp_kNprime=CreateInterpolation_Policies(kNprime,false,GRIDS)
    itp_kTprime=CreateInterpolation_Policies(kTprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kNprime_D,kTprime_D,kNprime,kTprime,bprime,Tr,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kNprime_D,itp_kTprime_D,itp_kNprime,itp_kTprime,itp_bprime)
end

function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment first
    @unpack VP, V, EV, q1 = SOLUTION
    @unpack kNprime, kTprime, bprime, Tr = SOLUTION
    VEC=reshape(q1,(:))

    VEC=vcat(VEC,reshape(VP,(:)))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(kNprime,(:)))
    VEC=vcat(VEC,reshape(kTprime,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))
    VEC=vcat(VEC,reshape(Tr,(:)))

    #Then stack vectors of default
    @unpack VD, EVD, kNprime_D, kTprime_D = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))
    VEC=vcat(VEC,reshape(kNprime_D,(:)))
    VEC=vcat(VEC,reshape(kTprime_D,(:)))

    return VEC
end

function SaveSolution_Vector(NAME::String,SOLUTION::Solution)
    #Stack vectors of repayment first
    VEC=StackSolution_Vector(SOLUTION)
    writedlm(NAME,VEC,',')

    return nothing
end

function VectorOfRelevantParameters(par::Pars)
    #Stack important values from parameters
    #Parameters for Grids go first
    #Grids sizes
    VEC=par.N_GLz               #1
    VEC=vcat(VEC,par.Nz)        #2
    VEC=vcat(VEC,par.Np)        #3
    VEC=vcat(VEC,par.NkN)       #4
    VEC=vcat(VEC,par.NkT)       #5
    VEC=vcat(VEC,par.Nb)        #6

    #Grids bounds
    VEC=vcat(VEC,par.kNlow)     #7
    VEC=vcat(VEC,par.kNhigh)    #8
    VEC=vcat(VEC,par.kTlow)     #9
    VEC=vcat(VEC,par.kThigh)    #10
    VEC=vcat(VEC,par.blow)      #11
    VEC=vcat(VEC,par.bhigh)     #12

    #Parameter values
    VEC=vcat(VEC,par.η)         #13
    VEC=vcat(VEC,par.yC)        #14
    VEC=vcat(VEC,par.β)         #15
    VEC=vcat(VEC,par.d0)        #16
    VEC=vcat(VEC,par.d1)        #17
    VEC=vcat(VEC,par.φ)         #18

    #Extra parameters
    VEC=vcat(VEC,par.cnt_max)   #19

    #Grid size for price q1
    VEC=vcat(VEC,par.q1_N_GLz)     #20
    VEC=vcat(VEC,par.q1_Nz)        #21
    VEC=vcat(VEC,par.q1_Np)        #22
    VEC=vcat(VEC,par.q1_NkN)       #23
    VEC=vcat(VEC,par.q1_NkT)       #24
    VEC=vcat(VEC,par.q1_Nb)        #25

    #Alternative Arellano cost
    if par.Arellano
        VEC=vcat(VEC,1.0)          #26
    else
        VEC=vcat(VEC,0.0)          #26
    end
    VEC=vcat(VEC,par.knk)          #27

    #More parameters
    VEC=vcat(VEC,par.γ)            #28

    return VEC
end

function SaveModel_Vector(NAME::String,SOLUTION::Solution,par::Pars)
    VEC_PAR=VectorOfRelevantParameters(par)
    N_parameters=length(VEC_PAR)
    VEC=vcat(N_parameters,VEC_PAR)

    #Stack SOLUTION in one vector
    VEC_SOL=StackSolution_Vector(SOLUTION)
    VEC=vcat(VEC,VEC_SOL)

    writedlm(NAME,VEC,',')
    return nothing
end

function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,VEC::Array{Float64},par::Pars)
    @unpack Nz, Np, NkN, NkT, Nb = par
    if IsDefault
        I=(NkT,NkN,Nz,Np)
    else
        I=(Nb,NkT,NkN,Nz,Np)
    end
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function ExtractPriceFromSolutionVector(start::Int64,size::Int64,VEC::Array{Float64},GRIDS::Grids)
    @unpack GRIDS_Q = GRIDS
    @unpack Nz, Np, NkN, NkT, Nb = GRIDS_Q
    I=(Nb,NkT,NkN,Nz,Np)
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function TransformVectorToSolution(VEC::Array{Float64},GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nz, Np, NkN, NkT, Nb = par
    size_repayment=Np*Nz*NkN*NkT*Nb
    size_default=Np*Nz*NkN*NkT

    @unpack GRIDS_Q = GRIDS
    size_q1=GRIDS_Q.Np*GRIDS_Q.Nz*GRIDS_Q.NkN*GRIDS_Q.NkT*GRIDS_Q.Nb

    #Allocate vectors into matrices
    #Price q1
    start=1
    q1=ExtractPriceFromSolutionVector(start,size_q1,VEC,GRIDS)
    start=start+size_q1

    #Repayment
    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    kNprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    kTprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    Tr=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    #Default
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    kNprime_D=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    kTprime_D=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    itp_kNprime_D=CreateInterpolation_Policies(kNprime_D,true,GRIDS)
    itp_kTprime_D=CreateInterpolation_Policies(kTprime_D,true,GRIDS)

    itp_kNprime=CreateInterpolation_Policies(kNprime,false,GRIDS)
    itp_kTprime=CreateInterpolation_Policies(kTprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)

    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kNprime_D,kTprime_D,kNprime,kTprime,bprime,Tr,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kNprime_D,itp_kTprime_D,itp_kNprime,itp_kTprime,itp_bprime)
end

function UnpackParameters_Vector(VEC::Array{Float64})
    par=Pars()

    #Parameters for Grids go first
    #Grids sizes
    par=Pars(par,N_GLz=convert(Int64,VEC[1]))
    par=Pars(par,Nz=convert(Int64,VEC[2]))
    par=Pars(par,Np=convert(Int64,VEC[3]))
    par=Pars(par,NkN=convert(Int64,VEC[4]))
    par=Pars(par,NkT=convert(Int64,VEC[5]))
    par=Pars(par,Nb=convert(Int64,VEC[6]))

    #Grids bounds
    par=Pars(par,kNlow=VEC[7])
    par=Pars(par,kNhigh=VEC[8])
    par=Pars(par,kNlowOpt=0.75*par.kNlow,kNhighOpt=1.25*par.kNhigh)

    par=Pars(par,kTlow=VEC[9])
    par=Pars(par,kThigh=VEC[10])
    par=Pars(par,kTlowOpt=0.75*par.kTlow,kThighOpt=1.25*par.kThigh)

    par=Pars(par,blow=VEC[11])
    par=Pars(par,bhigh=VEC[12])
    par=Pars(par,blowOpt=par.blow-0.01,bhighOpt=par.bhigh+0.1)

    #Parameter values
    par=Pars(par,η=VEC[13])
    par=Pars(par,yC=VEC[14])
    par=Pars(par,β=VEC[15])
    par=Pars(par,d0=VEC[16])
    par=Pars(par,d1=VEC[17])
    par=Pars(par,φ=VEC[18])

    #Extra parameters
    par=Pars(par,cnt_max=VEC[19])

    #Grid size for price q1
    par=Pars(par,q1_N_GLz=convert(Int64,VEC[20]))
    par=Pars(par,q1_Nz=convert(Int64,VEC[21]))
    par=Pars(par,q1_Np=convert(Int64,VEC[22]))
    par=Pars(par,q1_NkN=convert(Int64,VEC[23]))
    par=Pars(par,q1_NkT=convert(Int64,VEC[24]))
    par=Pars(par,q1_Nb=convert(Int64,VEC[25]))

    #Alternative Arellano cost
    if VEC[26]==1.0
        par=Pars(par,Arellano=true)
    else
        par=Pars(par,Arellano=false)
    end
    par=Pars(par,knk=VEC[27])

    #More parameters
    par=Pars(par,γ=VEC[28])

    return par
end

function Setup_From_Vector(VEC::Array{Float64})
    #Vector has the correct structure
    par=UnpackParameters_Vector(VEC)
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

function UnpackModel_Vector(NAME::String,FOLDER::String)
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end

    #Extract parameters and create grids
    N_parameters=convert(Int64,VEC[1])
    VEC_PAR=1.0*VEC[2:N_parameters+1]
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    #Extract solution object
    VEC_SOL=VEC[N_parameters+2:end]
    SOL=TransformVectorToSolution(VEC_SOL,GRIDS,par)

    return Model(SOL,GRIDS,par)
end

################################################################
########## Functions to compute expectations ###################
################################################################
function Expectation_over_shocks(foo,p_ind::Int64,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z'
    #kN', kT', and b' are given
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack PI_pC, GR_p = GRIDS
    int=0.0
    for i in 1:length(GR_p)
        for j in 1:length(ϵz_weights)
            int=int+PI_pC[p_ind,i]*ϵz_weights[j]*PDFz[j]*foo(ZPRIME[z_ind,j],i)
        end
    end
    return int/FacQz
end

function ComputeExpectationOverStates!(IsDefault::Bool,MAT::Array{Float64},E_MAT::SharedArray{Float64},GRIDS::Grids)
    #It will use MAT to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    if IsDefault
        @sync @distributed for I in CartesianIndices(MAT)
            (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[kT_ind,kN_ind,:,:],GRIDS)
            E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
        end
    else
        @sync @distributed for I in CartesianIndices(MAT)
            (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[b_ind,kT_ind,kN_ind,:,:],GRIDS)
            E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
        end
    end
    return nothing
end

function ValueBeforeDefault(pprime_ind::Int64,zprime::Float64,kNprime::Float64,
                            kTprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack itp_VP, itp_VD = SOLUTION
    vd=itp_VD(kTprime,kNprime,zprime,pprime_ind)
    vp=itp_VP(bprime,kTprime,kNprime,zprime,pprime_ind)
    return min(0.0,max(vd,vp))
end

function Integrate_ValueBeforeDefault(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_kN, GR_kT, GR_b = GRIDS
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    kNprime=GR_kN[kN_ind]; kTprime=GR_kT[kT_ind]; bprime=GR_b[b_ind]

    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack PI_pC, GR_p = GRIDS
    foo_V(pprime_ind::Int64,zprime::Float64)=ValueBeforeDefault(pprime_ind,zprime,kNprime,kTprime,bprime,SOLUTION,par)
    int=0.0
    for i in 1:length(GR_p)
        for j in 1:length(ϵz_weights)
            int=int+PI_pC[p_ind,i]*ϵz_weights[j]*PDFz[j]*foo_V(i,ZPRIME[z_ind,j])
        end
    end
    return int/FacQz
end

function Update_EV_OverStates!(SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to compute expectation over z' and p'
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sV)
        SHARED_AUX.sV[I]=Integrate_ValueBeforeDefault(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.EV .= SHARED_AUX.sV
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

################################################################
################ Preference functions ##########################
################################################################
function Utility(c::Float64,par::Pars)
    @unpack σ = par
    return (c^(1.0-σ))/(1.0-σ)
end

function U_c(c::Float64,par::Pars)
    @unpack σ = par
    return c^(-σ)
end

function zDefault(z::Float64,par::Pars)
    if par.Arellano
        @unpack knk = par
        return min(z,knk)
    else
        @unpack d0, d1 = par
        return z-max(0.0,d0*z+d1*z*z)
    end
end

function CapitalAdjustment(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return 0.5*φ*((kprime-k)^2)/k
    # return 0.5*φ*((kprime-k)^2)
end

function dΨ_dkprime(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return φ*(kprime-k)/k
    # return φ*(kprime-k)
end

function dΨ_dk(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return -φ*((kprime-k)/k)-0.5*φ*((kprime-k)^2)/(k^2)
    # return -φ*(kprime-k)
end

function SDF_Lenders(par::Pars)
    @unpack r_star = par
    return 1/(1+r_star)
end

################################################################
################ Production functions ##########################
################################################################
function NonTradedProduction(z::Float64,kN::Float64,par::Pars)
    @unpack αN = par
    return z*(kN^αN)
end

function TradedProduction(z::Float64,kT::Float64,par::Pars)
    @unpack αT = par
    return z*(kT^αT)
end

function Final_CES(cN::Float64,cT::Float64,cC::Float64,par::Pars)
    @unpack η, ωN, ωT, ωC = par
    if ωC == 0.0
        return ((ωN^(1.0/η))*(cN^((η-1.0)/η))+(ωT^(1.0/η))*(cT^((η-1.0)/η)))^(η/(η-1.0))
    else
        return ((ωN^(1.0/η))*(cN^((η-1.0)/η))+(ωT^(1.0/η))*(cT^((η-1.0)/η))+(ωC^(1.0/η))*(cC^((η-1.0)/η)))^(η/(η-1.0))
    end
end

function FinalOutput(z::Float64,p::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack yC, ωC, ωT, η = par
    #Total tradable disposable income
    W=TradedProduction(z,kT,par)+p*yC+T
    if W>0.0
        #Compute consumption of intermediate goods
        cN=NonTradedProduction(z,kN,par)
        #Compute cT first because ωT is always positive
        cT=(ωT/(ωT+ωC*(p^(1-η))))*W
        cC=(ωC/ωT)*(p^(-η))*cT
        #At this point we know W>0 so all cN>0, cT>0 and cC≧0
        #If cC=0 it is because ωC=0 and Final_CES will deal with it
        return Final_CES(cN,cT,cC,par)
    else
        return W
    end
end

################################################################
################### Setup functions ############################
################################################################
function PriceNonTraded(z::Float64,p::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack ωN, ωT, ωC, η, yC = par
    #Total tradable disposable income
    W=TradedProduction(z,kT,par)+p*yC+T
    if W>0.0
        #Compute consumption of intermediate goods
        #At this point we know W>0 so all cN>0 and cT>0
        cN=NonTradedProduction(z,kN,par)
        cT=(ωT/(ωT+ωC*(p^(1-η))))*W
        return ((ωN/ωT)*(cT/cN))^(1.0/η)
    else
        return 1e-2
    end
end

function PriceFinalGood(z::Float64,p::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack ωN, ωT, ωC, η = par
    pN=PriceNonTraded(z,p,kN,kT,T,par)
    if ωC>0.0
        return (ωN*(pN^(1.0-η))+ωT*(1.0^(1.0-η))+ωC*(p^(1.0-η)))^(1.0/(1.0-η))
    else
        return (ωN*(pN^(1.0-η))+ωT*(1.0^(1.0-η)))^(1.0/(1.0-η))
    end
end

function rN_dec(z::Float64,p::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack αN = par
    pN=PriceNonTraded(z,p,kN,kT,T,par)
    return αN*pN*z*(kN^(αN-1.0))
end

function rT_dec(z::Float64,kT::Float64,par::Pars)
    @unpack αT = par
    return αT*z*(kT^(αT-1.0))
end

###############################################################################
#Function to compute consumption net of investment and adjustment cost
###############################################################################
function ConsNet(y::Float64,kN::Float64,kT::Float64,kNprime::Float64,kTprime::Float64,par::Pars)
    #Resource constraint for the final good
    @unpack δ = par
    invN=kNprime-(1-δ)*kN
    invT=kTprime-(1-δ)*kT
    AdjN=CapitalAdjustment(kNprime,kN,par)
    AdjT=CapitalAdjustment(kTprime,kT,par)
    return y-invN-invT-AdjN-AdjT
end

function ComputeSpreadWithQ(qq::Float64,par::Pars)
    @unpack r_star, γ, κ = par
    ib=(((γ+(1-γ)*(κ+qq))/qq)^4)-1
    # ib=(((1/qq)+1-γ)^4)-1
    rf=((1+r_star)^4)-1
    return ib-rf
end

function SimpleSubsidy(qq::Float64,par::Pars)
    @unpack τsN0, τsT0, τsN1, τsT1 = par
    #Compute subsidies proportional to spreads
    spread=ComputeSpreadWithQ(qq,par)
    τsN=τsN0+τsN1*spread
    τsT=τsT0+τsT1*spread
    return τsN, τsT
end

function Calculate_Covenant_b(itp_q1,b::Float64,bprime::Float64,kNprime::Float64,kTprime::Float64,z::Float64,p_ind::Int64,par::Pars)
    @unpack γ, Covenants = par
    if Covenants
        #Compute compensation per remaining bond (1-γ)b for additional borrowing
        #(in this case there is no compensation for kN' and kT')
        qq0=itp_q1((1-γ)*b,kTprime,kNprime,z,p_ind)
        qq1=itp_q1(bprime,kTprime,kNprime,z,p_ind)
        return max(0.0,qq0-qq1)
    else
        return 0.0
    end
end

function Calculate_Tr(p_ind::Int64,z::Float64,b::Float64,kNprime::Float64,kTprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack γ, κ = par
    @unpack itp_q1 = SOLUTION
    #Compute net borrowing from the rest of the world
    Covenant=Calculate_Covenant_b(itp_q1,b,bprime,kNprime,kTprime,z,p_ind,par)
    qq=itp_q1(bprime,kTprime,kNprime,z,p_ind)
    return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b-(1-γ)*b*Covenant
end

###############################################################################
#Functions to compute value and consumption given state, policies, and guesses
###############################################################################
function Evaluate_cons_state(Default::Bool,I::CartesianIndex,kNprime::Float64,
                             kTprime::Float64,bprime::Float64,
                             SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    if Default
        (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)

        z=GR_z[z_ind]; p=GR_p[p_ind]
        kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]
        zD=zDefault(z,par)
        T=0.0
        y=FinalOutput(zD,p,kN,kT,T,par)

        return ConsNet(y,kN,kT,kNprime,kTprime,par)
    else
        @unpack GR_b = GRIDS
        @unpack itp_q1 = SOLUTION
        (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)

        z=GR_z[z_ind]; p=GR_p[p_ind]
        kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]
        T=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
        y=FinalOutput(z,p,kN,kT,T,par)

        return ConsNet(y,kN,kT,kNprime,kTprime,par)
    end
end

function Evaluate_VD(I::CartesianIndex,kNprime::Float64,kTprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, cmin = par
    @unpack itp_EV, itp_EVD = SOLUTION
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GRIDS.GR_z[z_ind]
    cons=Evaluate_cons_state(true,I,kNprime,kTprime,0.0,SOLUTION,GRIDS,par)
    if cons>cmin
        return Utility(cons,par)+β*θ*itp_EV(0.0,kTprime,kNprime,z,p_ind)+β*(1.0-θ)*itp_EVD(kTprime,kNprime,z,p_ind)
    else
        return Utility(cmin,par)+cons
    end
end

function ValueInDefault_PLA(I::CartesianIndex,X_REAL::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt = par
    #transform policy tries into interval
    kNprime=TransformIntoBounds(X_REAL[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(X_REAL[2],kTlowOpt,kThighOpt)
    return Evaluate_VD(I,kNprime,kTprime,SOLUTION,GRIDS,par)
end

function Evaluate_VP(vDmin::Float64,I::CartesianIndex,kNprime::Float64,kTprime::Float64,bprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_kN, GR_kT, GR_b, GR_z, GR_p = GRIDS

    #Unpack state
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]

    #Compute price
    T=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
    y=FinalOutput(z,p,kN,kT,T,par)
    if y>0.0
        #Compute consumption
        cons=ConsNet(y,kN,kT,kNprime,kTprime,par)
        if cons>0.0
            qq=itp_q1(bprime,kTprime,kNprime,z,p_ind)
            if bprime>0.0 && qq==0.0
                #Small penalty for larger debt positions
                #wrong side of laffer curve, it is decreasing
                return Utility(cons,par)+β*itp_EV(bprime,kTprime,kNprime,z,p_ind)-abs(bprime)*sqrt(eps(Float64))
            else
                return Utility(cons,par)+β*itp_EV(bprime,kTprime,kNprime,z,p_ind)
            end
        else
            return Utility(cmin,par)+cons
        end
    else
        return Utility(cmin,par)+y
    end
end

function ValueInRepayment_PLA(vDmin::Float64,I::CartesianIndex,X_REAL::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, blowOpt ,bhighOpt = par
    #transform policy tries into interval
    kNprime=TransformIntoBounds(X_REAL[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(X_REAL[2],kTlowOpt,kThighOpt)
    bprime=TransformIntoBounds(X_REAL[3],blowOpt,bhighOpt)
    vv=Evaluate_VP(vDmin,I,kNprime,kTprime,bprime,SOLUTION,GRIDS,par)
    return vv
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################
function InitialPolicyGuess_D(Kpol_1::Bool,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    if Kpol_1
        #Choose last policy as initial guess
        if SOLUTION.kNprime_D[I]==0.0
            #It is the first attempt, use current kN
            kNP=GRIDS.GR_kN[kN_ind]
        else
            #Keep it within the grids
            if SOLUTION.kNprime_D[I]<GRIDS.GR_kN[end]
                kNP=max(SOLUTION.kNprime_D[I],GRIDS.GR_kN[1])
            else
                kNP=min(SOLUTION.kNprime_D[I],GRIDS.GR_kN[end])
            end
        end

        if SOLUTION.kTprime_D[I]==0.0
            #It is the first attempt, use current kT
            kTP=GRIDS.GR_kT[kT_ind]
        else
            #Keep it within the grids
            if SOLUTION.kTprime_D[I]<GRIDS.GR_kT[end]
                kTP=max(SOLUTION.kTprime_D[I],GRIDS.GR_kT[1])
            else
                kTP=min(SOLUTION.kTprime_D[I],GRIDS.GR_kT[end])
            end
        end
    else
        #Choose present k as initial guess (stay the same)
        kNP=GRIDS.GR_kN[kN_ind]
        kTP=GRIDS.GR_kT[kT_ind]
    end

    #Check if guess gives positive consumption
    cons=Evaluate_cons_state(true,I,kNP,kTP,0.0,SOLUTION,GRIDS,par)
    if cons>0.0
        return [kNP, kTP]
    else
        #Choose present k as initial guess (stay the same)
        kNP=GRIDS.GR_kN[kN_ind]
        kTP=GRIDS.GR_kT[kT_ind]
        return [kNP, kTP]
    end
end

function OptimInDefault(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt = par
    # @unpack β, θ = par
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    #transform policy guess into reals
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)

    Kpol_1=true
    X0_BOUNDS=InitialPolicyGuess_D(Kpol_1,I,SOLUTION,GRIDS,par)
    #transform policy guess into reals
    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kTlowOpt,kThighOpt)

    #Setup function handle for optimization
    f(X::Array{Float64,1})=-ValueInDefault_PLA(I,X,SOLUTION,GRIDS,par)

    #Perform optimization with MatLab simplex
    inner_optimizer = NelderMead()
    OPTIONS=Optim.Options(g_tol = 1e-13)
    res=optimize(f,X0,inner_optimizer,OPTIONS)

    #Transform optimizer into bounds
    kNprime=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)

    return -Optim.minimum(res), kNprime, kTprime
end

function InitialPolicyGuess_P(vDmin::Float64,Kpol_1::Bool,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_kN, GR_kT, GR_b, GR_z, GR_p = GRIDS
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    if Kpol_1
        #Choose last policy as initial guess for capital
        if SOLUTION.kNprime[I]==0.0
            #It is the first attempt, use current kN
            kNP=GRIDS.GR_kN[kN_ind]
        else
            if SOLUTION.kNprime[I]<GR_kN[end]
                kNP=max(SOLUTION.kNprime[I],GR_kN[1])
            else
                kNP=min(SOLUTION.kNprime[I],GR_kN[end])
            end
        end

        if SOLUTION.kTprime[I]==0.0
            #It is the first attempt, use current kT
            kTP=GRIDS.GR_kT[kT_ind]
        else
            if SOLUTION.kTprime[I]<GR_kT[end]
                kTP=max(SOLUTION.kTprime[I],GR_kT[1])
            else
                kTP=min(SOLUTION.kTprime[I],GR_kT[end])
            end
        end

        if SOLUTION.bprime[I]<GR_b[end]
            b0=max(SOLUTION.bprime[I],GR_b[1])
        else
            b0=min(SOLUTION.bprime[I],GR_b[end])
        end
    else
        #Choose present k as initial guess (stay the same)
        kNP=GRIDS.GR_kN[kN_ind]
        kTP=GRIDS.GR_kT[kT_ind]
        b0=GR_b[b_ind]
    end

    return [kNP, kTP, b0]
end

function OptimInRepayment(vDmin::Float64,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, Nb = par
    @unpack kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, blowOpt, bhighOpt = par
    @unpack GR_kN, GR_kT, GR_b, GR_z, GR_p = GRIDS
    #Get initial guess
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)

    Kpol_1=true
    X0_BOUNDS=InitialPolicyGuess_P(vDmin,Kpol_1,I,SOLUTION,GRIDS,par)

    #transform policy guess into reals
    X0=Array{Float64,1}(undef,3)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kTlowOpt,kThighOpt)
    X0[3]=TransformIntoReals(X0_BOUNDS[3],blowOpt,bhighOpt)

    #Setup function handle for optimization
    f(X::Array{Float64,1})=-ValueInRepayment_PLA(vDmin,I,X,SOLUTION,GRIDS,par)

    #Perform optimization with MatLab simplex
    inner_optimizer = NelderMead()
    OPTIONS=Optim.Options(g_tol = 1e-13)
    res=optimize(f,X0,inner_optimizer,OPTIONS)


    #Transform optimizer into bounds
    kNprime=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)
    bprime=TransformIntoBounds(Optim.minimizer(res)[3],blowOpt,bhighOpt)

    #Check if optimizer yields negative consumption or negative output
    z=GR_z[z_ind]; p=GR_p[p_ind]; kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]
    Tr=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
    y=FinalOutput(z,p,kN,kT,Tr,par)
    cons=ConsNet(y,kN,kT,kNprime,kTprime,par)
    if y>0.0 && cons>0.0
        #Solution works, return as is
        return -Optim.minimum(res), kNprime, kTprime, bprime, Tr
    else
        #Negative output or consumption, return VD-ϵ
        vv=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]-sqrt(eps(Float64))
        #Make policies stay flat in b
        if SOLUTION.kNprime[max(1,b_ind-1),kT_ind,kN_ind,z_ind,p_ind]==0.0
            #First iteration, use default policies
            kNprime=SOLUTION.kNprime_D[kT_ind,kN_ind,z_ind,p_ind]
            kTprime=SOLUTION.kTprime_D[kT_ind,kN_ind,z_ind,p_ind]
        else
            #Make policies stay flat in b
            kNprime=SOLUTION.kNprime[max(1,b_ind-1),kT_ind,kN_ind,z_ind,p_ind]
            kTprime=SOLUTION.kTprime[max(1,b_ind-1),kT_ind,kN_ind,z_ind,p_ind]
        end
        bprime=SOLUTION.bprime[max(1,b_ind-1),kT_ind,kN_ind,z_ind,p_ind]
        Tr=0.0
        return vv, kNprime, kTprime, bprime, Tr
    end
end

###############################################################################
#Update solution
###############################################################################
function UpdateDefault!(SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack yC = par
    #Loop over all states to fill array of VD
    @sync @distributed for I in CartesianIndices(SOLUTION.VD)
        if yC==0.0
            #No commodities, solution at p_ind=1 is the same as p_ind=2
            #only solve if p_ind=1 and then allocate to p_ind==2
            (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
            if p_ind==1
                SHARED_AUX.sVD[I], SHARED_AUX.skNprime_D[I], SHARED_AUX.skTprime_D[I]=OptimInDefault(I,SOLUTION,GRIDS,par)
                SHARED_AUX.sVD[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sVD[I]
                SHARED_AUX.skNprime_D[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skNprime_D[I]
                SHARED_AUX.skTprime_D[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skTprime_D[I]
            end
        else
            SHARED_AUX.sVD[I], SHARED_AUX.skNprime_D[I], SHARED_AUX.skTprime_D[I]=OptimInDefault(I,SOLUTION,GRIDS,par)
        end
    end
    SOLUTION.VD .= SHARED_AUX.sVD
    SOLUTION.kNprime_D .= SHARED_AUX.skNprime_D
    SOLUTION.kTprime_D .= SHARED_AUX.skTprime_D

    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kNprime_D=CreateInterpolation_Policies(SOLUTION.kNprime_D,true,GRIDS)
    SOLUTION.itp_kTprime_D=CreateInterpolation_Policies(SOLUTION.kTprime_D,true,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(true,SOLUTION.VD,SHARED_AUX.sVD,GRIDS)
    SOLUTION.EVD .= SHARED_AUX.sVD
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function RepaymentUpdater_PLA!(vDmin::Float64,I::CartesianIndex,SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    if par.yC==0.0
        #No commodities, solution at p_ind=1 is the same as p_ind=2
        #only solve if p_ind=1 and then allocate to p_ind==2
        if p_ind==1
            SHARED_AUX.sVP[I], SHARED_AUX.skNprime[I], SHARED_AUX.skTprime[I], SHARED_AUX.sbprime[I], SHARED_AUX.sTr[I]=OptimInRepayment(vDmin,I,SOLUTION,GRIDS,par)
            if SHARED_AUX.sVP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
                SHARED_AUX.sV[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
            else
                SHARED_AUX.sV[I]=SHARED_AUX.sVP[I]
            end
            SHARED_AUX.sVP[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sVP[I]
            SHARED_AUX.skNprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skNprime[I]
            SHARED_AUX.skTprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skTprime[I]
            SHARED_AUX.sbprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sbprime[I]
            SHARED_AUX.sTr[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sTr[I]
            SHARED_AUX.sV[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sV[I]
        end
    else
        SHARED_AUX.sVP[I], SHARED_AUX.skNprime[I], SHARED_AUX.skTprime[I], SHARED_AUX.sbprime[I], SHARED_AUX.sTr[I]=OptimInRepayment(vDmin,I,SOLUTION,GRIDS,par)
        if SHARED_AUX.sVP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
            SHARED_AUX.sV[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
        else
            SHARED_AUX.sV[I]=SHARED_AUX.sVP[I]
        end
    end

    return nothing
end

function UpdateRepayment!(SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Minimum value in default to aide with
    #numerical optimization dealing with negative consumption
    vDmin=minimum(SOLUTION.VD)
    #Loop over all states to fill value of repayment
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sVP)
        RepaymentUpdater_PLA!(vDmin,I,SHARED_AUX,SOLUTION,GRIDS,par)
    end
    SOLUTION.VP .= SHARED_AUX.sVP
    SOLUTION.V .= SHARED_AUX.sV
    SOLUTION.kNprime .= SHARED_AUX.skNprime
    SOLUTION.kTprime .= SHARED_AUX.skTprime
    SOLUTION.bprime .= SHARED_AUX.sbprime
    SOLUTION.Tr .= SHARED_AUX.sTr

    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kNprime=CreateInterpolation_Policies(SOLUTION.kNprime,false,GRIDS)
    SOLUTION.itp_kTprime=CreateInterpolation_Policies(SOLUTION.kTprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)

    #Compute expectation
    Update_EV_OverStates!(SHARED_AUX,SOLUTION,GRIDS,par)
    return nothing
end

###############################################################################
#Update price
###############################################################################
function BondsPayoff(pprime_ind::Int64,zprime::Float64,kNprime::Float64,
                     kTprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack itp_VP, itp_VD = SOLUTION
    if min(0.0,itp_VD(kTprime,kNprime,zprime,pprime_ind))>min(0.0,itp_VP(bprime,kTprime,kNprime,zprime,pprime_ind))
        return 0.0
    else
        @unpack γ, κ = par
        SDF=SDF_Lenders(par)
        if γ==1.0
            return SDF
        else
            @unpack itp_q1, itp_bprime, itp_kNprime, itp_kTprime = SOLUTION
            kNkN=itp_kNprime(bprime,kTprime,kNprime,zprime,pprime_ind)
            kTkT=itp_kTprime(bprime,kTprime,kNprime,zprime,pprime_ind)
            bb=itp_bprime(bprime,kTprime,kNprime,zprime,pprime_ind)
            qq=itp_q1(bb,kTkT,kNkN,zprime,pprime_ind)
            Covenant=Calculate_Covenant_b(itp_q1,bprime,bb,kNkN,kTkT,zprime,pprime_ind,par)
            return SDF*(γ+(1-γ)*(κ+qq+Covenant))
            # return SDF*(1+(1-γ)*qq)
        end
    end
end

function IntegrateBondsPayoff(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GRIDS_Q = GRIDS
    @unpack GR_kN, GR_kT, GR_b = GRIDS_Q
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    kNprime=GR_kN[kN_ind]; kTprime=GR_kT[kT_ind]; bprime=GR_b[b_ind]

    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS_Q
    @unpack PI_pC, GR_p = GRIDS
    foo_q(pprime_ind::Int64,zprime::Float64)=BondsPayoff(pprime_ind,zprime,kNprime,kTprime,bprime,SOLUTION,par)
    int=0.0
    for i in 1:length(GR_p)
        for j in 1:length(ϵz_weights)
            int=int+PI_pC[p_ind,i]*ϵz_weights[j]*PDFz[j]*foo_q(i,ZPRIME[z_ind,j])
        end
    end
    return int/FacQz
end

function UpdateBondsPrice!(SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack yC = par
    #Loop over all states to compute expectation over z' and p'
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sq1)
        if yC==0.0
            #No commodities, solution at p_ind=1 is the same as p_ind=2
            #only solve if p_ind=1 and then allocate to p_ind==2
            (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
            if p_ind==1
                SHARED_AUX.sq1[I]=IntegrateBondsPayoff(I,SOLUTION,GRIDS,par)
                SHARED_AUX.sq1[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sq1[I]
            end
        else
            SHARED_AUX.sq1[I]=IntegrateBondsPayoff(I,SOLUTION,GRIDS,par)
        end
    end
    SOLUTION.q1 .= SHARED_AUX.sq1
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    return nothing
end

###############################################################################
#Update and iniciate VFI algorithm
###############################################################################
function UpdateSolution_PLA!(SHARED_AUX::SharedAux,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(SHARED_AUX,SOLUTION,GRIDS,par)
    UpdateRepayment!(SHARED_AUX,SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SHARED_AUX,SOLUTION,GRIDS,par)
    return nothing
end

function ComputeDistance_q(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    @unpack Tol_q = par
    dst_q, Ix=findmax(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1))
    NotConv=sum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.q1)
    return round(dst_q,digits=7), round(NotConvPct,digits=2), Ix
end

function ComputeDistanceV(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    dst_D=maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD))
    dst_V, Iv=findmax(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V))

    NotConv=sum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V) .> par.Tol_V)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.V)
    return round(abs(dst_D),digits=7), round(abs(dst_V),digits=7), Iv, round(NotConvPct,digits=2)
end

function SolveModel_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    SHARED_AUX=InitiateSharedAux(GRIDS,par)
    dst_V=1.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        UpdateSolution_PLA!(SHARED_AUX,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
    end
    return SOLUTION_NEXT
end

function SolveAndSaveModel(NAME::String,GRIDS::Grids,par::Pars)
    SOL=SolveModel_VFI(true,GRIDS,par)
    SaveModel_Vector(NAME,SOL,par)
    return nothing
end

function SolveClosed_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    parClosed=Pars(par,θ=0.0,d0=0.0,d1=0.0)
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,parClosed)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    SHARED_AUX=InitiateSharedAux(GRIDS,par)
    dst_V=1.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while (dst_V>Tol_V || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateDefault!(SHARED_AUX,SOLUTION_NEXT,GRIDS,parClosed)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,parClosed)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
    end
    return SOLUTION_NEXT
end

function ModelClosed_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    SOLUTION=SolveClosed_VFI(PrintProg,GRIDS,par)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    parClosed=Pars(par,θ=0.0,d0=0.0,d1=0.0)
    return Model(SOLUTION,GRIDS,parClosed)
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct Paths
    #Paths of shocks
    z::Array{Float64,1}
    p_ind::Array{Int64,1}
    p::Array{Float64,1}

    #Paths of chosen states
    Def::Array{Float64,1}
    KN::Array{Float64,1}
    KT::Array{Float64,1}
    B::Array{Float64,1}

    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    nGDP::Array{Float64,1}
    P::Array{Float64,1}
    GDPdef::Array{Float64,1}
    Cons::Array{Float64,1}
    Inv::Array{Float64,1}
    InvN::Array{Float64,1}
    InvT::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
    RER::Array{Float64,1}
end

function InitiateEmptyPaths(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    f1=zeros(Float64,T)
    f2=zeros(Int64,T)
    f3=zeros(Float64,T)
    f4=zeros(Float64,T)
    f5=zeros(Float64,T)
    f6=zeros(Float64,T)
    f7=zeros(Float64,T)
    f8=zeros(Float64,T)
    f9=zeros(Float64,T)
    f10=zeros(Float64,T)
    f11=zeros(Float64,T)
    f12=zeros(Float64,T)
    f13=zeros(Float64,T)
    f14=zeros(Float64,T)
    f15=zeros(Float64,T)
    f16=zeros(Float64,T)
    f17=zeros(Float64,T)
    f18=zeros(Float64,T)
    f19=zeros(Float64,T)
    return Paths(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19)
end

function Simulate_z_shocks(z0::Float64,T::Int64,GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    ϵz_TS=rand(dist_ϵz,T)
    z_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            z_TS[t]=z0
        else
            z_TS[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_TS[t-1])+ϵz_TS[t])
        end
    end
    return z_TS
end

function Get_CDF(PDF::Array{Float64,1})
    N,=size(PDF)
    CDF=zeros(Float64,N)
    CDF[1]=PDF[1]
    for i in 2:N
        CDF[i]=CDF[i-1]+PDF[i]
    end
    return CDF
end

function Draw_New_pC(p_ind::Int64,PI_pC::Array{Float64,2})
    PDF=PI_pC[p_ind,:]
    CDF=Get_CDF(PDF)
    x=rand()
    p_prime=0
    for i in 1:length(CDF)
        if x<=CDF[i]
            p_prime=i
            break
        end
    end
    return p_prime
end

function Simulate_pC_shocks(pC0_ind::Int64,T::Int64,PI_pC::Array{Float64,2})
    X=Array{Int64,1}(undef,T)
    X[1]=pC0_ind
    for t in 2:T
        X[t]=Draw_New_pC(X[t-1],PI_pC)
    end
    return X
end

function Simulate_AllShocks(T::Int64,GRIDS::Grids,par::Pars)
    z=Simulate_z_shocks(par.μ_z,T,GRIDS,par)
    p_ind=Simulate_pC_shocks(1,T,GRIDS.PI_pC)
    p=Array{Float64,1}(undef,T)
    for t in 1:T
        p[t]=GRIDS.GR_p[p_ind[t]]
    end
    return z, p_ind, p
end

function ComputeSpreads(z::Float64,p_ind::Int64,kNprime::Float64,
                        kTprime::Float64,bprime::Float64,
                        SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_q1 = SOLUTION
    q=max(itp_q1(bprime,kTprime,kNprime,z,p_ind),1e-2)
    return 100.0*ComputeSpreadWithQ(q,par)
end

function GenerateNextState!(t::Int64,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_VD, itp_VP, itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
    @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
    @unpack θ = par
    #It returns for t+1>=2, initial state defined outside of this function

    #Unpack current shocks
    z=PATHS.z[t]
    p=PATHS.p[t]
    p_ind=PATHS.p_ind[t]

    #Unpack current endogenous state and previous default
    if t==1
        d_=PATHS.Def[t]
    else
        d_=PATHS.Def[t-1]
    end
    kN=PATHS.KN[t]; kT=PATHS.KT[t]; b=PATHS.B[t]

    #Update next endogenous state
    if d_==1.0
        #Coming from default state yesterday, must draw for readmission
        if rand()<=θ
            #Get readmitted, choose whether to default or not today
            if itp_VD(kT,kN,z,p_ind)<=itp_VP(b,kT,kN,z,p_ind)
                #Choose not to default
                PATHS.Def[t]=0.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill kN', kT' and b', and spreads
                    PATHS.KN[t+1]=itp_kNprime(b,kT,kN,z,p_ind)
                    PATHS.KT[t+1]=itp_kTprime(b,kT,kN,z,p_ind)
                    PATHS.B[t+1]=max(itp_bprime(b,kT,kN,z,p_ind),0.0)
                    PATHS.Spreads[t]=ComputeSpreads(z,p_ind,PATHS.KN[t+1],PATHS.KT[t+1],PATHS.B[t+1],SOLUTION,GRIDS,par)
                else
                    #Only Fill spreads today
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            else
                #Choose to default
                PATHS.Def[t]=1.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill kN', kT' and b', and spreads
                    PATHS.KN[t+1]=itp_kNprime_D(kT,kN,z,p_ind)
                    PATHS.KT[t+1]=itp_kTprime_D(kT,kN,z,p_ind)
                    PATHS.B[t+1]=0.0
                    if t==1
                        PATHS.Spreads[t]=0.0
                    else
                        PATHS.Spreads[t]=PATHS.Spreads[t-1]
                    end
                else
                    #Only Fill spreads today
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        else
            #Stay in default
            PATHS.Def[t]=1.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kN', kT' and b'
                PATHS.KN[t+1]=itp_kNprime_D(kT,kN,z,p_ind)
                PATHS.KT[t+1]=itp_kTprime_D(kT,kN,z,p_ind)
                PATHS.B[t+1]=0.0
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            else
                #Fix spreads
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        end
    else
        #Coming from repayment, choose whether to default or not today
        if itp_VD(kT,kN,z,p_ind)<=itp_VP(b,kT,kN,z,p_ind)
            #Choose not to default
            PATHS.Def[t]=0.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kN', kT' and b', and spreads
                PATHS.KN[t+1]=itp_kNprime(b,kT,kN,z,p_ind)
                PATHS.KT[t+1]=itp_kTprime(b,kT,kN,z,p_ind)
                PATHS.B[t+1]=max(itp_bprime(b,kT,kN,z,p_ind),0.0)
                PATHS.Spreads[t]=ComputeSpreads(z,p_ind,PATHS.KN[t+1],PATHS.KT[t+1],PATHS.B[t+1],SOLUTION,GRIDS,par)
            else
                #Fix spreads
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        else
            #Choose to default
            PATHS.Def[t]=1.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kN', kT' and b'
                PATHS.KN[t+1]=itp_kNprime_D(kT,kN,z,p_ind)
                PATHS.KT[t+1]=itp_kTprime_D(kT,kN,z,p_ind)
                PATHS.B[t+1]=0.0
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            else
                #Fix spreads
                PATHS.Spreads[t]=PATHS.Spreads[t-1]
            end
        end
    end
    return nothing
end

function TimeIntervalFromPaths(PATHS::Paths,t0::Int64,tT::Int64)
    return Paths(PATHS.z[t0:tT],PATHS.p_ind[t0:tT],PATHS.p[t0:tT],PATHS.Def[t0:tT],PATHS.KN[t0:tT],PATHS.KT[t0:tT],PATHS.B[t0:tT],PATHS.Spreads[t0:tT],PATHS.GDP[t0:tT],PATHS.nGDP[t0:tT],PATHS.P[t0:tT],PATHS.GDPdef[t0:tT],PATHS.Cons[t0:tT],PATHS.Inv[t0:tT],PATHS.InvN[t0:tT],PATHS.InvT[t0:tT],PATHS.TB[t0:tT],PATHS.CA[t0:tT],PATHS.RER[t0:tT])
end

function SimulatePathsOfStates(drp::Int64,z_TS::Array{Float64,1},p_ind_TS::Array{Int64,1},p_TS::Array{Float64,1},X0::Array{Float64,1},MODEL::Model)
    #Given a path of shocks
    #Given initial conditions X0=(KN0,KT0,B0)

    TP=length(z_TS)
    T=TP-drp
    PATHS=InitiateEmptyPaths(drp+T)
    #Store shocks
    PATHS.z .= z_TS
    PATHS.p_ind .= p_ind_TS
    PATHS.p .= p_TS

    #Choose initial conditions
    PATHS.Def[1]=X0[1]
    PATHS.KN[1]=X0[2]
    PATHS.KT[1]=X0[3]
    PATHS.B[1]=X0[4]
    for t in 1:drp+T
        GenerateNextState!(t,PATHS,MODEL.SOLUTION,MODEL.GRIDS,MODEL.par)
    end
    return TimeIntervalFromPaths(PATHS,drp+1,drp+T)
end

function CalculateVariables_TS!(t::Int64,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack yC, δ = par

    #Unpack current shocks
    zstate=PATHS.z[t]
    p_ind=PATHS.p_ind[t]
    p=PATHS.p[t]

    #Unpack current endogenous state
    d=PATHS.Def[t]; kN=PATHS.KN[t]; kT=PATHS.KT[t]; b=PATHS.B[t]
    #z shock to evaluate
    if d==1.0
        z=zDefault(zstate,par)
    else
        z=zstate
    end

    #Compute policies
    if t==length(PATHS.z)
        #end of time, use policy functions for next state
        if d==1
            #in default
            @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
            kNprime=itp_kNprime_D(kT,kN,zstate,p_ind)
            kTprime=itp_kTprime_D(kT,kN,zstate,p_ind)
            bprime=0.0
            Tr=0.0
        else
            #in repayment
            @unpack itp_kNprime, itp_kTprime, itp_bprime, itp_q1 = SOLUTION
            kNprime=itp_kNprime(b,kT,kN,zstate,p_ind)
            kTprime=itp_kTprime(b,kT,kN,zstate,p_ind)
            bprime=itp_bprime(b,kT,kN,zstate,p_ind)
            Tr=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
        end
    else
        #Not end of time, use next state
        kNprime=PATHS.KN[t+1]
        kTprime=PATHS.KT[t+1]
        bprime=PATHS.B[t+1]
        if d==1
            #in default
            Tr=0.0
        else
            #in repayment
            Tr=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
        end
    end

    #Store base prices
    if t==1
        #Use today's prices as base
        p0=p
        P0=PriceFinalGood(z,p,kN,kT,Tr,par)
        PATHS.P[t]=P0
    else
        p0=PATHS.p[1]
        P0=PATHS.P[1]
    end

    #Calculate current nominal and real variables
    Y=FinalOutput(z,p,kN,kT,Tr,par)
    P=PriceFinalGood(z,p,kN,kT,Tr,par)
    NX=-Tr
    PATHS.GDP[t]=P0*Y+NX+(p0-p)*yC
    PATHS.nGDP[t]=P*Y+NX
    PATHS.P[t]=P
    PATHS.GDPdef[t]=PATHS.nGDP[t]/PATHS.GDP[t]
    PATHS.Cons[t]=ConsNet(Y,kN,kT,kNprime,kTprime,par)
    PATHS.InvN[t]=kNprime-(1-δ)*kN
    PATHS.InvT[t]=kTprime-(1-δ)*kT
    PATHS.Inv[t]=PATHS.InvN[t]+PATHS.InvT[t]
    PATHS.TB[t]=NX
    PATHS.CA[t]=-(bprime-b)
    PATHS.RER[t]=1/PATHS.GDPdef[t]
    return nothing
end

function UpdateOtherVariablesInPaths!(PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    for t=1:length(PATHS.z)
        CalculateVariables_TS!(t,PATHS,SOLUTION,GRIDS,par)
    end
end

function SimulatePaths(Def0::Float64,T::Int64,MODEL::Model;FixSeed::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL

    if FixSeed
        Random.seed!(1234)
    end

    z_TS, p_ind_TS, p_TS=Simulate_AllShocks(par.drp+T,GRIDS,par)
    KN0=0.5*(par.kNlow+par.kNhigh)
    KT0=0.5*(par.kTlow+par.kThigh)
    B0=0.0
    X0=[Def0; KN0; KT0; B0]

    PATHS=SimulatePathsOfStates(par.drp,z_TS,p_ind_TS,p_TS,X0,MODEL)
    UpdateOtherVariablesInPaths!(PATHS,MODEL)
    return PATHS
end

########## Compute moments
@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    MeanSpreads::Float64 = 0.0
    StdSpreads::Float64 = 0.0
    #Stocks
    Debt_GDP::Float64 = 0.0
    yC_GDP::Float64 = 0.0
    K_GDP::Float64 = 0.0
    kN_GDP::Float64 = 0.0
    kT_GDP::Float64 = 0.0
    Λshare::Float64 = 0.0
    #AR(1) GDP
    ρ_GDP::Float64 = 0.0
    σϵ_GDP::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_inv::Float64 = 0.0
    σ_TB_y::Float64 = 0.0
    σ_RER::Float64 = 0.0
    #Cyclicality
    Corr_con_GDP::Float64 = 0.0
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
    Corr_RER_GDP::Float64 = 0.0
end

function hp_filter(y::Vector{Float64}, lambda::Float64)
    #Returns trend component
    n = length(y)
    @assert n >= 4

    diag2 = lambda*ones(n-2)
    diag1 = [ -2lambda; -4lambda*ones(n-3); -2lambda ]
    diag0 = [ 1+lambda; 1+5lambda; (1+6lambda)*ones(n-4); 1+5lambda; 1+lambda ]

    #D = spdiagm((diag2, diag1, diag0, diag1, diag2), (-2,-1,0,1,2))
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)

    D\y
end

function ComputeMomentsOnce(MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL

    PATHS=SimulatePaths(1.0,par.Tmom,MODEL)
    MOM=Moments()
    #Compute default probability
    PATHS_long=SimulatePaths(1.0,10000,MODEL)
    Def_Ev=0.0
    for t in 2:length(PATHS_long.Def)
        if PATHS_long.Def[t]==1.0 && PATHS_long.Def[t-1]==0.0
            Def_Ev=Def_Ev+1
        end
    end
    QuarterlyDefaultFreq=Def_Ev/(length(PATHS_long.Def)-sum(PATHS_long.Def))
    MOM.DefaultPr=100*(((1+QuarterlyDefaultFreq)^4)-1)

    #Compute other easy moments
    CountForAv=ones(Float64,length(PATHS.Spreads))
    for t in 1:length(CountForAv)-1
        if PATHS.Def[t]==1.0
            CountForAv[t]=0.0
        else
            if t<length(CountForAv) && PATHS.Def[t+1]==1.0
                CountForAv[t]=0.0
            end
        end
    end
    CountForAv.=(1 .- PATHS.Def)
    MOM.MeanSpreads=sum((PATHS.Spreads) .* CountForAv)/sum(CountForAv)
    ex2=sum((PATHS.Spreads .^ 2) .* CountForAv)/sum(CountForAv)
    MOM.StdSpreads=sqrt(abs(ex2-(MOM.MeanSpreads^2)))
    #Compute stocks
    MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (4 .* PATHS.nGDP))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    MOM.yC_GDP=mean((PATHS.p .* par.yC) ./ PATHS.nGDP)
    MOM.K_GDP=mean(PATHS.P .* (PATHS.KN .+ PATHS.KT) ./ PATHS.nGDP)
    MOM.kN_GDP=mean(PATHS.P .* PATHS.KN ./ PATHS.nGDP)
    MOM.kT_GDP=mean(PATHS.P .* PATHS.KT ./ PATHS.nGDP)
    MOM.Λshare=mean(PATHS.KT ./ (PATHS.KT .+ PATHS.KN))

    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(PATHS.GDP))
    GDP_trend=hp_filter(log_GDP,par.HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #AR(1) GDP
    xx=0.01*GDP_cyc[1:end-1]
    yy=0.01*GDP_cyc[2:end]
    MOM.ρ_GDP=(xx'*xx)\(xx'*yy)
    yyhat=MOM.ρ_GDP*xx
    epshat=yyhat-yy
    MOM.σϵ_GDP=std(epshat)
    #Consumption
    log_con=log.(abs.(PATHS.Cons))
    con_trend=hp_filter(log_con,par.HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Investment
    log_inv=log.(abs.(PATHS.Inv))
    inv_trend=hp_filter(log_inv,par.HPFilter_Par)
    inv_cyc=100.0*(log_inv .- inv_trend)
    #Trade balance
    TB_y_TS=100 * PATHS.TB ./ PATHS.nGDP
    # TB_y_TS=100 * PATHS.TB ./ Youtput
    #Real exchange rate
    log_rer=log.(abs.(PATHS.RER))
    rer_trend=hp_filter(log_rer,par.HPFilter_Par)
    rer_cyc=100.0*(log_rer .- rer_trend)
    #Volatilities
    MOM.σ_GDP=std(GDP_cyc)
    MOM.σ_con=std(con_cyc)
    MOM.σ_inv=std(inv_cyc)
    MOM.σ_TB_y=std(TB_y_TS)
    MOM.σ_RER=std(rer_cyc)
    #Correlations with GDP
    MOM.Corr_con_GDP=cor(GDP_cyc,con_cyc)
    MOM.Corr_Spreads_GDP=cor(GDP_cyc,PATHS.Spreads)
    MOM.Corr_TB_GDP=cor(GDP_cyc,TB_y_TS)
    MOM.Corr_RER_GDP=cor(GDP_cyc,rer_cyc)
    return MOM
end

function AverageMomentsManySamples(MODEL::Model)
    Random.seed!(123)
    #Initiate them at 0.0 to facilitate average across samples
    NSamplesMoments=MODEL.par.NSamplesMoments
    MOMENTS=Moments()
    for i in 1:NSamplesMoments
        MOMS=ComputeMomentsOnce(MODEL)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        #Stocks
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        MOMENTS.yC_GDP=MOMENTS.yC_GDP+MOMS.yC_GDP/NSamplesMoments
        MOMENTS.K_GDP=MOMENTS.K_GDP+MOMS.K_GDP/NSamplesMoments
        MOMENTS.kN_GDP=MOMENTS.kN_GDP+MOMS.kN_GDP/NSamplesMoments
        MOMENTS.kT_GDP=MOMENTS.kT_GDP+MOMS.kT_GDP/NSamplesMoments
        MOMENTS.Λshare=MOMENTS.Λshare+MOMS.Λshare/NSamplesMoments
        #AR(1) GDP
        MOMENTS.ρ_GDP=MOMENTS.ρ_GDP+MOMS.ρ_GDP/NSamplesMoments
        MOMENTS.σϵ_GDP=MOMENTS.σϵ_GDP+MOMS.σϵ_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_inv=MOMENTS.σ_inv+MOMS.σ_inv/NSamplesMoments
        MOMENTS.σ_TB_y=MOMENTS.σ_TB_y+MOMS.σ_TB_y/NSamplesMoments
        MOMENTS.σ_RER=MOMENTS.σ_RER+MOMS.σ_RER/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
        MOMENTS.Corr_RER_GDP=MOMENTS.Corr_RER_GDP+MOMS.Corr_RER_GDP/NSamplesMoments
    end
    return MOMENTS
end

function ComputeAndSaveMoments(NAME::String,MODEL::Model)
    MOMENTS=AverageMomentsManySamples(MODEL)
    RowNames=["DefaultPr";
              "MeanSpreads";
              "StdSpreads";
              "Debt_GDP";
              "yC_GDP";
              "ρ_GDP";
              "σϵ_GDP";
              "σ_GDP";
              "σ_con";
              "σ_TB_y";
              "σ_RER";
              "Corr_con_GDP";
              "Corr_Spreads_GDP";
              "Corr_TB_GDP";
              "Corr_RER_GDP"]
    Values=[MOMENTS.DefaultPr;
            MOMENTS.MeanSpreads;
            MOMENTS.StdSpreads;
            MOMENTS.Debt_GDP;
            MOMENTS.yC_GDP;
            MOMENTS.ρ_GDP;
            MOMENTS.σϵ_GDP;
            MOMENTS.σ_GDP;
            MOMENTS.σ_con;
            MOMENTS.σ_TB_y;
            MOMENTS.σ_RER;
            MOMENTS.Corr_con_GDP;
            MOMENTS.Corr_Spreads_GDP;
            MOMENTS.Corr_TB_GDP;
            MOMENTS.Corr_RER_GDP]
    MAT=[RowNames Values]
    writedlm(NAME,MAT,',')
    return nothing
end

###############################################################################
#Functions to Solve decentralized model
###############################################################################
function RN_dec(kNprime::Float64,z::Float64,p::Float64,
                kN::Float64,kT::Float64,T::Float64,par::Pars)
    #Return to capital in terms of the final consumption good
    @unpack δ = par
    rN=rN_dec(z,p,kN,kT,T,par)
    P=PriceFinalGood(z,p,kN,kT,T,par)
    # ψ1=dΨ_dkprime(kNprime,kN,par)
    ψ2=dΨ_dk(kNprime,kN,par)
    return (rN/P)+(1-δ)-ψ2
end

function RT_dec(kTprime::Float64,z::Float64,p::Float64,
                kN::Float64,kT::Float64,T::Float64,par::Pars)
    #Return to capital in terms of the final consumption good
    @unpack δ = par
    rT=rT_dec(z,kT,par)
    P=PriceFinalGood(z,p,kN,kT,T,par)
    # ψ1=dΨ_dkprime(kTprime,kT,par)
    ψ2=dΨ_dk(kTprime,kT,par)
    return (rT/P)+(1-δ)-ψ2
end

#Interpolation objects for households' FOCs
@with_kw mutable struct HH_itpObjects{T1,T2,T3,T4}
    #Auxiliary arrays for expectations
    dRN_Def::T1
    dRT_Def::T1
    dRN_Rep::T2
    dRT_Rep::T2
    #Arrays of expectations
    ERN_Def::T1
    ERT_Def::T1
    ERN_Rep::T2
    ERT_Rep::T2
    #Interpolation objects
    itp_dRN_Def::T3
    itp_dRT_Def::T3
    itp_dRN_Rep::T4
    itp_dRT_Rep::T4
    itp_ERN_Def::T3
    itp_ERT_Def::T3
    itp_ERN_Rep::T4
    itp_ERT_Rep::T4
end

function ComputeExpectations_HHDef!(itp_D,itp_R,E_MAT::SharedArray{Float64,4},GRIDS::Grids,par::Pars)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack θ = par
    @unpack GR_b, GR_kN, GR_kT = GRIDS
    @sync @distributed for I in CartesianIndices(E_MAT)
        (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
        kNprime=GR_kN[kN_ind]
        kTprime=GR_kT[kT_ind]
        function foo_mat(zprime::Float64,pprime_ind::Int64)
            return θ*max(0.0,itp_R(0.0,kTprime,kNprime,zprime,pprime_ind))+(1-θ)*max(0.0,itp_D(kTprime,kNprime,zprime,pprime_ind))
        end
        E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
    end
    return nothing
end

function ComputeExpectations_HHRep!(itp_D,itp_R,E_MAT::SharedArray{Float64,5},SOLUTION::Solution,GRIDS::Grids)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack itp_VD, itp_VP = SOLUTION
    @unpack GR_b, GR_kN, GR_kT = GRIDS
    @sync @distributed for I in CartesianIndices(E_MAT)
        (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
        bprime=GR_b[b_ind]; kNprime=GR_kN[kN_ind]; kTprime=GR_kT[kT_ind]
        function foo_mat(zprime::Float64,pprime_ind::Int64)
            if itp_VD(kTprime,kNprime,zprime,pprime_ind)>itp_VP(bprime,kTprime,kNprime,zprime,pprime_ind)
                return max(0.0,itp_D(kTprime,kNprime,zprime,pprime_ind))
            else
                return max(0.0,itp_R(bprime,kTprime,kNprime,zprime,pprime_ind))
            end
        end
        E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
    end
    return nothing
end

function HH_k_Returns_D(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, β = par
    @unpack kNprime_D, kTprime_D = SOLUTION
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    #Unpack state and compute output
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    zD=zDefault(z,par)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]
    #Unpack policies
    kNprimef=kNprime_D[I]; kTprimef=kTprime_D[I]
    T=0.0
    cons=Evaluate_cons_state(true,I,kNprimef,kTprimef,0.0,SOLUTION,GRIDS,par)
    if cons<=0.0
        cons=cmin
    end
    RN_Def=β*U_c(cons,par)*RN_dec(kNprimef,zD,p,kN,kT,T,par)
    RT_Def=β*U_c(cons,par)*RT_dec(kTprimef,zD,p,kN,kT,T,par)

    return RN_Def, RT_Def
end

function HH_k_Returns_P(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, β = par
    @unpack GR_z, GR_p, GR_kN, GR_kT, GR_b = GRIDS
    @unpack kNprime, kTprime, bprime, Tr = SOLUTION
    @unpack VP, VD = SOLUTION
    #Unpack state index
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    #Unpack state and policies
    z=GR_z[z_ind]; p=GR_p[p_ind]
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]
    kNprimef=kNprime[I]; kTprimef=kTprime[I]

    #Compute consumption
    T=Tr[I]
    y=FinalOutput(z,p,kN,kT,T,par)
    NegConDef=0.0
    cons=ConsNet(y,kN,kT,kNprimef,kTprimef,par)

    if y>0.0 && cons >0.0
        #Calculate return
        RN_Rep=β*U_c(cons,par)*RN_dec(kNprimef,z,p,kN,kT,T,par)
        RT_Rep=β*U_c(cons,par)*RT_dec(kTprimef,z,p,kN,kT,T,par)

        return RN_Rep, RT_Rep, NegConDef
    else
        #Use return in default and flag if default is somehow not chosen
        #this avoids overshooting of interpolation of RN and RT
        if VP[I]>VD[kT_ind,kN_ind,z_ind,p_ind]
            #Problem, somewhow repay with negative consumption
            #Flag it
            NegConDef=1.0
        end
        @unpack dRN_Def, dRT_Def = HH_OBJ
        RN_Rep=dRN_Def[kT_ind,kN_ind,z_ind,p_ind]
        RT_Rep=dRT_Def[kT_ind,kN_ind,z_ind,p_ind]
        return RN_Rep, RT_Rep, NegConDef
    end
end

function UpdateHH_Obj!(IncludeRepayment::Bool,SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill auxiliary matrices in default
    #Do default first, it will be needed below
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sdRN_Def)
        SHARED_AUX.sdRN_Def[I], SHARED_AUX.sdRT_Def[I]=HH_k_Returns_D(I,SOLUTION,GRIDS,par)
    end
    HH_OBJ.dRN_Def .= SHARED_AUX.sdRN_Def
    HH_OBJ.dRT_Def .= SHARED_AUX.sdRT_Def
    HH_OBJ.itp_dRN_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dRN_Def,true,GRIDS)
    HH_OBJ.itp_dRT_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dRT_Def,true,GRIDS)

    if IncludeRepayment
        #Loop over all states to fill auxiliary matrices in repayment
        @sync @distributed for I in CartesianIndices(SHARED_AUX.sdRN_Rep)
            SHARED_AUX.sdRN_Rep[I], SHARED_AUX.sdRT_Rep[I], SHARED_AUX.sSMOOTHED[I]=HH_k_Returns_P(I,HH_OBJ,SOLUTION,GRIDS,par)
        end
        HH_OBJ.dRN_Rep .= SHARED_AUX.sdRN_Rep
        HH_OBJ.dRT_Rep .= SHARED_AUX.sdRT_Rep

        HH_OBJ.itp_dRN_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.dRN_Rep,false,GRIDS)
        HH_OBJ.itp_dRT_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.dRT_Rep,false,GRIDS)
    end

    ##################################################
    ### Need all itp objects for both expectations ###
    ##################################################
    #Compute expectations in default
    ComputeExpectations_HHDef!(HH_OBJ.itp_dRN_Def,HH_OBJ.itp_dRN_Rep,SHARED_AUX.sdRN_Def,GRIDS,par)
    ComputeExpectations_HHDef!(HH_OBJ.itp_dRT_Def,HH_OBJ.itp_dRT_Rep,SHARED_AUX.sdRT_Def,GRIDS,par)
    HH_OBJ.ERN_Def .= SHARED_AUX.sdRN_Def
    HH_OBJ.ERT_Def .= SHARED_AUX.sdRT_Def
    HH_OBJ.itp_ERN_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.ERN_Def,true,GRIDS)
    HH_OBJ.itp_ERT_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.ERT_Def,true,GRIDS)

    if IncludeRepayment
        #Compute expectations in repayment
        ComputeExpectations_HHRep!(HH_OBJ.itp_dRN_Def,HH_OBJ.itp_dRN_Rep,SHARED_AUX.sdRN_Rep,SOLUTION,GRIDS)
        ComputeExpectations_HHRep!(HH_OBJ.itp_dRT_Def,HH_OBJ.itp_dRT_Rep,SHARED_AUX.sdRT_Rep,SOLUTION,GRIDS)
        HH_OBJ.ERN_Rep .= SHARED_AUX.sdRN_Rep
        HH_OBJ.ERT_Rep .= SHARED_AUX.sdRT_Rep
        HH_OBJ.itp_ERN_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.ERN_Rep,false,GRIDS)
        HH_OBJ.itp_ERT_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.ERT_Rep,false,GRIDS)
    end

    return nothing
end

function InitiateHH_Obj(GRIDS::Grids,par::Pars)
    @unpack Nz, Np, NkN, NkT, Nb = par
    #Allocate empty arrays
    #Auxiliary arrays for expectations
    dRN_Def=zeros(Float64,NkT,NkN,Nz,Np)
    dRT_Def=zeros(Float64,NkT,NkN,Nz,Np)
    dRN_Rep=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    dRT_Rep=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    #Arrays of expectations
    ERN_Def=zeros(Float64,NkT,NkN,Nz,Np)
    ERT_Def=zeros(Float64,NkT,NkN,Nz,Np)
    ERN_Rep=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    ERT_Rep=zeros(Float64,Nb,NkT,NkN,Nz,Np)
    #Create interpolation objects
    itp_dRN_Def=CreateInterpolation_HouseholdObjects(ERN_Def,true,GRIDS)
    itp_dRT_Def=CreateInterpolation_HouseholdObjects(ERT_Def,true,GRIDS)
    itp_dRN_Rep=CreateInterpolation_HouseholdObjects(ERN_Rep,false,GRIDS)
    itp_dRT_Rep=CreateInterpolation_HouseholdObjects(ERT_Rep,false,GRIDS)

    itp_ERN_Def=CreateInterpolation_HouseholdObjects(ERN_Def,true,GRIDS)
    itp_ERT_Def=CreateInterpolation_HouseholdObjects(ERT_Def,true,GRIDS)
    itp_ERN_Rep=CreateInterpolation_HouseholdObjects(ERN_Rep,false,GRIDS)
    itp_ERT_Rep=CreateInterpolation_HouseholdObjects(ERT_Rep,false,GRIDS)

    #Arrange objects in structure
    return HH_itpObjects(dRN_Def,dRT_Def,dRN_Rep,dRT_Rep,ERN_Def,ERT_Def,ERN_Rep,ERT_Rep,itp_dRN_Def,itp_dRT_Def,itp_dRN_Rep,itp_dRT_Rep,itp_ERN_Def,itp_ERT_Def,itp_ERN_Rep,itp_ERT_Rep)
end

#Functions to compute FOCs given (s,x) and a try of K'
function HH_FOCs_Def(I::CartesianIndex,kNprime::Float64,kTprime::Float64,
                     HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Compute sum of squared Euler Equation residuals
    @unpack cmin = par
    @unpack GR_z, GR_p, GR_kN, GR_kT = GRIDS
    @unpack itp_ERN_Def, itp_ERT_Def = HH_OBJ

    #Compute present consumption
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]
    cons=Evaluate_cons_state(true,I,kNprime,kTprime,0.0,SOLUTION,GRIDS,par)

    if cons>0.0
        # compute expectation over z and pC
        EvN=itp_ERN_Def(kTprime,kNprime,z,p_ind)
        EvT=itp_ERT_Def(kTprime,kNprime,z,p_ind)
        #compute extra term and return FOCs
        ψ1N=dΨ_dkprime(kNprime,kN,par)
        ψ1T=dΨ_dkprime(kTprime,kT,par)

        #Compute Euler equation residuals
        Euler_N=EvN-U_c(cons,par)*(1+ψ1N)
        Euler_T=EvT-U_c(cons,par)*(1+ψ1T)

        return (Euler_N^2)+(Euler_T^2)
    else
        #Make it worse than staying the same
        kkNN=kN
        kkTT=kT
        cons=Evaluate_cons_state(true,I,kkNN,kkTT,0.0,SOLUTION,GRIDS,par)
        EvN=itp_ERN_Def(kkTT,kkNN,z,p_ind)
        EvT=itp_ERT_Def(kkTT,kkNN,z,p_ind)
        #compute extra term and return FOCs
        ψ1N=dΨ_dkprime(kkNN,kN,par)
        ψ1T=dΨ_dkprime(kkTT,kT,par)

        #Compute Euler equation residuals
        Euler_N=EvN-U_c(cons,par)*(1+ψ1N)
        Euler_T=EvT-U_c(cons,par)*(1+ψ1T)

        return ((Euler_N^2)+(Euler_T^2))*2
    end
end

function HH_FOCs_Rep(I::CartesianIndex,kNprime::Float64,kTprime::Float64,bprime::Float64,
                      HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Compute sum of squared Euler Equation residuals
    @unpack cmin, WithInvTax = par
    @unpack GR_z, GR_p, GR_kN, GR_kT, GR_b = GRIDS
    @unpack itp_q1 = SOLUTION
    @unpack itp_ERN_Rep, itp_ERT_Rep = HH_OBJ

    #Compute output
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]
    T=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)
    y=FinalOutput(z,p,kN,kT,T,par)

    #Compute subsidy
    if WithInvTax
        qq=itp_q1(bprime,kTprime,kNprime,z,p_ind)
        τsN, τsT=SimpleSubsidy(qq,par)
    else
        τsN=0.0
        τsT=0.0
    end
    cons=ConsNet(y,kN,kT,kNprime,kTprime,par)

    if cons>0.0 && y>0.0
        #compute expectation over z and yC
        EvN=itp_ERN_Rep(bprime,kTprime,kNprime,z,p_ind)
        EvT=itp_ERT_Rep(bprime,kTprime,kNprime,z,p_ind)
        #compute extra term and return FOC
        ψ1N=dΨ_dkprime(kNprime,kN,par)
        ψ1T=dΨ_dkprime(kTprime,kT,par)

        #Compute Euler equation residuals
        if WithInvTax
            P=PriceFinalGood(z,p,kN,kT,T,par)
            Euler_N=EvN-U_c(cons,par)*(1+ψ1N-(τsN/P))
            Euler_T=EvT-U_c(cons,par)*(1+ψ1T-(τsT/P))
        else
            Euler_N=EvN-U_c(cons,par)*(1+ψ1N)
            Euler_T=EvT-U_c(cons,par)*(1+ψ1T)
        end
        return (Euler_N^2)+(Euler_T^2)
    else
        #Make it worse than staying the same
        kkNN=kN
        kkTT=kT
        cons=ConsNet(y,kN,kT,kkNN,kkTT,par)

        EvN=itp_ERN_Rep(bprime,kkTT,kkNN,z,p_ind)
        EvT=itp_ERT_Rep(bprime,kkTT,kkNN,z,p_ind)
        #compute extra term and return FOCs
        ψ1N=dΨ_dkprime(kkNN,kN,par)
        ψ1T=dΨ_dkprime(kkTT,kT,par)

        #Compute Euler equation residuals
        Euler_N=EvN-U_c(cons,par)*(1+ψ1N)
        Euler_T=EvT-U_c(cons,par)*(1+ψ1T)

        return ((Euler_N^2)+(Euler_T^2))*2
    end
end

#Functions to compute optimal policy from FOCs
function HHOptim_Def(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    @unpack kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt = par
    @unpack GR_kN, GR_kT = GRIDS

    #Compute present consumption
    (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]

    #Choose last policy as initial guess only if it yields positive consumption
    Kpol_1=true
    X0_BOUNDS=InitialPolicyGuess_D(Kpol_1,I,SOL_GUESS,GRIDS,par)

    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kTlowOpt,kThighOpt)

    #Perform optimization with MatLab simplex
    fooEuler(X::Array{Float64,1})=HH_FOCs_Def(I,TransformIntoBounds(X[1],kNlowOpt,kNhighOpt),TransformIntoBounds(X[2],kTlowOpt,kThighOpt),HH_OBJ,SOLUTION,GRIDS,par)
    inner_optimizer = NelderMead()
    OPTIONS=Optim.Options(g_tol = 1e-13)
    res=optimize(fooEuler,X0,inner_optimizer,OPTIONS)

    #Transform optimizer into bounds
    kNprime=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)

    cons=Evaluate_cons_state(true,I,kNprime,kTprime,0.0,SOLUTION,GRIDS,par)
    if cons>0.0
        return kNprime, kTprime
    else
        #Return planner's policy, which was the guess
        return X0_BOUNDS[1], X0_BOUNDS[2]
    end
end

function HHOptim_Rep(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    @unpack kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt = par
    @unpack GR_kN, GR_kT = GRIDS

    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]
    #Choose last policy as initial guess
    Kpol_1=true
    vDmin=-0.0
    X0_BOUNDS=InitialPolicyGuess_P(vDmin,Kpol_1,I,SOL_GUESS,GRIDS,par)

    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kTlowOpt,kThighOpt)

    #Perform optimization
    fooEuler(X::Array{Float64,1})=HH_FOCs_Rep(I,TransformIntoBounds(X[1],kNlowOpt,kNhighOpt),TransformIntoBounds(X[2],kTlowOpt,kThighOpt),bprime,HH_OBJ,SOLUTION,GRIDS,par)
    inner_optimizer = NelderMead()
    OPTIONS=Optim.Options(g_tol = 1e-13)
    res=optimize(fooEuler,X0,inner_optimizer,OPTIONS)

    #Transform optimizer into bounds
    kNprime=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)

    return kNprime, kTprime
end

#Update equilibrium objects in decentralized economy
function DefaultUpdater_DEC!(I::CartesianIndex,SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    if par.yC==0.0
        #No commodities, solution at p_ind=1 is the same as p_ind=2
        #only solve if p_ind=1 and then allocate to p_ind==2
        (kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)

        if p_ind==1
            SHARED_AUX.skNprime_D[I], SHARED_AUX.skTprime_D[I]=HHOptim_Def(I,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
            SHARED_AUX.sVD[I]=Evaluate_VD(I,SHARED_AUX.skNprime_D[I],SHARED_AUX.skTprime_D[I],SOLUTION,GRIDS,par)

            SHARED_AUX.sVD[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sVD[I]
            SHARED_AUX.skNprime_D[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skNprime_D[I]
            SHARED_AUX.skTprime_D[kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skTprime_D[I]
        end
    else
        SHARED_AUX.skNprime_D[I], SHARED_AUX.skTprime_D[I]=HHOptim_Def(I,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
        SHARED_AUX.sVD[I]=Evaluate_VD(I,SHARED_AUX.skNprime_D[I],SHARED_AUX.skTprime_D[I],SOLUTION,GRIDS,par)
    end
    return nothing
end

function UpdateDefault_DEC!(SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    #Loop over all states to fill shared arrays
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sVD)
        DefaultUpdater_DEC!(I,SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    end

    SOLUTION.VD .= SHARED_AUX.sVD
    SOLUTION.kNprime_D .= SHARED_AUX.skNprime_D
    SOLUTION.kTprime_D .= SHARED_AUX.skTprime_D

    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kNprime_D=CreateInterpolation_Policies(SOLUTION.kNprime_D,true,GRIDS)
    SOLUTION.itp_kTprime_D=CreateInterpolation_Policies(SOLUTION.kTprime_D,true,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(true,SOLUTION.VD,SHARED_AUX.sVD,GRIDS)
    SOLUTION.EVD=SHARED_AUX.sVD
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function ValueInRepayment_DEC(vDmin::Float64,I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    kNprime, kTprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    return Evaluate_VP(vDmin,I,kNprime,kTprime,bprime,SOLUTION,GRIDS,par)
end

function OptimInRepayment_DEC(vDmin::Float64,I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    @unpack GR_kN, GR_kT, GR_z, GR_p, GR_b = GRIDS
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]; kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]

    foo(bprime::Float64)=-ValueInRepayment_DEC(vDmin,I,bprime,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    res=optimize(foo,par.blowOpt,par.bhighOpt,GoldenSection())

    vv=-Optim.minimum(res)
    bprime=Optim.minimizer(res)

    kNprime, kTprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    Tr=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOLUTION,par)

    #Check if optimizer yields negative consumption or negative output
    y=FinalOutput(z,p,kN,kT,Tr,par)
    cons=ConsNet(y,kN,kT,kNprime,kTprime,par)
    if y>0.0 && cons>0.0
        #Solution works, return as is
        return vv, kNprime, kTprime, bprime, Tr
    else
        #Negative output or consumption, return value in default VD-ϵ
        #and default capital policies
        vv=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]-sqrt(eps(Float64))
        kNprime=SOLUTION.kNprime_D[kT_ind,kN_ind,z_ind,p_ind]
        kTprime=SOLUTION.kTprime_D[kT_ind,kN_ind,z_ind,p_ind]
        #Make b' stay flat
        bprime=SOLUTION.bprime[max(1,b_ind-1),kT_ind,kN_ind,z_ind,p_ind]
        Tr=0.0
        return vv, kNprime, kTprime, bprime, Tr
    end
end

function RepaymentUpdater_DEC!(vDmin::Float64,I::CartesianIndex,SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    (b_ind,kT_ind,kN_ind,z_ind,p_ind)=Tuple(I)
    if par.yC==0.0
        #No commodities, solution at p_ind=1 is the same as p_ind=2
        #only solve if p_ind=1 and then allocate to p_ind==2
        if p_ind==1
            SHARED_AUX.sVP[I], SHARED_AUX.skNprime[I], SHARED_AUX.skTprime[I], SHARED_AUX.sbprime[I], SHARED_AUX.sTr[I]=OptimInRepayment_DEC(vDmin,I,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
            if SHARED_AUX.sVP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
                SHARED_AUX.sV[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
            else
                SHARED_AUX.sV[I]=SHARED_AUX.sVP[I]
            end
            SHARED_AUX.sVP[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sVP[I]
            SHARED_AUX.skNprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skNprime[I]
            SHARED_AUX.skTprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.skTprime[I]
            SHARED_AUX.sbprime[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sbprime[I]
            SHARED_AUX.sTr[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sTr[I]
            SHARED_AUX.sV[b_ind,kT_ind,kN_ind,z_ind,2]=SHARED_AUX.sV[I]
        end
    else
        SHARED_AUX.sVP[I], SHARED_AUX.skNprime[I], SHARED_AUX.skTprime[I], SHARED_AUX.sbprime[I], SHARED_AUX.sTr[I]=OptimInRepayment_DEC(vDmin,I,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
        if SHARED_AUX.sVP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
            SHARED_AUX.sV[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind,p_ind]
        else
            SHARED_AUX.sV[I]=SHARED_AUX.sVP[I]
        end
    end
    return nothing
end

function UpdateRepayment_DEC!(SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    #Minimum value in default to aide with
    #numerical optimization dealing with negative consumption
    vDmin=minimum(SOLUTION.VD)
    #Loop over all states to fill value of repayment
    @sync @distributed for I in CartesianIndices(SHARED_AUX.sVP)
        RepaymentUpdater_DEC!(vDmin,I,SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    end

    SOLUTION.VP .= SHARED_AUX.sVP
    SOLUTION.V .= SHARED_AUX.sV
    SOLUTION.kNprime .= SHARED_AUX.skNprime
    SOLUTION.kTprime .= SHARED_AUX.skTprime
    SOLUTION.bprime .= SHARED_AUX.sbprime
    SOLUTION.Tr .= SHARED_AUX.sTr

    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kNprime=CreateInterpolation_Policies(SOLUTION.kNprime,false,GRIDS)
    SOLUTION.itp_kTprime=CreateInterpolation_Policies(SOLUTION.kTprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)

    #Compute expectation
    Update_EV_OverStates!(SHARED_AUX,SOLUTION,GRIDS,par)
    return nothing
end

function UpdateSolution_DEC!(SHARED_AUX::SharedAux,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    IncludeRepayment=true
    UpdateDefault_DEC!(SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    UpdateRepayment_DEC!(SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par,SOL_GUESS)
    UpdateBondsPrice!(SHARED_AUX,SOLUTION,GRIDS,par)
    UpdateHH_Obj!(IncludeRepayment,SHARED_AUX,HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

#Solve decentralized model
function SolveModel_VFI_DEC(PrintProg::Bool,NAME::String,GRIDS::Grids,par::Pars,SOL_GUESS::Solution)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    SHARED_AUX=InitiateSharedAux(GRIDS,par)
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    HH_OBJ=InitiateHH_Obj(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)

    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)# && NotConvPct>Tolpct_q))
        UpdateSolution_DEC!(SHARED_AUX,HH_OBJ,SOLUTION_NEXT,GRIDS,par,SOL_GUESS)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        SaveModel_Vector(NAME,SOLUTION_NEXT,par)
    end
    return SOLUTION_NEXT
end

function SolveAndSaveModel_DEC(NAME::String,SOL_GUESS::Solution,GRIDS::Grids,par::Pars)
    SOL=SolveModel_VFI_DEC(true,NAME,GRIDS,par,SOL_GUESS)
    SaveModel_Vector(NAME,SOL,par)
    return nothing
end

#Solve decentralized and planner simultaneously
#here the planner policy for capital will serve as the decentralized initial guess
function SolveModel_VFI_BOTH(NAME_DEC::String,NAME_PLA::String,PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par

    SHARED_AUX=InitiateSharedAux(GRIDS,par)

    SOLUTION_CURRENT_PLA=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT_PLA=deepcopy(SOLUTION_CURRENT_PLA)

    SOLUTION_CURRENT_DEC=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT_DEC=deepcopy(SOLUTION_CURRENT_DEC)
    HH_OBJ=InitiateHH_Obj(GRIDS,par)

    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)# && NotConvPct>Tolpct_q))
        UpdateSolution_PLA!(SHARED_AUX,SOLUTION_NEXT_PLA,GRIDS,par)
        UpdateSolution_DEC!(SHARED_AUX,HH_OBJ,SOLUTION_NEXT_DEC,GRIDS,par,SOLUTION_NEXT_PLA)
        cnt=cnt+1

        dst_q_DEC, NotConvPct_DEC, Ix_DEC=ComputeDistance_q(SOLUTION_CURRENT_DEC,SOLUTION_NEXT_DEC,par)
        dst_D_DEC, dst_P_DEC, Iv_DEC, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT_DEC,SOLUTION_NEXT_DEC,par)
        dst_V_DEC=max(dst_D_DEC,dst_P_DEC)
        SOLUTION_CURRENT_DEC=deepcopy(SOLUTION_NEXT_DEC)

        dst_q_PLA, NotConvPct_PLA, Ix_PLA=ComputeDistance_q(SOLUTION_CURRENT_PLA,SOLUTION_NEXT_PLA,par)
        dst_D_PLA, dst_P_PLA, Iv_PLA, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT_PLA,SOLUTION_NEXT_PLA,par)
        dst_V_PLA=max(dst_D_PLA,dst_P_PLA)
        SOLUTION_CURRENT_PLA=deepcopy(SOLUTION_NEXT_PLA)

        dst_V=max(dst_V_DEC,dst_V_PLA)
        dst_q=max(dst_q_DEC,dst_q_PLA)

        SaveModel_Vector(NAME_DEC,SOLUTION_NEXT_DEC,par)
        SaveModel_Vector(NAME_PLA,SOLUTION_NEXT_PLA,par)
    end
    return SOLUTION_NEXT_DEC, SOLUTION_NEXT_PLA
end

function SolveAndSaveModel_BOTH(NAME_DEC::String,NAME_PLA::String,GRIDS::Grids,par::Pars)
    SOL_DEC, SOL_PLA=SolveModel_VFI_BOTH(NAME_DEC,NAME_PLA,true,GRIDS,par)
    SaveModel_Vector(NAME_DEC,SOL_DEC,par)
    SaveModel_Vector(NAME_PLA,SOL_PLA,par)
    return nothing
end

function SolveClosed_VFI_DEC(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    parClosed=Pars(par,θ=0.0,d0=0.0,d1=0.0)
    IncludeRepayment=false
    SHARED_AUX=InitiateSharedAux(GRIDS,par)
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,parClosed)
    HH_OBJ=InitiateHH_Obj(GRIDS,parClosed)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)

    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while (dst_V>Tol_V || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateDefault_DEC!(SHARED_AUX,HH_OBJ,SOLUTION_NEXT,GRIDS,parClosed,SOLUTION_CURRENT)
        UpdateHH_Obj!(IncludeRepayment,SHARED_AUX,HH_OBJ,SOLUTION_NEXT,GRIDS,parClosed)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,parClosed)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
    end
    return SOLUTION_NEXT
end

function ModelClosed_VFI_DEC(PrintProg::Bool,GRIDS::Grids,par::Pars)
    SOL=SolveClosed_VFI_DEC(PrintProg,GRIDS,par)
    parClosed=Pars(par,θ=0.0,d0=0.0,d1=0.0)
    return Model(SOL,GRIDS,parClosed)
end

###############################################################################
#Functions to compute welfare
###############################################################################
function WelfareGains_VF(z::Float64,p_ind::Int64,kN::Float64,kT::Float64,b::Float64,MOD0::Model,MOD1::Model)
    @unpack par = MOD0
    @unpack σ = par
    v0=MOD0.SOLUTION.itp_V(b,kT,kN,z,p_ind)
    v1=MOD1.SOLUTION.itp_V(b,kT,kN,z,p_ind)
    return 100*(((v1/v0)^(1/(1-σ)))-1)
end

function AverageWelfareGains_VF(N::Int64,MOD0::Model,MOD1::Model)
    @unpack par, GRIDS = MOD0
    @unpack ρ_z, μ_z, σ_ϵz = par
    Random.seed!(1234)
    T=N
    TS=SimulatePaths(0.0,T,MOD0;FixSeed=false)
    wg=0.0
    for i in 1:N
        z=TS.z[i]; p_ind=TS.p_ind[i]
        kN=TS.KN[i]; kT=TS.KT[i]; b=TS.B[i]

        wg=wg+WelfareGains_VF(z,p_ind,kN,kT,b,MOD0,MOD1)
    end
    return wg/N
end

###############################################################################
#Analyze optimal taxes
###############################################################################
function StateContingentSubsidies(z::Float64,p_ind::Int64,p::Float64,kN::Float64,kT::Float64,b::Float64,SOL_PLA::Solution,par::Pars)
    @unpack γ = par
    @unpack itp_q1, itp_kNprime, itp_kTprime, itp_bprime = SOL_PLA
    hh=1e-6
    kNprime=itp_kNprime(b,kN,kT,z,p_ind)
    kTprime=itp_kTprime(b,kN,kT,z,p_ind)
    bprime=itp_bprime(b,kN,kT,z,p_ind)
    dq_dkN=(itp_q1(bprime,kTprime,kNprime+hh,z,p_ind)-itp_q1(bprime,kTprime,kNprime-hh,z,p_ind))/(2*hh)
    dq_dkT=(itp_q1(bprime,kTprime+hh,kNprime,z,p_ind)-itp_q1(bprime,kTprime-hh,kNprime,z,p_ind))/(2*hh)

    T=Calculate_Tr(p_ind,z,b,kNprime,kTprime,bprime,SOL_PLA,par)

    τsN=100*dq_dkN*(bprime-(1-γ)*b)
    τsT=100*dq_dkT*(bprime-(1-γ)*b)

    return τsN, τsT
end

function TimeSeriesOfSubsidies(PATHS::Paths,MOD::Model)
    @unpack SOLUTION, par = MOD
    T=length(PATHS.z)
    τsN_TS=Array{Float64,1}(undef,T)
    τsT_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        z=PATHS.z[t]; p=PATHS.p[t]; p_ind=PATHS.p_ind[t]
        kN=PATHS.KN[t]; kT=PATHS.KT[t]; b=PATHS.B[t]
        τsN_TS[t], τsT_TS[t]=StateContingentSubsidies(z,p_ind,p,kN,kT,b,SOLUTION,par)
    end
    return τsN_TS, τsT_TS
end

function SubsidyRuleRegression(PATHS::Paths,MOD::Model)
    τsN_TS, τsT_TS=TimeSeriesOfSubsidies(PATHS,MOD)
    T=length(PATHS.p)
    X=ones(Float64,T,2)
    X[:,2] .= 0.01*(PATHS.Spreads .* (1 .- PATHS.Def))

    YN=τsN_TS
    BETA_N=(X'*X)\(X'*YN)

    YT=τsT_TS
    BETA_T=(X'*X)\(X'*YT)

    return BETA_N, BETA_T
end
