
using Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots

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
    γ::Float64 = 1.0#0.05         #Reciprocal of average maturity
    κ::Float64 = 0.03         #Coupon payments
    #Default cost and debt parameters
    θ::Float64 = 1/(mean([12 1 6])*4)        #Probability of re-admission (2 years)
    d0::Float64 = -0.18819#-0.2914       #income default cost
    d1::Float64 = 0.24558#0.4162        #income default cost
    #Capital allocations
    WithInvTax::Bool = false     #Consider distortionary tax
    τs::Float64 = 0.0       #Fixed subsidy for Λ
    kbar::Float64 = 1.0     #Fixed capital stock in the economy
    φ::Float64 = 0.0        #Adjustment cost
    #Production functions
    #Final consumption good
    η::Float64 = 0.83
    ω::Float64 = 0.66
    #Value added of intermediates
    αN::Float64 = 0.36#0.66          #Capital share in non-traded sector
    αT::Float64 = 0.36#0.57          #Capital share in manufacturing sector
    #Commodity endowment
    yC::Float64 = 1.35
    #Stochastic process
    #parameters for productivity shock
    σ_ϵz::Float64 = 0.017
    μ_ϵz::Float64 = 0.0-0.5*(σ_ϵz^2)
    dist_ϵz::UnivariateDistribution = truncated(Normal(μ_ϵz,σ_ϵz),-3.0*σ_ϵz,3.0*σ_ϵz)
    ρ_z::Float64 = 0.95
    μ_z::Float64 = exp(μ_ϵz+0.5*(σ_ϵz^2))
    zlow::Float64 = exp(log(μ_z)-3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #parameters for commodity price shock
    AvDevFromMean::Float64 = 0.406
    pCL::Float64 = 1.00-AvDevFromMean          #Normal times
    pCH::Float64 = 1.00+AvDevFromMean          #Commodity windfall
    L_dur::Float64 = 12.8          #Average quarters low price
    H_dur::Float64 = 16.0          #Average quarters high price
    #Quadrature parameter
    N_GLz::Int64 = 50
    #Grids
    Nz::Int64 = 11
    Np::Int64 = 2
    NΛ::Int64 = 11
    Nb::Int64 = 21
    Λlow::Float64 = 0.01#0.001 for η=0.5
    Λhigh::Float64 = 0.6#0.46 for η=0.5
    blow::Float64 = 0.0
    bhigh::Float64 = 0.75#15.0
    #Parameters for solution algorithm
    cmin::Float64 = 1e-2
    Tol_V::Float64 = 1e-6       #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-4       #Tolerance for absolute distance for q
    Tolpct_q::Float64 = 1.0        #Tolerance for % of states for which q has not converged
    cnt_max::Int64 = 100           #Maximum number of iterations on VFI
    MaxIter_Opt::Int64 = 1000
    g_tol_Opt::Float64 = 1e-4
    ΛlowOpt::Float64 = 0.9*Λlow             #Minimum level of capital for optimization
    ΛhighOpt::Float64 = 1.1*Λhigh           #Maximum level of capital for optimization
    blowOpt::Float64 = blow-0.01             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    #Simulation parameters
    Tsim::Int64 = 10000
    drp::Int64 = 5000
    Tmom::Int64 = 400
    NSamplesMoments::Int64 = 100
    Npaths::Int64 = 100
    HPFilter_Par::Float64 = 1600.0
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64,1}
    GR_p::Array{Float64,1}
    GR_Λ::Array{Float64,1}
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
    @unpack pCL, pCH, L_dur, H_dur = par
    GR_p=Array{Float64,1}(undef,2)
    GR_p[1]=pCL
    GR_p[2]=pCH
    PI_pC=Array{Float64,2}(undef,2,2)
    πW=1/L_dur
    PI_pC[1,1]=1.0-πW
    PI_pC[1,2]=πW
    PI_pC[2,1]=1.0/H_dur
    PI_pC[2,2]=1.0-PI_pC[2,1]

    #Grid of capital allocation
    @unpack NΛ, Λlow, Λhigh = par
    GR_Λ=collect(range(Λlow,stop=Λhigh,length=NΛ))

    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    return Grids(GR_z,GR_p,GR_Λ,GR_b,ϵz_weights,ϵz_nodes,FacQz,ZPRIME,PDFz,PI_pC)
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
    Λprime_D::T1
    Λprime::T2
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
    itp_Λprime_D::T6
    itp_Λprime::T7
    itp_bprime::T7
end

################################################################
#################### Auxiliary functions #######################
################################################################
function MyBisection(foo,a::Float64,b::Float64;xatol::Float64=1e-8)
    s=sign(foo(a))
    x=(a+b)/2.0
    d=(b-a)/2.0
    while d>xatol
        d=d/2.0
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
    @unpack GR_z, GR_p, GR_Λ = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Λs=range(GR_Λ[1],stop=GR_Λ[end],length=length(GR_Λ))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()#Cubic(Line(OnGrid()))
    ORDER_B_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Λs,Zs,Ps),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Λs,Zs,Ps),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_p, GR_Λ, GR_b = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Λs=range(GR_Λ[1],stop=GR_Λ[end],length=length(GR_Λ))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Λs,Zs,Ps),Interpolations.Flat())
end

function CreateInterpolation_Policies(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_Λ = GRIDS
    Ps=range(1,stop=length(GR_p),length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Λs=range(GR_Λ[1],stop=GR_Λ[end],length=length(GR_Λ))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Λs,Zs,Ps),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Λs,Zs,Ps),Interpolations.Flat())
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

function ComputeExpectationOverStates!(IsDefault::Bool,MAT::Array{Float64},E_MAT::Array{Float64},GRIDS::Grids)
    #It will use MAT to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    if IsDefault
        for I in CartesianIndices(MAT)
            (Λ_ind,z_ind,p_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[Λ_ind,:,:],GRIDS)
            E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
        end
    else
        for I in CartesianIndices(MAT)
            (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[b_ind,Λ_ind,:,:],GRIDS)
            E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
        end
    end
    return nothing
end

################################################################################
### Functions to save solution in CSV
################################################################################
function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nz, Np, NΛ, Nb = par
    ### Allocate all values to object
    VD=zeros(Float64,NΛ,Nz,Np)
    VP=zeros(Float64,Nb,NΛ,Nz,Np)
    V=zeros(Float64,Nb,NΛ,Nz,Np)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,NΛ,Nz,Np)
    EV=zeros(Float64,Nb,NΛ,Nz,Np)
    q1=zeros(Float64,Nb,NΛ,Nz,Np)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    #Policy functions
    Λprime_D=zeros(Float64,NΛ,Nz,Np)
    Λprime=zeros(Float64,Nb,NΛ,Nz,Np)
    bprime=zeros(Float64,Nb,NΛ,Nz,Np)
    Tr=zeros(Float64,Nb,NΛ,Nz,Np)
    itp_Λprime_D=CreateInterpolation_Policies(Λprime_D,true,GRIDS)
    itp_Λprime=CreateInterpolation_Policies(Λprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,Λprime_D,Λprime,bprime,Tr,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_Λprime_D,itp_Λprime,itp_bprime)
end

function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment first
    @unpack VP, V, EV, q1 = SOLUTION
    @unpack Λprime, bprime, Tr = SOLUTION
    VEC=reshape(VP,(:))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(q1,(:)))
    VEC=vcat(VEC,reshape(Λprime,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))
    VEC=vcat(VEC,reshape(Tr,(:)))

    #Then stack vectors of default
    @unpack VD, EVD, Λprime_D = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))
    VEC=vcat(VEC,reshape(Λprime_D,(:)))

    return VEC
end

function SaveSolution_Vector(NAME::String,SOLUTION::Solution)
    #Stack vectors of repayment first
    VEC=StackSolution_Vector(SOLUTION)
    writedlm(NAME,VEC,',')

    return nothing
end

function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,VEC::Matrix{Float64},par::Pars)
    @unpack Nz, Np, NΛ, Nb = par
    if IsDefault
        I=(NΛ,Nz,Np)
    else
        I=(Nb,NΛ,Nz,Np)
    end
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function UnpackSolution_Vector(NAME::String,FOLDER::String,GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nz, Np, NΛ, Nb = par
    size_repayment=Np*Nz*NΛ*Nb
    size_default=Np*Nz*NΛ
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end
    #Allocate vectors into matrices
    #Repayment
    start=1
    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    q1=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    Λprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    Tr=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)

    #Default
    start=start+size_repayment
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    Λprime_D=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    itp_Λprime_D=CreateInterpolation_Policies(Λprime_D,true,GRIDS)

    itp_Λprime=CreateInterpolation_Policies(Λprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)

    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,Λprime_D,Λprime,bprime,Tr,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_Λprime_D,itp_Λprime,itp_bprime)
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
    @unpack d0, d1 = par
    return z-max(0.0,d0*z+d1*z*z)
end

function CapitalAdjustment(Λprime::Float64,Λ::Float64,par::Pars)
    @unpack φ = par
    return 0.5*φ*(Λprime-Λ)^2
end

function dΨ_d1(Λprime::Float64,Λ::Float64,par::Pars)
    @unpack φ = par
    return φ*(Λprime-Λ)
end

function dΨ_d2(Λprime::Float64,Λ::Float64,par::Pars)
    @unpack φ = par
    return -φ*(Λprime-Λ)
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

function Final_CES(cN::Float64,cT::Float64,par::Pars)
    @unpack η, ω = par
    return ((ω^(1.0/η))*(cN^((η-1.0)/η))+((1.0-ω)^(1.0/η))*(cT^((η-1.0)/η)))^(η/(η-1.0))
end

function FinalOutput(z::Float64,p::Float64,Λ::Float64,T::Float64,par::Pars)
    @unpack yC, kbar = par
    #Compute consumption of intermediate goods
    cT=TradedProduction(z,Λ*kbar,par)+p*yC+T
    cN=NonTradedProduction(z,(1-Λ)*kbar,par)
    if cT>0.0
        return Final_CES(cN,cT,par)
    else
        return cT
    end
end

################################################################
################### Setup functions ############################
################################################################
function PriceNonTraded(z::Float64,p::Float64,Λ::Float64,T::Float64,par::Pars)
    @unpack αN, αT, ω, η, yC, kbar = par
    cT=TradedProduction(z,Λ*kbar,par)+p*yC+T
    cN=NonTradedProduction(z,(1-Λ)*kbar,par)
    if cT>0.0
        return ((ω/(1.0-ω))*(cT/cN))^(1.0/η)
    else
        return 1e-2
    end
end

function PriceFinalGood(z::Float64,p::Float64,Λ::Float64,T::Float64,par::Pars)
    @unpack ω, η = par
    pN=PriceNonTraded(z,p,Λ,T,par)
    return (ω*(pN^(1.0-η))+(1.0-ω)*(1.0^(1.0-η)))^(1.0/(1.0-η))
end

function rN_dec(z::Float64,p::Float64,Λ::Float64,T::Float64,par::Pars)
    @unpack αN, kbar = par
    pN=PriceNonTraded(z,p,Λ,T,par)
    kN=(1-Λ)*kbar
    return αN*pN*z*(kN^(αN-1.0))
end

function rT_dec(z::Float64,Λ::Float64,par::Pars)
    @unpack αT, kbar = par
    kT=Λ*kbar
    return αT*z*(kT^(αT-1.0))
end

function Λsteady_state(pss::Float64,par::Pars)
    @unpack μ_z = par
    #as Λ→1 rN approaches infinity
    foo(Λ::Float64)=rT_dec(μ_z,Λ,par)-rN_dec(μ_z,pss,Λ,0.0,par)
    #so with high lambda foo<0 and with low lambda foo>0

    Λlow=1e-3
    Λhigh=1-Λlow
    return MyBisection(foo,Λlow,Λhigh;xatol=1e-4)
end

function Setup_GR(max_b_y::Float64,par::Pars)
    #Setup range for bonds
    NX=0.0
    Λss=Λsteady_state(par.pCH,par)
    Yhigh=FinalOutput(1.0,par.pCH,Λss,-NX,par)
    Phigh=PriceFinalGood(1.0,par.pCH,Λss,-NX,par)
    ngdphigh=Phigh*Yhigh+NX
    bhigh=max_b_y*Yhigh#ngdphigh
    par=Pars(par,bhigh=bhigh)
    GRIDS=CreateGrids(par)
    #Set bounds for optimization algorithm
    return GRIDS
end

function Setup(max_b_y::Float64,β::Float64,d0::Float64,d1::Float64,yC::Float64;η::Float64=0.83)
    par=Pars(β=β,d0=d0,d1=d1,yC=yC,η=η)
    GRIDS=Setup_GR(max_b_y,par)

    bhigh=GRIDS.GR_b[end]
    bhighOpt=bhigh+0.1
    par=Pars(par,bhigh=bhigh,bhighOpt=bhighOpt)

    return par, GRIDS
end

###############################################################################
#Function to compute consumption net of investment and adjustment cost
###############################################################################
function ConsNet(y::Float64,Λprime::Float64,Λ::Float64,par::Pars)
    #Resource constraint for the final good
    return y-CapitalAdjustment(Λprime,Λ,par)
end

function ComputeSpreadWithQ(qq::Float64,par::Pars)
    @unpack r_star, γ, κ = par
    ib=(((γ+(1-γ)*(κ+qq))/qq)^4)-1
    rf=((1+r_star)^4)-1
    return ib-rf
end

function SimpleSubsidy(qq::Float64,par::Pars)
    @unpack τs = par
    #Compute subsidies proportional to spreads
    spread=ComputeSpreadWithQ(qq,par)
    return 0.05+τs*spread
end

function Calculate_Tr(qq::Float64,b::Float64,bprime::Float64,par::Pars)
    @unpack γ, κ = par
    #Compute net borrowing from the rest of the world
    return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
end

###############################################################################
#Functions to compute value given state, policies, and guesses
###############################################################################
function ValueInDefault(I::CartesianIndex,Λprime::Float64,
                        SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, ΛlowOpt, ΛhighOpt, cmin = par
    @unpack GR_z, GR_p, GR_Λ = GRIDS
    @unpack itp_EV, itp_EVD = SOLUTION
    #Unpack state and compute output
    (Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; zD=zDefault(z,par); p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]
    y=FinalOutput(zD,p,Λ,0.0,par)

    #Compute consumption and value
    cons=ConsNet(y,Λprime,Λ,par)
    if cons>cmin
        return Utility(cons,par)+β*θ*min(0.0,itp_EV(0.0,Λprime,z,p_ind))+β*(1.0-θ)*min(0.0,itp_EVD(Λprime,z,p_ind))
    else
        return Utility(cmin,par)-CapitalAdjustment(Λprime,Λ,par)
    end
end

function ValueInRepayment(I::CartesianIndex,X_REAL::Array{Float64,1},
                          SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, ΛlowOpt, ΛhighOpt, blowOpt ,bhighOpt, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_Λ, GR_b, GR_z, GR_p = GRIDS

    #Unpack state
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]; b=GR_b[b_ind]

    #transform policy tries into interval
    Λprime=TransformIntoBounds(X_REAL[1],ΛlowOpt,ΛhighOpt)
    bprime=TransformIntoBounds(X_REAL[2],blowOpt,bhighOpt)

    #Compute output
    qq=itp_q1(bprime,Λprime,z,p_ind)
    T=Calculate_Tr(qq,b,bprime,par)
    aa=0.0
    if qq==0.0
        aa=-bprime
    end
    y=FinalOutput(z,p,Λ,T,par)
    #Compute consumption
    cons=ConsNet(y,Λprime,Λ,par)
    if y>0.0 && cons>cmin
        return Utility(cons,par)+β*min(0.0,itp_EV(bprime,Λprime,z,p_ind))+aa
    else
        return Utility(cmin,par)-CapitalAdjustment(Λprime,Λ,par)+aa
    end
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################
function OptimInDefault(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, MaxIter_Opt, g_tol_Opt, ΛlowOpt, ΛhighOpt = par
    @unpack GR_z, GR_p, GR_Λ = GRIDS
    #transform policy guess into reals
    (Λ_ind,z_ind,p_ind)=Tuple(I)

    #Setup function handle for optimization
    f(Λprime::Float64)=-ValueInDefault(I,Λprime,SOLUTION,GRIDS,par)

    #Perform optimization
    inner_optimizer = GoldenSection()
    res=optimize(f,ΛlowOpt,ΛhighOpt,inner_optimizer,abs_tol=1e-3)

    #Transform optimizer into bounds
    Λprime=TransformIntoBounds(Optim.minimizer(res),ΛlowOpt,ΛhighOpt)

    return -Optim.minimum(res), Λprime
end

function GridSearch_b(I::CartesianIndex,Λprime::Float64,
                      SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, ΛlowOpt, ΛhighOpt, blowOpt, bhighOpt = par
    @unpack GR_b = GRIDS

    X_REAL=Array{Float64,1}(undef,2)
    X_REAL[1]=TransformIntoReals(Λprime,ΛlowOpt,ΛhighOpt)

    val=-Inf
    bpol=0
    for btry in 1:par.Nb
        bprime=GR_b[btry]
        if bprime<blowOpt || bprime>bhighOpt
            println([btry bprime bhighOpt])
        end
        X_REAL[2]=TransformIntoReals(bprime,blowOpt,bhighOpt)
        #Compute value
        vv=ValueInRepayment(I,X_REAL,SOLUTION,GRIDS,par)
        if vv>val
            val=vv
            bpol=btry
        end
    end
    if bpol==0
        return b
    else
        return GR_b[bpol]
    end
end

function InitialPolicyGuess(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_Λ, GR_b, GR_z, GR_p = GRIDS
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    Λ=GR_Λ[Λ_ind]
    ΛΛ=Λ
    # b0=GridSearch_b(I,ΛΛ,SOLUTION,GRIDS,par)
    b0=GR_b[b_ind]
    return [ΛΛ, b0]
end

function OptimInRepayment(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, MaxIter_Opt, g_tol_Opt, Nb, ΛlowOpt, ΛhighOpt, blowOpt, bhighOpt = par
    @unpack GR_Λ, GR_b, GR_z, GR_p = GRIDS
    #Get initial guess
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)

    X0_BOUNDS=InitialPolicyGuess(I,SOLUTION,GRIDS,par)

    #transform policy guess into reals
    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],ΛlowOpt,ΛhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],blowOpt,bhighOpt)

    #Setup function handle for optimization
    f(X::Array{Float64,1})=-ValueInRepayment(I,X,SOLUTION,GRIDS,par)

    #Perform optimization with MatLab simplex
    inner_optimizer = NelderMead()
    res=optimize(f,X0,inner_optimizer,
        Optim.Options(x_tol = 1e-3))

    #Transform optimizer into bounds
    Λprime=TransformIntoBounds(Optim.minimizer(res)[1],ΛlowOpt,ΛhighOpt)
    bprime=TransformIntoBounds(Optim.minimizer(res)[2],blowOpt,bhighOpt)
    z=GR_z[z_ind]; b=GR_b[b_ind]
    qq=SOLUTION.itp_q1(bprime,Λprime,z,p_ind)
    Tr=Calculate_Tr(qq,b,bprime,par)
    return -Optim.minimum(res), Λprime, bprime, Tr
end

###############################################################################
#Update solution
###############################################################################
function UpdateDefault!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VD)
        SOLUTION.VD[I], SOLUTION.Λprime_D[I]=OptimInDefault(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_Λprime_D=CreateInterpolation_Policies(SOLUTION.Λprime_D,true,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(true,SOLUTION.VD,SOLUTION.EVD,GRIDS)
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function UpdateRepayment!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(SOLUTION.VP)
        (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
        SOLUTION.VP[I], SOLUTION.Λprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=OptimInRepayment(I,SOLUTION,GRIDS,par)
        if SOLUTION.VP[I]<SOLUTION.VD[Λ_ind,z_ind,p_ind]
            SOLUTION.V[I]=SOLUTION.VD[Λ_ind,z_ind,p_ind]
        else
            SOLUTION.V[I]=SOLUTION.VP[I]
        end
    end
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_Λprime=CreateInterpolation_Policies(SOLUTION.Λprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(false,SOLUTION.V,SOLUTION.EV,GRIDS)
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

###############################################################################
#Update price
###############################################################################
function BondsPayoff(pprime_ind::Int64,zprime::Float64,Λprime::Float64,
                     bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack itp_VP, itp_VD = SOLUTION
    if min(0.0,itp_VD(Λprime,zprime,pprime_ind))>min(0.0,itp_VP(bprime,Λprime,zprime,pprime_ind))
        return 0.0
    else
        @unpack γ, κ = par
        SDF=SDF_Lenders(par)
        if γ==1.0
            return SDF
        else
            @unpack itp_q1, itp_bprime, itp_Λprime = SOLUTION
            ΛΛ=itp_Λprime(bprime,Λprime,zprime,pprime_ind)
            bb=itp_bprime(bprime,Λprime,zprime,pprime_ind)
            return SDF*(γ+(1-γ)*(κ+itp_q1(bb,ΛΛ,zprime,pprime_ind)))
        end
    end
end

function IntegrateBondsPayoff(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_Λ, GR_b = GRIDS
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    Λprime=GR_Λ[Λ_ind]
    bprime=GR_b[b_ind]

    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack PI_pC, GR_p = GRIDS
    foo_q(pprime_ind::Int64,zprime::Float64)=BondsPayoff(pprime_ind,zprime,Λprime,bprime,SOLUTION,par)
    int=0.0
    for i in 1:length(GR_p)
        for j in 1:length(ϵz_weights)
            int=int+PI_pC[p_ind,i]*ϵz_weights[j]*PDFz[j]*foo_q(i,ZPRIME[z_ind,j])
        end
    end
    return int/FacQz
end

function UpdateBondsPrice!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to compute expectation over z' and p'
    for I in CartesianIndices(SOLUTION.q1)
        SOLUTION.q1[I]=IntegrateBondsPayoff(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    return nothing
end

###############################################################################
#Update and iniciate VFI algorithm
###############################################################################
function UpdateSolution!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(SOLUTION,GRIDS,par)
    UpdateRepayment!(SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    return nothing
end

function ComputeDistance_q(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    @unpack Tol_q = par
    dst_q=maximum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1))
    NotConv=sum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.q1)
    return round(dst_q,digits=5), round(NotConvPct,digits=2)
end

function ComputeDistanceV(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_D=maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD))
    dst_V=maximum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V))
    return round(abs(dst_D),digits=5), round(abs(dst_V),digits=5)
end

function SolveModel_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while (dst_V>Tol_V || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution!(SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        if PrintProg
            println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P, dst_q=$dst_q, $NotConvPct% of q not converged")
        end
    end
    return SOLUTION_NEXT
end

function SolveAndSaveModel(NAME::String,GRIDS::Grids,par::Pars)
    SOL=SolveModel_VFI(true,GRIDS,par)
    SaveSolution_Vector(NAME,SOL)
    return nothing
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
    Λ::Array{Float64,1}
    B::Array{Float64,1}

    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    nGDP::Array{Float64,1}
    P::Array{Float64,1}
    GDPdef::Array{Float64,1}
    Cons::Array{Float64,1}
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
    return Paths(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15)
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

function Simulate_pC_shocks(T::Int64,PI_pC::Array{Float64,2})
    X=Array{Int64,1}(undef,T)
    X[1]=1
    for t in 2:T
        X[t]=Draw_New_pC(X[t-1],PI_pC)
    end
    return X
end

function Simulate_AllShocks(T::Int64,GRIDS::Grids,par::Pars)
    z=Simulate_z_shocks(par.μ_z,T,GRIDS,par)
    p_ind=Simulate_pC_shocks(T,GRIDS.PI_pC)
    p=Array{Float64,1}(undef,T)
    for t in 1:T
        p[t]=GRIDS.GR_p[p_ind[t]]
    end
    return z, p_ind, p
end

function ComputeSpreads(z::Float64,p_ind::Int64,Λprime::Float64,bprime::Float64,
                        SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_q1 = SOLUTION
    q=max(itp_q1(bprime,Λprime,z,p_ind),1e-2)
    return 100.0*ComputeSpreadWithQ(q,par)
end

function GenerateNextState!(t::Int64,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_VD, itp_VP, itp_Λprime, itp_bprime = SOLUTION
    @unpack itp_Λprime_D = SOLUTION
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
    Λ=PATHS.Λ[t]; b=PATHS.B[t]

    #Update next endogenous state
    if d_==1.0
        #Coming from default state yesterday, must draw for readmission
        if rand()<=θ
            #Get readmitted, choose whether to default or not today
            if itp_VD(Λ,z,p_ind)<=itp_VP(b,Λ,z,p_ind)
                #Choose not to default
                PATHS.Def[t]=0.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill kN', kT' and b', and spreads
                    PATHS.Λ[t+1]=itp_Λprime(b,Λ,z,p_ind)
                    PATHS.B[t+1]=max(itp_bprime(b,Λ,z,p_ind),0.0)
                    PATHS.Spreads[t]=ComputeSpreads(z,p_ind,PATHS.Λ[t+1],PATHS.B[t+1],SOLUTION,GRIDS,par)
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
                    PATHS.Λ[t+1]=itp_Λprime_D(Λ,z,p_ind)
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
                PATHS.Λ[t+1]=itp_Λprime_D(Λ,z,p_ind)
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
        if itp_VD(Λ,z,p_ind)<=itp_VP(b,Λ,z,p_ind)
            #Choose not to default
            PATHS.Def[t]=0.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kN', kT' and b', and spreads
                PATHS.Λ[t+1]=itp_Λprime(b,Λ,z,p_ind)
                PATHS.B[t+1]=max(itp_bprime(b,Λ,z,p_ind),0.0)
                PATHS.Spreads[t]=ComputeSpreads(z,p_ind,PATHS.Λ[t+1],PATHS.B[t+1],SOLUTION,GRIDS,par)
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
                PATHS.Λ[t+1]=itp_Λprime_D(Λ,z,p_ind)
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
    return Paths(PATHS.z[t0:tT],PATHS.p_ind[t0:tT],PATHS.p[t0:tT],PATHS.Def[t0:tT],PATHS.Λ[t0:tT],PATHS.B[t0:tT],PATHS.Spreads[t0:tT],PATHS.GDP[t0:tT],PATHS.nGDP[t0:tT],PATHS.P[t0:tT],PATHS.GDPdef[t0:tT],PATHS.Cons[t0:tT],PATHS.TB[t0:tT],PATHS.CA[t0:tT],PATHS.RER[t0:tT])
end

function SimulatePathsOfStates(drp::Int64,z_TS::Array{Float64,1},p_ind_TS::Array{Int64,1},p_TS::Array{Float64,1},X0::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack μ_z = par
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
    PATHS.Λ[1]=X0[2]
    PATHS.B[1]=X0[3]
    for t in 1:drp+T
        GenerateNextState!(t,PATHS,SOLUTION,GRIDS,par)
    end
    return TimeIntervalFromPaths(PATHS,drp+1,drp+T)
end

function CalculateVariables_TS!(t::Int64,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ, η, ω, yC = par

    #Unpack current shocks
    zstate=PATHS.z[t]
    p_ind=PATHS.p_ind[t]
    p=PATHS.p[t]

    #Unpack current endogenous state
    d=PATHS.Def[t]; Λ=PATHS.Λ[t]; b=PATHS.B[t]
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
            @unpack itp_Λprime_D = SOLUTION
            Λprime=itp_Λprime_D(Λ,zstate,p_ind)
            bprime=0.0
            Tr=0.0
        else
            #in repayment
            @unpack itp_Λprime, itp_bprime, itp_q1 = SOLUTION
            Λprime=itp_Λprime(b,Λ,zstate,p_ind)
            bprime=itp_bprime(b,Λ,zstate,p_ind)
            qq=itp_q1(bprime,Λprime,zstate,p_ind)
            Tr=Calculate_Tr(qq,b,bprime,par)
        end
    else
        #Not end of time, use next state
        Λprime=PATHS.Λ[t+1]
        bprime=PATHS.B[t+1]
        if d==1
            #in default
            Tr=0.0
        else
            #in repayment
            @unpack itp_q1 = SOLUTION
            qq=itp_q1(bprime,Λprime,zstate,p_ind)
            Tr=Calculate_Tr(qq,b,bprime,par)
        end
    end

    #Store base prices
    if t==1
        #Use today's prices as base
        p0=p
        P0=PriceFinalGood(z,p,Λ,Tr,par)
        PATHS.P[t]=P0
    else
        p0=PATHS.p[1]
        P0=PATHS.P[1]
    end

    #Calculate current nominal and real variables
    Y=FinalOutput(z,p,Λ,Tr,par)
    P=PriceFinalGood(z,p,Λ,Tr,par)
    NX=-Tr
    PATHS.GDP[t]=P0*Y+NX+(p0-p)*yC
    PATHS.nGDP[t]=P*Y+NX
    PATHS.P[t]=P
    PATHS.GDPdef[t]=PATHS.nGDP[t]/PATHS.GDP[t]
    PATHS.Cons[t]=P0*Y+(p0-p)*yC
    PATHS.TB[t]=NX
    PATHS.CA[t]=-(bprime-b)
    PATHS.RER[t]=1/PATHS.GDPdef[t]
    return nothing
end

function UpdateOtherVariablesInPaths!(PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    for t=1:length(PATHS.z)
        CalculateVariables_TS!(t,PATHS,SOLUTION,GRIDS,par)
    end
end

function SimulatePaths(T::Int64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, pCL = par
    z_TS, p_ind_TS, p_TS=Simulate_AllShocks(drp+T,GRIDS,par)
    Λ0=Λsteady_state(pCL,par)
    B0=0.0
    Def0=0.0
    X0=[Def0; Λ0; B0]
    PATHS=SimulatePathsOfStates(drp,z_TS,p_ind_TS,p_TS,X0,SOLUTION,GRIDS,par)
    UpdateOtherVariablesInPaths!(PATHS,SOLUTION,GRIDS,par)
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
    #AR(1) GDP
    ρ_GDP::Float64 = 0.0
    σϵ_GDP::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
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

function ComputeMomentsOnce(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tmom, γ, κ, HPFilter_Par = par
    @unpack itp_q1, itp_Λprime, itp_bprime = SOLUTION
    @unpack itp_Λprime_D = SOLUTION
    PATHS=SimulatePaths(Tmom,SOLUTION,GRIDS,par)
    MOM=Moments()
    #Compute default probability
    PATHS_long=SimulatePaths(10000,SOLUTION,GRIDS,par)
    Def_Ev=0.0
    for t in 2:length(PATHS_long.Def)
        if PATHS_long.Def[t]==1.0 && PATHS_long.Def[t-1]==0.0
            Def_Ev=Def_Ev+1
        end
    end
    MOM.DefaultPr=100*Def_Ev/length(PATHS_long.Def)

    #Compute other easy moments
    MOM.MeanSpreads=sum((PATHS.Spreads) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    ex2=sum((PATHS.Spreads .^ 2) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    MOM.StdSpreads=sqrt(abs(ex2-(MOM.MeanSpreads^2)))
    #Compute stocks
    # MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (1 .* PATHS.nGDP))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    Youtput=(PATHS.nGDP .- PATHS.TB) ./ PATHS.P
    MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (1 .* Youtput))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    MOM.yC_GDP=mean((PATHS.p .* par.yC) ./ PATHS.nGDP)

    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(PATHS.GDP))
    GDP_trend=hp_filter(log_GDP,HPFilter_Par)
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
    con_trend=hp_filter(log_con,HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Trade balance
    # TB_y_TS=100 * PATHS.TB ./ PATHS.nGDP
    TB_y_TS=100 * PATHS.TB ./ Youtput
    #Real exchange rate
    log_rer=log.(abs.(PATHS.RER))
    rer_trend=hp_filter(log_rer,HPFilter_Par)
    rer_cyc=100.0*(log_rer .- rer_trend)
    #Volatilities
    MOM.σ_GDP=std(GDP_cyc)
    MOM.σ_con=std(con_cyc)
    MOM.σ_TB_y=std(TB_y_TS)
    MOM.σ_RER=std(rer_cyc)
    #Correlations with GDP
    MOM.Corr_con_GDP=cor(GDP_cyc,con_cyc)
    MOM.Corr_Spreads_GDP=cor(GDP_cyc,PATHS.Spreads)
    MOM.Corr_TB_GDP=cor(GDP_cyc,TB_y_TS)
    MOM.Corr_RER_GDP=cor(GDP_cyc,rer_cyc)
    return MOM
end

function AverageMomentsManySamples(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(123)
    #Initiate them at 0.0 to facilitate average across samples
    MOMENTS=Moments()
    for i in 1:NSamplesMoments
        MOMS=ComputeMomentsOnce(SOLUTION,GRIDS,par)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        #Stocks
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        MOMENTS.yC_GDP=MOMENTS.yC_GDP+MOMS.yC_GDP/NSamplesMoments
        #AR(1) GDP
        MOMENTS.ρ_GDP=MOMENTS.ρ_GDP+MOMS.ρ_GDP/NSamplesMoments
        MOMENTS.σϵ_GDP=MOMENTS.σϵ_GDP+MOMS.σϵ_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
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

function ComputeAndSaveMoments(NAME::String,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    MOMENTS=AverageMomentsManySamples(SOLUTION,GRIDS,par)
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
function RΛ_dec(Λprime::Float64,z::Float64,p::Float64,Λ::Float64,T::Float64,par::Pars)
    @unpack kbar = par
    #Return to capital in terms of the final consumption good
    rN=rN_dec(z,p,Λ,T,par)
    rT=rT_dec(z,Λ,par)
    P=PriceFinalGood(z,p,Λ,T,par)
    ψ2=dΨ_d2(Λprime,Λ,par)
    return (kbar*(rT-rN)/P)-ψ2
end

#Interpolation objects for households' FOCs
@with_kw mutable struct HH_itpObjects{T1,T2,T3,T4}
    #Auxiliary arrays for expectations
    dRΛ_Def::T1
    dRΛ_Rep::T2
    #Arrays of expectations
    ERΛ_Def::T1
    ERΛ_Rep::T2
    #Interpolation objects
    itp_dRΛ_Def::T3
    itp_dRΛ_Rep::T4
    itp_ERΛ_Def::T3
    itp_ERΛ_Rep::T4
end

function ComputeExpectations_HHDef!(itp_D,itp_R,E_MAT::Array{Float64,3},GRIDS::Grids,par::Pars)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack θ = par
    @unpack GR_b, GR_Λ = GRIDS
    for I in CartesianIndices(E_MAT)
        (Λ_ind,z_ind,p_ind)=Tuple(I)
        Λprime=GR_Λ[Λ_ind]
        function foo_mat(zprime::Float64,pprime_ind::Int64)
            return θ*itp_R(0.0,Λprime,zprime,pprime_ind)+(1-θ)*itp_D(Λprime,zprime,pprime_ind)
        end
        E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
    end
    return nothing
end

function ComputeExpectations_HHRep!(itp_D,itp_R,E_MAT::Array{Float64,4},SOLUTION::Solution,GRIDS::Grids)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack itp_VD, itp_VP = SOLUTION
    @unpack GR_b, GR_Λ = GRIDS
    for I in CartesianIndices(E_MAT)
        (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
        bprime=GR_b[b_ind]; Λprime=GR_Λ[Λ_ind]
        function foo_mat(zprime::Float64,pprime_ind::Int64)
            if itp_VD(Λprime,zprime,pprime_ind)>itp_VP(bprime,Λprime,zprime,pprime_ind)
                return itp_D(Λprime,zprime,pprime_ind)
            else
                return itp_R(bprime,Λprime,zprime,pprime_ind)
            end
        end
        E_MAT[I]=Expectation_over_shocks(foo_mat,p_ind,z_ind,GRIDS)
    end
    return nothing
end

function UpdateHH_Obj!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NΛ, Nb, cmin, β, θ = par
    @unpack GR_z, GR_p, GR_Λ, GR_b = GRIDS
    @unpack Λprime_D = SOLUTION
    @unpack Λprime, bprime, Tr = SOLUTION
    #Loop over all states to fill auxiliary matrices in default
    for I in CartesianIndices(HH_OBJ.dRΛ_Def)
        (Λ_ind,z_ind,p_ind)=Tuple(I)
        #Unpack shocks and policies
        z=GR_z[z_ind]; zD=zDefault(z,par)
        p=GR_p[p_ind]; Λ=GR_Λ[Λ_ind]
        Λprimef=Λprime_D[I]
        T=0.0
        #Compute consumption cD(yC',z',kN',kT')
        y=FinalOutput(zD,p,Λ,T,par)
        if y>0.0
            cons=max(ConsNet(y,Λ,Λprimef,par),cmin)
        else
            cons=cmin
        end

        #Fill matrices for expectations in default
        HH_OBJ.dRΛ_Def[I]=β*U_c(cons,par)*RΛ_dec(Λprimef,zD,p,Λ,T,par)
    end
    HH_OBJ.itp_dRΛ_Def=CreateInterpolation_ValueFunctions(HH_OBJ.dRΛ_Def,true,GRIDS)

    #Loop over all states to fill auxiliary matrices in repayment
    for I in CartesianIndices(HH_OBJ.dRΛ_Rep)
        (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
        #Unpack shocks and policies
        z=GR_z[z_ind]; p=GR_p[p_ind]
        Λ=GR_Λ[Λ_ind]; Λprimef=Λprime[I]
        T=Tr[I]
        y=FinalOutput(z,p,Λ,T,par)
        if y>0.0
            cons=max(ConsNet(y,Λ,Λprimef,par),cmin)
        else
            cons=cmin
        end

        #Fill matrices for expectations in repayment
        HH_OBJ.dRΛ_Rep[I]=β*U_c(cons,par)*RΛ_dec(Λprimef,z,p,Λ,T,par)
    end
    HH_OBJ.itp_dRΛ_Rep=CreateInterpolation_ValueFunctions(HH_OBJ.dRΛ_Rep,false,GRIDS)

    #Compute expectations in default
    ComputeExpectations_HHDef!(HH_OBJ.itp_dRΛ_Def,HH_OBJ.itp_dRΛ_Rep,HH_OBJ.ERΛ_Def,GRIDS,par)
    HH_OBJ.itp_ERΛ_Def=CreateInterpolation_ValueFunctions(HH_OBJ.ERΛ_Def,true,GRIDS)

    #Compute expectations in repayment
    ComputeExpectations_HHRep!(HH_OBJ.itp_dRΛ_Def,HH_OBJ.itp_dRΛ_Rep,HH_OBJ.ERΛ_Rep,SOLUTION,GRIDS)
    HH_OBJ.itp_ERΛ_Rep=CreateInterpolation_ValueFunctions(HH_OBJ.ERΛ_Rep,false,GRIDS)
    return nothing
end

function InitiateHH_Obj(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, Np, NΛ, Nb = par
    #Allocate empty arrays
    #Auxiliary arrays for expectations
    dRΛ_Def=zeros(Float64,NΛ,Nz,Np)
    dRΛ_Rep=zeros(Float64,Nb,NΛ,Nz,Np)
    #Arrays of expectations
    ERΛ_Def=zeros(Float64,NΛ,Nz,Np)
    ERΛ_Rep=zeros(Float64,Nb,NΛ,Nz,Np)
    #Create interpolation objects
    itp_dRΛ_Def=CreateInterpolation_ValueFunctions(ERΛ_Def,true,GRIDS)
    itp_dRΛ_Rep=CreateInterpolation_ValueFunctions(ERΛ_Rep,false,GRIDS)

    itp_ERΛ_Def=CreateInterpolation_ValueFunctions(ERΛ_Def,true,GRIDS)
    itp_ERΛ_Rep=CreateInterpolation_ValueFunctions(ERΛ_Rep,false,GRIDS)
    #Arrange objects in structure
    HH_OBJ=HH_itpObjects(dRΛ_Def,dRΛ_Rep,ERΛ_Def,ERΛ_Rep,itp_dRΛ_Def,itp_dRΛ_Rep,itp_ERΛ_Def,itp_ERΛ_Rep)
    #Modify values using SOLUTION at end of time
    UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    return HH_OBJ
end

#Functions to compute FOCs given (s,x) and a try of K'
function HH_FOC_Λ_Def(I::CartesianIndex,Λprime::Float64,HH_OBJ::HH_itpObjects,
                      SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack GR_z, GR_p, GR_Λ = GRIDS
    @unpack itp_ERΛ_Def = HH_OBJ
    #Compute present consumption
    (Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]
    zD=zDefault(z,par)
    T=0.0
    y=FinalOutput(zD,p,Λ,T,par)
    c=max(cmin,ConsNet(y,Λ,Λprime,par))

    # compute expectation over z and pC
    Ev=itp_ERΛ_Def(Λprime,z,p_ind)
    #compute extra term and return FOC
    ψ1=dΨ_d1(Λprime,Λ,par)
    return Ev-U_c(c,par)*ψ1
end

function HH_FOC_Λ_Rep(I::CartesianIndex,Λprime::Float64,bprime::Float64,
                      HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ, cmin, WithInvTax = par
    @unpack GR_z, GR_p, GR_Λ, GR_b = GRIDS
    @unpack itp_q1 = SOLUTION
    @unpack itp_ERΛ_Rep = HH_OBJ

    #Compute present consumption
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]; b=GR_b[b_ind]
    qq=itp_q1(bprime,Λprime,z,p_ind)
    T=Calculate_Tr(qq,b,bprime,par)

    #Compute subsidy
    if WithInvTax
        τs=SimpleSubsidy(qq,par)
    else
        τs=0.0
    end
    y=FinalOutput(z,p,Λ,T,par)
    c=max(cmin,ConsNet(y,Λ,Λprime,par))

    #compute expectation over z and yC
    Ev=itp_ERΛ_Rep(bprime,Λprime,z,p_ind)
    #compute extra term and return FOC
    ψ1=dΨ_d1(Λprime,Λ,par)
    return Ev-U_c(c,par)*(ψ1-τs)
end

#Functions to compute optimal policy from FOCs

function HHOptim_Def(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)

    fooΛ(Λprime::Float64)=HH_FOC_Λ_Def(I,Λprime,HH_OBJ,SOLUTION,GRIDS,par)
    ΛΛL=1e-2; ΛΛH=1-1e-2
    Λprime=MyBisection(fooΛ,ΛΛL,ΛΛH;xatol=1e-4)

    return Λprime
end

function HHOptim_Rep(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)

    fooΛ(Λprime::Float64)=HH_FOC_Λ_Rep(I,Λprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    ΛΛL=1e-2; ΛΛH=1-1e-2
    Λprime=MyBisection(fooΛ,ΛΛL,ΛΛH;xatol=1e-4)

    return Λprime
end

#Update equilibrium objects in decentralized economy
function ValueInDefault_DEC(I::CartesianIndex,Λprime::Float64,
                            SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, cmin = par
    @unpack itp_EV, itp_EVD = SOLUTION
    @unpack GR_Λ, GR_z, GR_p = GRIDS
    (Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]
    #Compute output, consumption and value
    zD=zDefault(z,par)
    y=FinalOutput(zD,p,Λ,0.0,par)
    cons=ConsNet(y,Λ,Λprime,par)
    if cons>0.0
        vv=Utility(cons,par)+β*θ*itp_EV(0.0,Λprime,z,p_ind)+β*(1.0-θ)*itp_EVD(Λprime,z,p_ind)
        if vv<0.0
            return vv
        else
            return Utility(cons,par)+β*θ*itp_EV(0.0,min(Λprime,GR_Λ[end]),z,p_ind)+β*(1.0-θ)*itp_EVD(min(Λprime,GR_Λ[end]),z,p_ind)
        end
    else
        return Utility(cmin,par)-(Λprime-Λ)^2
    end
end

function UpdateDefault_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VD)
        SOLUTION.Λprime_D[I]=HHOptim_Def(I,HH_OBJ,SOLUTION,GRIDS,par)
        SOLUTION.VD[I]=ValueInDefault_DEC(I,SOLUTION.Λprime_D[I],SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_Λprime_D=CreateInterpolation_Policies(SOLUTION.Λprime_D,true,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(true,SOLUTION.VD,SOLUTION.EVD,GRIDS)
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function ValueInRepayment_DEC(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_Λ, GR_z, GR_p, GR_b = GRIDS
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]; b=GR_b[b_ind]
    #Get HH's policy for bprime
    Λprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    #Compute output
    qq=itp_q1(bprime,Λprime,z,p_ind)
    T=Calculate_Tr(qq,b,bprime,par)
    aa=0.0
    if qq==0.0
        aa=-bprime
    end
    y=FinalOutput(z,p,Λ,T,par)
    #Compute consumption
    cons=ConsNet(y,Λ,Λprime,par)
    if y>0.0 && cons>cmin
        vv=Utility(cons,par)+β*itp_EV(bprime,Λprime,z,p_ind)+aa
        if vv<0.0
            return vv
        else
            return  Utility(cons,par)+β*itp_EV(max(bprime,0.0),min(Λprime,GR_Λ[end]),z,p_ind)+aa
        end
    else
        return Utility(cmin,par)-(Λprime-Λ)^2+aa
    end
end

function OptimInRepayment_DEC(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_Λ, GR_z, GR_p, GR_b = GRIDS
    (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
    z=GR_z[z_ind]; p=GR_p[p_ind]
    Λ=GR_Λ[Λ_ind]; b=GR_b[b_ind]

    blow=GR_b[1]
    bhigh=GR_b[end]
    foo(bprime::Float64)=-ValueInRepayment_DEC(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    res=optimize(foo,blow,bhigh,GoldenSection(),abs_tol=1e-3)
    vv=-Optim.minimum(res)
    bprime=Optim.minimizer(res)

    Λprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    return vv, Λprime, bprime
end

function UpdateRepayment_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_p, GR_b = GRIDS
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(SOLUTION.VP)
        SOLUTION.VP[I], SOLUTION.Λprime[I], SOLUTION.bprime[I]=OptimInRepayment_DEC(I,HH_OBJ,SOLUTION,GRIDS,par)
        (b_ind,Λ_ind,z_ind,p_ind)=Tuple(I)
        if SOLUTION.VP[I]<SOLUTION.VD[Λ_ind,z_ind,p_ind]
            SOLUTION.V[I]=SOLUTION.VD[Λ_ind,z_ind,p_ind]
        else
            SOLUTION.V[I]=SOLUTION.VP[I]
        end
        qq=SOLUTION.itp_q1(SOLUTION.bprime[I],SOLUTION.Λprime[I],GR_z[z_ind],p_ind)
        SOLUTION.Tr[I]=Calculate_Tr(qq,GR_b[b_ind],SOLUTION.bprime[I],par)
    end
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_Λprime=CreateInterpolation_Policies(SOLUTION.Λprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)

    #Compute expectation
    ComputeExpectationOverStates!(false,SOLUTION.V,SOLUTION.EV,GRIDS)
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

function UpdateSolution_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateRepayment_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

#Solve decentralized model
function SolutionEndOfTime(GRIDS::Grids,par::Pars)
    SOLUTION=InitiateEmptySolution(GRIDS,par)
    UpdateSolution!(SOLUTION,GRIDS,par)
    return SOLUTION
end

function SolveModel_VFI_DEC(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while (dst_V>Tol_V || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution_DEC!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        if PrintProg
            println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P, dst_q=$dst_q, $NotConvPct% of q not converged")
        end
    end
    return SOLUTION_NEXT
end

function SolveAndSaveModel_DEC(NAME::String,GRIDS::Grids,par::Pars)
    SOL=SolveModel_VFI_DEC(true,GRIDS,par)
    SaveSolution_Vector(NAME,SOL)
    return nothing
end

###############################################################################
#Functions to compute welfare
###############################################################################
function WelfareGains(z::Float64,p_ind::Int64,Λ::Float64,b::Float64,SOL0::Solution,SOL1::Solution,par::Pars)
    @unpack σ = par
    v0=SOL0.itp_V(b,Λ,z,p_ind)
    v1=SOL1.itp_V(b,Λ,z,p_ind)
    return 100*(((v1/v0)^(1/(1-σ)))-1)
end

function AverageWelfareGains(N::Int64,SOL0::Solution,SOL1::Solution,GRIDS::Grids,par::Pars)
    Random.seed!(1234)
    T=1000
    wg=0.0
    for i in 1:N
        TS=SimulatePaths(T,SOL0,GRIDS,par)
        cnt=1
        while TS.Def[end]==1
            #pick an ending point with no default
            TS=SimulatePaths(T,SOL0,GRIDS,par)
            cnt=cnt+1
            if cnt>10
                break
            end
        end
        z=TS.z[end]; p_ind=TS.p_ind[end]
        Λ=TS.Λ[end]; b=TS.B[end]
        wg=wg+WelfareGains(z,p_ind,Λ,b,SOL0,SOL1,par)/N
    end
    return wg
end

###############################################################################
#Functions to calibrate decentralized
###############################################################################
function SolveCalibration_DEC(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, Tolpct_q, cnt_max = par
    SOL=SolveModel_VFI_DEC(false,GRIDS,par)
    return AverageMomentsManySamples(SOL,GRIDS,par)
end

function CheckMomentsForTry_DEC(PARS_TRY::Array{Float64,1})
    β=PARS_TRY[1]
    knk=PARS_TRY[2]
    d1=PARS_TRY[3]
    d0=-d1*knk
    yC=PARS_TRY[4]
    par, GRIDS=Setup(0.35,β,d0,d1,yC)

    MOM=SolveCalibration_DEC(GRIDS,par)

    MOM_VEC=Array{Float64,1}(undef,15)

    MOM_VEC[1]=MOM.DefaultPr
    MOM_VEC[2]=MOM.MeanSpreads
    MOM_VEC[3]=MOM.StdSpreads
    MOM_VEC[4]=MOM.Debt_GDP
    MOM_VEC[5]=MOM.yC_GDP
    MOM_VEC[6]=MOM.σ_GDP
    MOM_VEC[7]=MOM.σ_con/MOM.σ_GDP
    MOM_VEC[8]=MOM.σ_TB_y
    MOM_VEC[9]=MOM.σ_RER
    MOM_VEC[10]=MOM.Corr_con_GDP
    MOM_VEC[11]=MOM.Corr_Spreads_GDP
    MOM_VEC[12]=MOM.Corr_TB_GDP
    MOM_VEC[13]=MOM.Corr_RER_GDP
    MOM_VEC[14]=MOM.ρ_GDP
    MOM_VEC[15]=MOM.σϵ_GDP

    return MOM_VEC
end

function CalibrateMatchingMoments_DEC(N::Int64,lb::Vector{Float64},ub::Vector{Float64},yC::Float64)
    #Here N is the number of grid points for each parameter
    #Total computations will be N^2
    pp=Pars()
    nn=N^2
    MAT_TRY=Array{Float64,2}(undef,nn,4)
    gr_knk=collect(range(lb[1],stop=ub[1],length=N))
    gr_d1=collect(range(lb[2],stop=ub[2],length=N))
    i=1
    for d1_ind in 1:N
        for knk_ind in 1:N
            MAT_TRY[i,1]=pp.β
            MAT_TRY[i,2]=gr_knk[knk_ind]
            MAT_TRY[i,3]=gr_d1[d1_ind]
            MAT_TRY[i,4]=yC
            i=i+1
        end
    end

    #Loop paralelly over all parameter tries
    #There are 15 moments plus dst plus dst_ar, columns should be 15+4 parameters
    PARAMETER_MOMENTS_MATRIX=SharedArray{Float64,2}(nn,19)
    COL_NAMES=["beta" "knk" "d1" "yC" "DefPr" "Av spreads" "Std spreads" "debt_GDP" "yC_GDP" "vol_y" "vol_c/vol_y" "vol_tb" "vol_rer" "cor(c,y)" "cor(r-r*,y)" "cor(tb/y,y)" "cor(rer,y)" "rho_gdp" "sigmaEps_gdp"]
    @sync @distributed for i in 1:nn
        PARAMETER_MOMENTS_MATRIX[i,1:4]=MAT_TRY[i,:]
        PARAMETER_MOMENTS_MATRIX[i,5:19]=CheckMomentsForTry_DEC(MAT_TRY[i,:])
        MAT=[COL_NAMES; PARAMETER_MOMENTS_MATRIX]
        writedlm("TriedCalibrations.csv",MAT,',')
        println("Done with $i of $nn")
    end
    return nothing
end

###############################################################################
#Analyze optimal taxes
###############################################################################
function StateContingentSubsidies(z::Float64,p_ind::Int64,p::Float64,Λ::Float64,b::Float64,SOL_PLA::Solution,par::Pars)
    @unpack γ = par
    @unpack itp_q1, itp_Λprime, itp_bprime = SOL_PLA
    hh=1e-6
    ΛΛ=itp_Λprime(b,Λ,z,p_ind)
    bb=itp_bprime(b,Λ,z,p_ind)
    dq_dΛ=(itp_q1(bb,ΛΛ+hh,z,p_ind)-itp_q1(bb,ΛΛ-hh,z,p_ind))/(2*hh)

    qq=itp_q1(bb,ΛΛ,z,p_ind)

    T=Calculate_Tr(qq,b,bb,par)
    P=PriceFinalGood(z,p,Λ,T,par)

    τs=dq_dΛ*(bb-(1-γ)*b)/P

    return 100*τs
end

function TimeSeriesOfSubsidies(PATHS::Paths,SOL_PLA::Solution,par::Pars)
    T=length(PATHS.z)
    τs_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        z=PATHS.z[t]; p=PATHS.p[t]; p_ind=PATHS.p_ind[t]
        Λ=PATHS.Λ[t]; b=PATHS.B[t]
        τs_TS[t]=StateContingentSubsidies(z,p_ind,p,Λ,b,SOL_PLA,par)
    end
    return τs_TS
end

###############################################################################
#Functions to find best tax
###############################################################################
function TryDifferentSubsidies(τ_GRID::Array{Float64,1},GRIDS::Grids,par::Pars)
    # SOL_DEC=UnpackSolution_Vector(NAME," ",GRIDS,par)
    SOL_DEC=SolveModel_VFI_DEC(true,GRIDS,par)
    Nτ=length(τ_GRID)
    Ng=10000

    MAT_GAINS=SharedArray{Float64,2}(Nτ,2)
    @sync @distributed for i in 1:Nτ
        if i==1
            #Initial in grid, τ=0 is the benchmark
            par_τ=Pars(par,WithInvTax=true,τs=τ_GRID[i])
            SOL_τ=SolveModel_VFI_DEC(true,GRIDS,par_τ)
            # SOL_τ=deepcopy(SOL_DEC)
        else
            par_τ=Pars(par,WithInvTax=true,τs=τ_GRID[i])
            SOL_τ=SolveModel_VFI_DEC(true,GRIDS,par_τ)
        end
        MAT_GAINS[i,1]=τ_GRID[i]
        MAT_GAINS[i,2]=AverageWelfareGains(Ng,SOL_DEC,SOL_τ,GRIDS,par)
        writedlm("TAX_GAINS.csv",MAT_GAINS,',')
        println("Done with $i of $Nτ")
    end

    return nothing
end
