module Hamiltonians


using LinearAlgebra
using ITensors
using ITensorMPS
using ProgressMeter
import TensorCrossInterpolation as TCI
import QuanticsTCI as QTCI
using Quantics

# Load your module (must be in the same directory)
include("QuantumKPM.jl")
include("kin_builders.jl")
# include("exciton_builders.jl")
using .QuantumKPM
using .kin_builders
# using .exciton_builders


########################
# Model registry + builder
########################

# Simple param parser: "k=v,k2=v2" -> Dict{Symbol,Any}
# (bool/int/float detection; else keep as String)
"""
    _parse_param_string(s::AbstractString) -> Dict{Symbol,Any}

Parses a parameter string of the form `"key1=val1, key2=val2, ..."` into a dictionary mapping symbols to values.

# Arguments
- `s::AbstractString`: The input string containing key-value pairs separated by commas, spaces, or tabs. Each key-value pair should be in the form `key=value`.

# Returns
- `Dict{Symbol,Any}`: A dictionary where keys are symbols and values are parsed as `Bool`, `Int`, `Float64`, or left as `String` if they do not match those types.

# Parsing Rules
- Boolean values: `"true"` or `"false"` (case-insensitive) are parsed as `Bool`.
- Integer values: Strings matching an integer pattern are parsed as `Int`.
- Floating-point values: Strings matching a float pattern (including scientific notation) are parsed as `Float64`.
- All other values are kept as `String`.

# Errors
Throws an error if a token does not match the expected `key=value` format.
"""
function _parse_param_string(s::AbstractString)
    d = Dict{Symbol,Any}()
    t = strip(s)
    isempty(t) && return d
    for tok in split(t, [',',' ','\t'])
        isempty(tok) && continue
        kv = split(tok, '=', limit=2)
        length(kv) == 2 || error("Bad model param token: '$tok' (expected key=value)")
        k = Symbol(strip(kv[1])); v = strip(kv[2])
        vl = lowercase(v)
        val::Any =
            vl in ("true","false") ? (vl == "true") :
            occursin(r"^[+-]?\d+$", v) ? parse(Int, v) :
            occursin(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", v) ? parse(Float64, v) :
            v
        d[k] = val
    end
    return d
end

# Registry maps a model name to:
#   (function symbol, dimension, required positional arg names in order,
#    known keyword defaults, interacting::Bool)
const MODEL_REGISTRY = Dict{String,Tuple{Symbol,Int,Vector{Symbol},NamedTuple,Bool}}(
    # --- Non-interacting Hamiltonians ---
    "qc2dsquare"       => (:HQC2Dsquare,       2, [:t],           (; tol_quantics=1e-9, maxbonddim_quantics=250, cutoff=1e-10), false),
    # --- Interacting Hamiltonians ---
    "intqc1d"          => (:HIntQC1D,          1, [:U],           (; t=1.0, max_iter=100, Nmu=60, threshold=5e-3,
                                                                     Egrid=nothing, Efermi=0.0, mix=0.8,
                                                                     tol_quantics=1e-12, maxbonddim_quantics=100,
                                                                     apply_kwargs=(; cutoff=1e-10, maxdim=100),
                                                                     verbose=true, domainwall=false, gpu = false), true),
)

# Helper to check interaction
is_interacting(model::AbstractString) = MODEL_REGISTRY[lowercase(model)][5]


"""
    build_hamiltonian(model::AbstractString, L::Integer; mparams="", mparam_dict=Dict()) -> MPO

Build a Hamiltonian MPO by *model name*, validating required model-specific parameters.

Arguments
---------
- `model`      : one of keys in MODEL_REGISTRY (e.g. "aah", "ssh", "uniform")
- `L`          : log of number of sites (Qubits), i.e. `N=2^L`
- `mparams`    : (optional) string `"k=v,k2=v2"` for model-specific params
- `mparam_dict`: (optional) Dict{Symbol,Any} with model params (merged after `mparams`)

Returns
-------
- `H::MPO`

Notes
-----
- Required params are enforced. Any extra keys are passed as keywords.
- Positional arg order comes from the registry entry.
"""
# Common helper (don’t export)
function _build_hamiltonian_impl(model::AbstractString; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    haskey(MODEL_REGISTRY, lowercase(model)) || error("Unknown model '$model'. Known: $(collect(keys(MODEL_REGISTRY)))")
    return MODEL_REGISTRY[lowercase(model)]
end

# 1D signature: pass L positionally
function build_hamiltonian(model::AbstractString, L::Integer; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    fn_sym, dim, required_syms, kw_defaults = _build_hamiltonian_impl(model; mparams, mparam_dict)
    dim == 1 || error("Model '$model' expects 2D sizes (Lx, Ly). Use the 2D method build_hamiltonian(model, Lx, Ly; ...).")
    fn = getfield(@__MODULE__, fn_sym)

    # Merge params: string first, then dict overrides
    p = _parse_param_string(mparams)
    for (k,v) in mparam_dict
        p[k] = v
    end

    # Check required (positional-after-L)
    missing = [k for k in required_syms if !haskey(p, k)]
    !isempty(missing) && error("Missing required params for '$model': $(missing). Provided: $(collect(keys(p))).")

    pos = [p[k] for k in required_syms]  # e.g. [:t, :d] -> [t, d]
    extra = Dict(k=>v for (k,v) in p if !(k in required_syms))
    kw_final = (; kw_defaults..., extra...)

    return fn(L, pos...; kw_final...)
end

# 2D signature: pass (Lx, Ly) positionally
function build_hamiltonian(model::AbstractString, Lx::Integer, Ly::Integer; mparams::AbstractString="", mparam_dict=Dict{Symbol,Any}())
    fn_sym, dim, required_syms, kw_defaults = _build_hamiltonian_impl(model; mparams, mparam_dict)
    dim == 2 || error("Model '$model' expects 1D size L. Use the 1D method build_hamiltonian(model, L; ...).")
    fn = getfield(@__MODULE__, fn_sym)

    # Merge params: string first, then dict overrides
    p = _parse_param_string(mparams)
    for (k,v) in mparam_dict
        p[k] = v
    end

    # Check required (positional-after-Lx,Ly)
    missing = [k for k in required_syms if !haskey(p, k)]
    !isempty(missing) && error("Missing required params for '$model': $(missing). Provided: $(collect(keys(p))).")

    pos = [p[k] for k in required_syms]  # e.g. [:t] -> [t]
    extra = Dict(k=>v for (k,v) in p if !(k in required_syms))
    kw_final = (; kw_defaults..., extra...)

    return fn(Lx, Ly, pos...; kw_final...)
end


export build_hamiltonian


function qtt_mpo(L, xvals, sites, func; tol_quantics=1e-8, maxbonddim_quantics=50) 
    qtt  = QTCI.quanticscrossinterpolate(ComplexF64, func, xvals; tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    tt   = TCI.tensortrain(qtt.tci)
    mps  = MPS(tt; sites)
    mpo  = outer(mps', mps)
    for s in 1:L
        mpo.data[s] = Quantics._asdiagonal(mps.data[s], sites[s])
    end
    return mpo
end

# -------------------------- Model Hamiltonians -------------------------


function HQC2Dsquare(Lx::Integer, Ly::Integer, t::Real = 1.0;
                  tol_quantics::Real = 1e-8,
                  maxbonddim_quantics::Integer = 100, 
                  cutoff::Real = 1e-10) 

    # System sizes: Nx, Ny sites in each direction (powers of 2)
    Nx, Ny = 2^Lx, 2^Ly
    L      = Lx + Ly           # number of qubits in TT representation
    N      = Nx * Ny           # total number of physical sites

    sites = siteinds("Qubit", L)
    xvals = 0:N-1               # linearized 2D grid indices

        # --- 8-fold modulation term on the 2D coordinates
    function func8fold(x, y, V;  Nx=Nx)
        # 4 vectors: (2π/a) * e_x, e_y, and their 45° rotation
        a = 1          # lattice constant for 8-fold modulation

        b1 = (5*sqrt(5)*a/2) # atomic scale wavevector
        b2 = (sqrt(3)*(Nx*a/16))  # superlattice scale wavevector

        Ka1 = 2π .* ( [1.0, 0.0] )
        Kb1 = 2π .* ( [0.0, 1.0] )
        tht   = deg2rad(45.0)
        Rt  = [cos(tht)  sin(tht);
                -sin(tht) cos(tht)]
        Ka2 = Rt * Ka1
        Kb2 = Rt * Kb1
        K   = (Ka1, Kb1, Ka2, Kb2)
        xy  = [x - Nx/2, y - Nx/2] # the offset is to obtain a nice symmetric pattern (assumes Nx=Ny)
        
        cosines = 0.0
        for k in K
            cosines += (2.5*cos(dot(k, xy)/b1) + cos(dot(k, xy)/b2)) #same as yitao
        end
        return V * (1 + 0.1 * cosines)

    end

    # Wrappers that evaluate f at the bond midpoints on square lattice (for hoppings)
    wrap2D_mid_x(f, Nx) = i -> begin
        x = i % Nx
        y = i ÷ Nx
        f(x + 0.5, y)          # midpoint between (x,y) and (x+1,y)
    end

    wrap2D_mid_y(f, Nx) = i -> begin
        x = i % Nx
        y = i ÷ Nx
        f(x, y + 0.5)          # midpoint between (x,y) and (x,y+1)
    end

    # Intra-row and inter-row nearest-neighbor hoppings at midpoints
    intra_row_hop = wrap2D_mid_x((x, y) -> func8fold(x, y, t; Nx=Nx), Nx)
    inter_row_hop = wrap2D_mid_y((x, y) -> func8fold(x, y, t; Nx=Nx), Nx)


    # Quantics cross interpolation of the hopping fields
    hops_MPOintra = qtt_mpo(L, xvals, sites, intra_row_hop; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPOinter = qtt_mpo(L, xvals, sites, inter_row_hop; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    # Build kinetic Hamiltonians
    Hintra = kin_builders.kineticintra2DNNN(Lx, Ly, sites, hops_MPOintra, 1)
    Hinter = kin_builders.kineticNNN(L,    sites, hops_MPOinter, Nx)

    # Total non-interacting Hamiltonian
    H0 = +(Hinter,  Hintra;  cutoff = cutoff)

    return H0
end


# -------------------------- Mean Field Interacting Hamiltonian ------------------------

# ----------- initial guesses for SCF ------------

# Antiferromagnetic guess 1D for Hubbard mean field

function initial_guess_trivial_up_1D(L, sites)
    
    xvals =  range(0, (2^L - 1); length=2^L)
    f(x) =  ((x )%2)  
    qtt = QTCI.quanticscrossinterpolate(
        Float64, f, xvals,
        maxbonddim = 10; tolerance = 1e-8
    )[1]
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt; sites = sites)
    density_mpo = outer(density_mps', density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return density_mpo, density_mps
end

function initial_guess_trivial_down_1D(L, sites)
    
    xvals =  range(0, (2^L - 1); length=2^L)
    f(x) =  ((x + 1)%2)  
    qtt = QTCI.quanticscrossinterpolate(
        Float64, f, xvals,
        maxbonddim = 10; tolerance = 1e-8
    )[1]
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt; sites = sites)
    density_mpo = outer(density_mps', density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return density_mpo, density_mps
end



"""
    HIntQC1D(L; t=1.0, U=0.0,
             k1 = 2π/(5*sqrt(2)),
             k2 = 2π/(sqrt(3)*(2^L)/2),
             max_iter=100, Nmu=100, threshold=1e-4,
             Egrid=nothing, Efermi=0.0, mix=0.5,
             tol_quantics=1e-8, maxbonddim_quantics=50,
             apply_kwargs=(; cutoff=1e-10, maxdim=100),
             verbose=true, domainwall=false)

Build 1D Hamiltonian with mid-bond modulated hopping. If `U>0`, run SCF (with
`U::Real`) to produce mean-field spin-resolved Hamiltonians. If `U==0`, skip SCF
and return `H_up = H_dn = H0`.

# Arguments
- `L::Integer`: Log₂ of the number of sites along x  
  (system size is `N = 2^L`).
- `U::Real`: On-site Hubbard interaction (mean-field level). If `U==0`,
  returns only the kinetic Hamiltonian.
- `t::Real`: Base hopping amplitude.
- `max_iter::Integer`: Maximum number of SCF iterations.
- `Nmu::Integer`: Number of Chebyshev moments used in KPM.
- `threshold::Real`: Convergence threshold for SCF.
- `Egrid`: Energy grid for KPM (if `nothing`, defaults to `[-4,4]`).
- `Efermi::Real`: Fermi energy for filling.
- `mix::Real`: Mixing factor in SCF updates (0 < mix ≤ 1).
- `tol_quantics::Real`: Accuracy tolerance in Quantics interpolation.
- `maxbonddim_quantics::Integer`: Maximum bond dimension for Quantics TT.
- `apply_kwargs`: Keyword arguments passed to MPO applications 
  (e.g. `cutoff`, `maxdim`).
- `verbose::Bool`: If true, prints SCF progress.
- `domainwall::Bool`: If true, modifies buckling wavevector to simulate
  a domain wall along x.

# Returns
Named tuple with:
- `H_up`, `H_dn`: Spin-resolved mean-field Hamiltonians (MPO).
- `H0`: Non-interacting kinetic Hamiltonian (MPO).
- `rho_up`, `rho_dn`: Self-consistent mean-field densities (MPO), or
  `nothing` if `U==0`.

# Notes
- The hopping modulation is defined by a function of the bond 
  center.
- Initial guesses for the SCF are antiferromagnetic.
- Kinetic MPOs are built using `kin_builders.jl`.
- Interaction is included at the mean-field level via Chebychev Tensor Network calculations (see QuantumKPM.jl).
"""
function HIntQC1D(L::Integer, U::Real = 0.0;
                  t::Real = 1.0,
                  max_iter::Integer = 100,
                  Nmu::Integer = 100,
                  threshold::Real = 1e-3,
                  Egrid = nothing,
                  Efermi::Real = 0.0,
                  mix::Real = 0.9,
                  tol_quantics::Real = 1e-12,
                  maxbonddim_quantics::Integer = 100,
                  apply_kwargs = (; cutoff=1e-12, maxdim=100),
                  verbose::Bool = true, Tn_nonint::Bool = true,
                  domainwall::Bool = false, gpu::Bool = false)

    # --- sites and grid
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:(N-1)

    # --- mid-bond modulated hopping: evaluate at x + 0.5
    function mod_hop(x; t0=t, N=N, domainwall=domainwall)
        # b1 = sqrt(5)/2*(1 + 0*x/N)  #linear variation along x 
        b1 = 3*sqrt(5)/2  # no variation as QTCI has troubles with changes at this scale
        k1 = 2*pi/b1
        b2= sqrt(3)*(N)/15*(1 + 1*x/N)  #linear variation along x large scale
        k2 = 2*pi/b2
        if domainwall == true
            Xdw = N/2
            W = 1/sqrt(N)
            k2 *= (1+0.3*tanh((x+0.5-Xdw)/W))
        end

        tt = t0*(1 + 0.75*x/N)  #linear variation of t along x (using lam*sin(x/lam) approx x to bypass quantics issues)
    return tt*(1 + 0.2*(cos(k1*(x + 0.5)) + cos(k2*(x + 0.5))))
    end
    # --- Quantics MPO -------------------
    hops_MPO = qtt_mpo(L, xvals, sites, x -> mod_hop(x; t0=t, N=N, domainwall=domainwall);
        tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    # --- kinetic term -------------------
    H0 = kin_builders.kineticNNN(L, sites, hops_MPO, 1)

    # --- no interaction: return early
    if U == 0
        return (H_up = H0, H_dn = H0, H0 = H0,
                rho_up = nothing, rho_dn = nothing)
    end

    # --- antiferromagnetic initial guesses you provided
    mpo_guess_up, mps_guess_up = initial_guess_trivial_up_1D(L, sites)
    mpo_guess_dn, mps_guess_dn = initial_guess_trivial_down_1D(L, sites)

    # --- KPM energy grid
    Ewidth = 4*2.5*t + U + 1 # rough estimate of full bandwidth with interactions and modulation :4*max(t) + U + 1 (+1 for safety)
    Egrid_vec = if Egrid === nothing
        [-Ewidth/2, Ewidth/2]
    else
        collect(range(float(first(Egrid)), float(last(Egrid)); length = 2))
    end

    println("Calculating mean-field densities with U = $U ...")
    # --- run SCF with U::Real (no U_MPO)
    scf = QuantumKPM.SCF_Hubbard1D(
        H0, U,                         # << scalar U here
        max_iter, Nmu, threshold,
        mpo_guess_up, mpo_guess_dn,
        mps_guess_up, mps_guess_dn,
        Egrid_vec, Efermi;
        mix = mix,
        apply_kwargs = apply_kwargs,
        verbose = verbose, Tn_nonint = Tn_nonint, gpu = gpu,
    )

    # densities as MPOs for operator algebra
    rho_up = scf.den_mpo_up
    rho_dn = scf.den_mpo_dn

    # The Tn lists 
    Tn_up = scf.Tn_up
    Tn_dn = scf.Tn_dn   
    if Tn_nonint == true   
        Tn_0 = scf.Tn_0
    end

    # --- spin-resolved mean-field Hamiltonians:
    # H_up = H0 + U * rho_dn - (U/(2L)) * Id,  etc.
    Id_op = MPO(sites, "Id")
    H_up = +(H0, U * rho_dn; apply_kwargs...)
    H_up = +(H_up, - U * Id_op / 2; apply_kwargs...)
    H_dn = +(H0, U * rho_up, - U * Id_op / 2; apply_kwargs...)

    if Tn_nonint == false
        return (H_up = H_up, H_dn = H_dn, H0 = H0,
            rho_up = rho_up, rho_dn = rho_dn,
            Tn_up = Tn_up, Tn_dn = Tn_dn)
    else
        return (H_up = H_up, H_dn = H_dn, H0 = H0,
        rho_up = rho_up, rho_dn = rho_dn,
        Tn_up = Tn_up, Tn_dn = Tn_dn, Tn_0 = Tn_0)
    end

end


end # module Hamiltonians