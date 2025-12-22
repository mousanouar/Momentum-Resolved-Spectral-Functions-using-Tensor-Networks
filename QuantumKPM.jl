module QuantumKPM

"""
QuantumKPM — KPM/TCI utilities for ITensor/Quantics workflows

Author: [Anouar Moustaj]; Date: 03/09/2025; Version: 0.1
=============================================================

This module gathers small, practical tools to build spectral quantities with the
**Kernel Polynomial Method (KPM)** on ITensor **MPO/MPS** objects, and to
interoperate with **Quantics / Tensor Cross Interpolation (TCI)** tensor-trains.
It also provides helpers to move operators to **momentum space** via a Quantics
**quantum Fourier transform (QFT)** and to extract dense representations when
needed.

"""

# ---------------------------------------------------------------------
# Dependencies 
# ---------------------------------------------------------------------
import TensorCrossInterpolation as TCI
import QuanticsTCI as QTCI
import QuanticsGrids as QG
using ProgressMeter
using ITensors 
using ITensorMPS
using ITensorMPS: MPO, MPS, OpSum, expect, inner, siteinds
using LinearAlgebra
using QuanticsTCI
using Quantics
using Statistics


# Load your module (must be in the same directory)
include("kin_builders.jl")
using .kin_builders


# ---------------------------------------------------------------------
# Operator definitions 
# ---------------------------------------------------------------------

"""
ITensors operator for σ⁺ on a Qubit site.
"""
ITensors.op(::OpName"sigma_plus",::SiteType"Qubit") =
 [0 1
  0 0]

"""
ITensors operator for σ⁻ on a Qubit site.
"""
ITensors.op(::OpName"sigma_minus",::SiteType"Qubit") =
 [0 0
  1 0]

# ---------------------------------------------------------------------
# Utilities 
# ---------------------------------------------------------------------


function to_binary_vector(n, size)
    # Convert the integer n to a binary string (without leading zeros)
    binary_str = string(n, base=2)
    
    # Pad the binary string with leading zeros on the left
    # so that its length matches the desired size
    padded_binary_str = lpad(binary_str, size, '0')
    
    # Convert the padded string into a vector of characters,
    # then map each character to a proper String ("0" or "1")
    return collect(padded_binary_str) |> x -> map(s -> string(s), x)
end

function qtt_mpo(L, xvals, sites, func; initpivot=nothing, tol_quantics=1e-8, maxbonddim_quantics=50) 
    if initpivot == nothing
        qtt  = QTCI.quanticscrossinterpolate(ComplexF64, func, xvals; tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    else
        qtt  = QTCI.quanticscrossinterpolate(ComplexF64, func, xvals, initpivot; tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    end
    
    tt   = TCI.tensortrain(qtt.tci)
    mps  = MPS(tt; sites)
    mpo  = outer(mps', mps)
    for s in 1:L
        mpo.data[s] = Quantics._asdiagonal(mps.data[s], sites[s])
    end
    return mpo
end


function get_dens_x_quantics(qtt, num_x, num_spl, L; D::Int = 1)

    # ---------------- offset helpers ----------------

    @inline function shift_with_fallback(u::Int, du::Int, Umax::Int)
        u_try = u + du
        if 0 ≤ u_try ≤ Umax
            return u_try
        end
        u_try = u - du
        if 0 ≤ u_try ≤ Umax
            return u_try
        end
        return clamp(u, 0, Umax)
    end

    function make_scaled_offsets(step_size::Real, maxd::Int)
        while 2 * maxd + 1 > 2 * step_size
            # @warn "Number of samples per point ($(2*maxd+1)) exceeds step size ($step_size). Reducing `maxd` by half."
            maxd = div(maxd, 2)
        end
        if maxd < 1
            # @warn "maxd < 1; no offsets needed."
            return Int[0], Int[0]
        else
            offs = round.(Int, range(-step_size, step_size; length = 2 * maxd + 1))
            pos  = [o for o in offs if o > 0]
            neg  = [o for o in offs if o < 0]
            return pos, neg
        end
    end

    function make_offset_idx_vectors(xs::Vector{Int}, Lx::Int, Nx::Int,
                                     x_pos_offs::Vector{Int}, x_neg_offs::Vector{Int})
        idxvecs = Dict{Symbol, Vector{Int}}()

        idxvecs[:base] = xs

        for (k, off) in enumerate(x_pos_offs)
            idxvecs[Symbol("x_p", k)] =
                vec([shift_with_fallback(x, off, Nx - 1) for x in xs])
        end

        for (k, off) in enumerate(x_neg_offs)
            idxvecs[Symbol("x_m", k)] =
                vec([shift_with_fallback(x, off, Nx - 1) for x in xs])
        end

        return idxvecs
    end

    # Sample some points using quantics:

    # ==================================================================
    #                           2D CASE : still needs to be made with average sampling
    # ================================================================== 


    function ilinspace(xmin, xmax, num_x::Int)
        xvals = xmin:xmax
        _N = length(xvals)
        @assert 1 ≤ num_x ≤ _N
        num_x == 1 && return [0]
        step = (_N - 1) ÷ (num_x - 1)
        return collect(xmin:step:(xmin+step*(num_x-1)))
    end

    if D == 2
        Lx = div(L,2)
        Ly = div(L,2)
        Nx = 2^Lx
        Ny = 2^Ly

        # Iterate over some x to sample the spectral function A(x)
        # Step sizes between sample centers divided by 2 (half-step sizes
        # because we want to avoid overlap)
        step_size_x = div(Nx, num_x) / 2

        # Distance offsets: number of samples per point
        maxd = num_spl

        # Sampling positions in 1D
        xs = ilinspace(0, Nx-1, num_x)     # length == num_x
        ys = ilinspace(0, Ny-1, num_x)     # length == num_x

        # Positive and negative offsets
        x_pos_offs, x_neg_offs = make_scaled_offsets(step_size_x, maxd)
        # Build index vectors for base and offset points
        idxvecs_x = make_offset_idx_vectors(xs, Lx, Nx, x_pos_offs, x_neg_offs)
        idxvecs_y = make_offset_idx_vectors(ys, Ly, Ny, x_pos_offs, x_neg_offs)

        # Evaluate f(x) on a given index vector
        eval_on = v -> [qtt.(i) for i in v]

        idx_names_x = collect(keys(idxvecs_x))
        idx_names_y = collect(keys(idxvecs_y))

        sort!(idx_names_x, by = String)
        sort!(idx_names_y, by = String)

        vals = []

        for nx in idx_names_x, ny in idx_names_y
            idx = vec([ (y << Lx) | x for y in idxvecs_y[ny], x in idxvecs_x[nx] ])  # linear indices of all combinations
            push!(vals, eval_on(idx))
        end

        # Average over offsets and return as a 1D vector (to be reshaped later)
        return mean(reduce(hcat, vals), dims = 2)[:]

        ## 1D CASE
    elseif D == 1
        Lx = L
        Nx = 2^Lx

        @assert 1 ≤ num_x ≤ Nx

        # Step sizes between sample centers divided by 2 (half-step sizes
        # because we want to avoid too much overlap)
        step_size_x = div(Nx, num_x) / 2

        # Distance offsets: number of samples per point
        maxd = num_spl

        # Sampling positions in 1D
        xs = ilinspace(0, Nx-1, num_x)

        # Positive and negative offsets
        x_pos_offs, x_neg_offs = make_scaled_offsets(step_size_x, maxd)

        # Build index vectors for base and offset points
        idxvecs = make_offset_idx_vectors(xs, Lx, Nx, x_pos_offs, x_neg_offs)

        # Evaluate f(x) on a given index vector
        eval_on = v -> [qtt.(i) for i in v]

        idx_names = collect(keys(idxvecs))
        sort!(idx_names, by = String)

        vals = [eval_on(idxvecs[k]) for k in idx_names]

        # Average over offsets and return as a 1D vector
        return mean(reduce(hcat, vals), dims = 2)[:]
    end
end


# ---------------------------------------------------------------------
# KPM / Chebyshev 
# ---------------------------------------------------------------------

"""
    KPM_Tn(H, N, E) -> Vector{MPO}

Kernel Polynomial Method (KPM): build the sequence of Chebyshev polynomials
`[T₀(Ĥ), T₁(Ĥ), …, T_{N-1}(Ĥ)]` as **MPOs** for a Hamiltonian `H` that is
first **normalized** to lie approximately in `[-1, 1]` over the target energy
window defined by `E`.

Arguments
---------
- `H`     : Hamiltonian as an MPO on `L` sites.
- `N`     : number of Chebyshev polynomials to generate (moments).
- `E`     : energy grid (physical units). Only `minimum(E)` and `maximum(E)`
            are used to compute the shift `a` and half-width `W2` for normalizing `H`.

Algorithm (high-level)
------
1. Build identity MPO: `Id_op = MPO(sites,"Id")`.
2. Normalize `H` to `Ĥ` with spectrum ~ `[-1+err, 1-err]` using
      `a  = (E_max + E_min)/2`, `W2 = (E_max - E_min)/2`, and a small
      `ε = 1e-2` to avoid touching ±1:
      `Ĥ = (H - a * Id) / (W2 + ε)`.
3. Initialize Chebyshev seeds:
      `T₀ = Id`, `T₁ = Ĥ`.
4. Use the Chebyshev recurrence (as MPO algebra):
      `T_k = 2 * Ĥ * T_{k-1} - T_{k-2}`.
   Composition is done with `apply(Ĥ, T_{k-1}; cutoff=...)` and the sum
   with `+( ... ; maxdim=...)`, followed by `ITensorMPS.truncate!` to control
   bond growth.

Returns
-------
- `Tn_list :: Vector{MPO}` containing `[T₀, T₁, …, T_{N-1}]`.

Notes
-----
- Truncation parameters (`cutoff`, `maxdim`) are set to modest values inside
  the recurrence; tune as needed for accuracy vs. cost.
"""
function KPM_Tn(H::MPO, N::Int64, E; cutoff=1e-9, maxdim=200)
     # Kernel Polynomial Method for computing the Tn polynomials of the Hamiltonian H
     # H is an MPO, N is the number of Tn polynomials to compute

    apply_kwargs = (cutoff=cutoff, maxdim=maxdim)
    sites = getindex.(siteinds(H), 2)
    L = length(H)  # Number of spins 
    Id_op = MPO(sites, "Id")

    # --- Normalize H using the provided energy window E ---
    e  = 1e-2                              # small epsilon: keeps spec away from ±1
    W2 = (maximum(E) - minimum(E)) / 2     # half-width of the energy range
    a  = (maximum(E) + minimum(E)) / 2     # center of the energy range
    # Shift & rescale so Ĥ ≈ (H - a I)/(W2 + e) has spectrum in [-1, 1]
    Ham_n = (H - a*Id_op) / (W2 + e)
    # Ham_n = H/10


    # --- Chebyshev seeds ---
    T_k_minus_2 = Id_op  # T₀
    T_k_minus_1 = Ham_n  # T₁
    Tn_list = [T_k_minus_2, T_k_minus_1]

    # --- Recurrence: T_k = 2 Ĥ T_{k-1} - T_{k-2} ---
    for k in 3:N
        # Apply Ĥ to T_{k-1} and form the recurrence with truncation
        T_k = +(2 * apply(Ham_n, T_k_minus_1; apply_kwargs...),
                -T_k_minus_2; apply_kwargs...)
        T_k = ITensorMPS.truncate!(T_k; apply_kwargs...)

        # Shift the window and store T_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        push!(Tn_list, T_k)

    end
    return Tn_list
end



function get_DOS_from_Tn(Tn_list,Nmu,E) #needs normalized energies E # output needs to be divided by the amount of sites N=2^L to get the actual density of states

    # Jackson damping kernel coefficients g_n, n = 0:(Nmu-1)
    jackson_kernel = [(Nmu - n ) * cos(π * n / (Nmu)) + sin(π * n / (Nmu)) / tan(π / (Nmu)) for n in 0:Nmu-1]/Nmu

    # Chebyshev basis factor T_{n-1}(E) = cos((n-1) * arccos(E))
    function G_n(n)
        return cos((n-1)*acos(E))
    end

    # Chebyshev/Jackson expansion of the DOS operator A(E)
    A = Tn_list[1] * G_n(1) * jackson_kernel[1]      # T₀ term (no factor 2)
    for n in 2:Nmu
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n] ; cutoff = 1e-8, maxdim=100) # This is the Chebyshev sum with factor 2 for n≥2
        A = ITensorMPS.truncate!(A; cutoff=1e-8)       # keep MPO manageable
    end

    # Kernel polynomial normalization 
    A /= (π * sqrt(1-E^2)) # Normalization of the density MPO

    return  A, tr(A)
end

#calculation of zero-T density matrix for fermions at filling given by Efermi 
function get_dens_mat_from_Tn(Tn_list,N,Efermi=0)
    # Jackson damping kernel coefficients g_n, n = 0:(Nmu-1)
    jackson_kernel = [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]

    function G_n(n)
        if n == 1
            return acos(Efermi) # T₀(E) = 1, integral is ∫(1/√(1-E^2)) dE = arcsin(E) = π - arccos(E) so what's up? ----> when computing dens quantics we do 1-inner so it works out
        else
            return sin((n-1) * acos(Efermi)) / (n-1) 
        end
    end
    # # Chebyshev-Jackson expansion of the density matrix operator 
    A = Tn_list[1] * G_n(1) * jackson_kernel[1] 
    for n in 2:N
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n] ; cutoff = 1e-8, maxdim=100)
        A = ITensorMPS.truncate!(A; cutoff=1e-8)
    end

    # Kernel polynomial normalization (without sqrt(1-E^2) factor as it is integrated out for the density matrix)
    A /= (π* N)  # Normalization of the density MPO
    
    return  A
end

function dens_mat_quantics(A,L)
    sites = getindex.(siteinds(A), 2)  # Extract site indices from the MPO
    # Total number of sites is N = 2^L; we work on the integer grid 0..N-1
    xvals = range(0, (2^L - 1); length = 2^L)
    # Define f(x) = ⟨x|A|x⟩ using computational-basis states |x⟩ as MPS.
    # `to_binary_vector(Int(x), L)` returns the bitstring for x (length L).
    f(x) = 1-inner(MPS(sites, to_binary_vector(Int(x), L))', A,
                 MPS(sites, to_binary_vector(Int(x), L)))

    # Build a Quantics (TT) approximation of f over the 2^L-point grid.
    # Using Float64 assumes f(x) is real-valued; change to ComplexF64 if needed.
    qtt = quanticscrossinterpolate(ComplexF64, f, xvals; tolerance = 1e-8)[1]

    # TT -> MPS -> MPO
    A_tt   = TCI.tensortrain(qtt.tci)
    A_MPS = MPS(A_tt; sites)
    A_MPO = outer(A_MPS', A_MPS)
    for i in 1:L
        A_MPO.data[i] = Quantics._asdiagonal(A_MPS.data[i], sites[i])
    end

    return qtt, A_MPS, A_MPO
  end



  # -----------------------SCMF loop-------------------------------

  """
    SCF_Hubbard1D(H0, U, max_iter, Nmom, threshold,
                  mpo_guess_up, mpo_guess_dn,
                  mps_guess_up, mps_guess_dn,
                  Egrid, Efermi;
                  mix=0.5, apply_kwargs=(;cutoff=1e-8, maxdim=200), verbose=true)

Self-consistent mean-field loop for a 1D Hubbard-like Hamiltonian split into
spin-up and spin-down blocks.

Mean-field form:
- H_up   = H0 + U * rho_dn - 0.5 * U
- H_down = H0 + U * rho_up - 0.5 * U

Inputs
------
- H0::MPO, U::MPO
- max_iter::Int                : max number of SCF iterations
- Nmom::Int                    : number of KPM Chebyshev moments
- threshold::Real              : convergence tolerance for relative density change
- mpo_guess_up/dn::MPO         : initial density MPO guesses (up/down)
- mps_guess_up/dn::MPS         : initial density MPS guesses (up/down)
- Egrid::AbstractVector{<:Real}: KPM energy grid (as required by QuantumKPM.KPM_Tn)
- Efermi::Real                 : Fermi level for building density matrix

Keywords
--------
- mix::Real=0.5                       : linear mixing parameter in [0,1]
- apply_kwargs=(;cutoff=1e-8, maxdim=200)  : forwarded to `apply` and MPO sums
- verbose::Bool=true                  : print iteration progress

Returns
-------
NamedTuple(
  qtt_den_up, qtt_den_dn,          # Quantics TT objects (from your dens_mat_quantics)
  den_mpo_up, den_mpo_dn,          # MPO densities
  den_mps_up, den_mps_dn,          # MPS densities
  Tn_up, Tn_dn,                    # KPM moment lists
  history,                         # Vector of convergence metrics per iteration
  converged::Bool,
  iters::Int
)

Notes
-----
- Convergence metric = 0.5 * ( ||Δrho_up|| + ||Δrho_dn|| )
- Assumes your `QuantumKPM` module provides: KPM_Tn, get_dens_mat_from_Tn, dens_mat_quantics.
"""
function SCF_Hubbard1D(H0::MPO,
                       U,
                       max_iter::Integer,
                       Nmom::Integer,
                       threshold::Real,
                       mpo_guess_up::MPO,
                       mpo_guess_dn::MPO,
                       mps_guess_up::MPS,
                       mps_guess_dn::MPS,
                       Egrid::AbstractVector{<:Real},
                       Efermi::Real;
                       mix::Real = 0.5,
                       apply_kwargs = (; cutoff=1e-8, maxdim=100),
                       Tn_nonint::Bool = true,
                       verbose::Bool = true, gpu::Bool = false)

    L = length(H0)
    sites = getindex.(siteinds(H0), 2)  # Extract site indices from the MPO
    Id_op = MPO(sites, "Id")
    ctf, md = apply_kwargs.cutoff, apply_kwargs.maxdim
    # Initial mean-field Hamiltonians
    if U isa Real
        U_rho_up_ini = U * mpo_guess_up
        U_rho_dn_ini = U * mpo_guess_dn
        H_up = +(H0, U_rho_dn_ini, - U/2 * Id_op;  apply_kwargs...)
        H_dn = +(H0, U_rho_up_ini, - U/2 * Id_op;  apply_kwargs...)
    elseif U isa MPO
        U_rho_up_ini = apply(U, mpo_guess_up; cutoff=ctf, maxdim=md)
        U_rho_dn_ini = apply(U, mpo_guess_dn; cutoff=ctf, maxdim=md)
        H_up = +(H0, U_rho_dn_ini, - U/2 ;  apply_kwargs...)
        H_dn = +(H0, U_rho_up_ini, - U/2 ;  apply_kwargs...)
    else
        error("Unsupported type for U: $(typeof(U))")
    end

    #I need to add some keyword that allows the fastest calculation of the interacting Hamiltonian
    #This would entail that I just need to calculate the Tn_up list each time, as rho_dn is just the identity matrix minus rho_up (for half filling)

    if Tn_nonint == true
        if gpu == true
            Tn_0 = QuantumKPM.KPM_Tn_gpu(H0, Nmom, Egrid; cutoff=ctf, maxdim=md)
        elseif gpu == false
            Tn_0 = QuantumKPM.KPM_Tn(H0, Nmom, Egrid; cutoff=ctf, maxdim=md)
        end
    end


    # Working copies of density guesses
    rho_mpo_up = mpo_guess_up
    rho_mpo_dn = mpo_guess_dn
    rho_mps_up = mps_guess_up
    rho_mps_dn = mps_guess_dn

    history = Float64[]
 
    # # Variables to return even if not converged

    for it in 1:max_iter
        # KPM moments
        if gpu == true
            Tn_up = QuantumKPM.KPM_Tn_gpu(H_up, Nmom, Egrid; cutoff=ctf, maxdim=md)
            Tn_dn = QuantumKPM.KPM_Tn_gpu(H_dn, Nmom, Egrid; cutoff=ctf, maxdim=md)
        else
            
            Tn_up = QuantumKPM.KPM_Tn(H_up, Nmom, Egrid; cutoff=ctf, maxdim=md)
            Tn_dn = QuantumKPM.KPM_Tn(H_dn, Nmom, Egrid; cutoff=ctf, maxdim=md)
        end

        # Densities from moments at Efermi
        A_up = QuantumKPM.get_dens_mat_from_Tn(Tn_up, Nmom, Efermi)
        A_dn = QuantumKPM.get_dens_mat_from_Tn(Tn_dn, Nmom, Efermi)
        # A_dn = +(Id_op, -A_up; apply_kwargs...)  # rho_dn = I - rho_up at half-filling ==> not working... why? 
        # Convert to Quantics TT / MPO / MPS (your helper returns all three)
        qtt_up, mps_up, mpo_up = QuantumKPM.dens_mat_quantics(A_up, L)
        qtt_dn, mps_dn, mpo_dn = QuantumKPM.dens_mat_quantics(A_dn, L)

        # Convergence metric (relative change in MPS)
        r_up = norm(mps_up - rho_mps_up)/norm(Id_op)
        r_dn = norm(mps_dn - rho_mps_dn)/norm(Id_op)
        metric = 0.5 * (r_up + r_dn)
        push!(history, metric)
        if verbose && it % 5 == 0
            @info "SCF iter $it: rel_change=$(metric)"
        end
        if metric < threshold
            verbose && @info "SCF converged in $it iterations."
            # Tn_dn = QuantumKPM.KPM_Tn(H_dn, Nmom, Egrid; cutoff=ctf, maxdim=md)
            #I can also just output the Hamiltonians here without having to reconstruct them outside this function
            if Tn_nonint == true
                return (
                        den_mpo_up = mpo_up, den_mpo_dn = mpo_dn,
                        den_mps_up = mps_up, den_mps_dn = mps_dn,
                        Tn_up = Tn_up, Tn_dn = Tn_dn, Tn_0 = Tn_0,
                        history = history, converged = true, iters = it)
            else 
                return (
                        den_mpo_up = mpo_up, den_mpo_dn = mpo_dn,
                        den_mps_up = mps_up, den_mps_dn = mps_dn,
                        Tn_up = Tn_up, Tn_dn = Tn_dn,
                        history = history, converged = true, iters = it)
            end
        end

        # Linear mixing
        rho_mpo_up_new = mix * mpo_up + (1 - mix) * rho_mpo_up
        rho_mpo_dn_new = mix * mpo_dn + (1 - mix) * rho_mpo_dn
        rho_mps_up_new = mix * mps_up + (1 - mix) * rho_mps_up
        rho_mps_dn_new = mix * mps_dn + (1 - mix) * rho_mps_dn

        # Update mean-field Hamiltonians for next iteration

        if U isa Real
          U_rho_up = U*rho_mpo_up_new
          U_rho_dn = U*rho_mpo_dn_new
          H_up = +(H0, U_rho_dn, - U/2 * Id_op; apply_kwargs...)  # up depends on down
          H_dn = +(H0, U_rho_up, - U/2 * Id_op; apply_kwargs...)  # down depends on up
        elseif U isa MPO
          U_rho_up = apply(U, rho_mpo_up_new; cutoff=ctf, maxdim=md)
          U_rho_dn = apply(U, rho_mpo_dn_new; cutoff=ctf, maxdim=md)
          H_up = +(H0, U_rho_dn, - U/2; apply_kwargs...)  # up depends on down
          H_dn = +(H0, U_rho_up, - U/2; apply_kwargs...)  # down depends on up
          # println("The down matrix is" get_matrix(H_dn, 2^L))
        end
        
        # Commit new guesses
        rho_mpo_up, rho_mpo_dn = rho_mpo_up_new, rho_mpo_dn_new
        rho_mps_up, rho_mps_dn = rho_mps_up_new, rho_mps_dn_new

    end

    @warn "SCF loop did not converge in $(max_iter) iterations (final metric=$(last(history)))"
    
    return (
            den_mpo_up = rho_mpo_up, den_mpo_dn = rho_mpo_dn,
            den_mps_up = rho_mps_up, den_mps_dn = rho_mps_dn,
            history = history, converged = false, iters = max_iter)

end




# ---------------------------------------------------------------------
# Momentum space transformations
# ---------------------------------------------------------------------


"""
    conjugate_by_qft(W; tol=1e-15, maxdim=100) -> TensorTrain{<:Complex,4}

Return the **TT-operator** representing the unitary conjugation
`U * W * U^{-1}`, where `U` is the (Quantics) quantum Fourier transform (QFT).

Arguments
---------
- `W`        : an operator on `R` qubits (typically an `MPO`). It will be
               converted to a TT-operator internally.
- `tol`      : truncation / compression tolerance used by TCI `contract`.
- `maxdim`   : maximum TT bond dimension allowed during contractions.

Algorithm (high level)
--------
1. Extract the **unprimed** site indices from `W` to determine `R` (number of qubits).
2. Build the **forward** QFT TT-operator `FT` (with `sign = -1`) and the
   **inverse** QFT TT-operator `FTi` (with `sign = +1`) using QuanticsTCI.
3. Convert `W` to a **TT-operator** (`TensorTrain{ComplexF64,4}`).
4. Perform the unitary sandwich in TT form:
      `ttkspace = FT * W`  and  `W2 = FTi * ttkspace`
   (implemented as TCI `contract`, with `reverse` calls to align core order).
5. Return `W2`, the **conjugated operator** as a TT-operator.

"""
function conjugate_by_qft(W; tol=1e-9, maxdim::Int=100)
    # Determine the system size (number of qubits R) from W's site indices.
    # For an MPO, siteinds(W) returns pairs like (prime(site), site); getindex.(..., 2)
    # picks the **unprimed** (physical) sites.
    sites = getindex.(siteinds(W), 2)
    R = length(sites)

    # Build Quantics TT-operators for the QFT and its inverse.
    # sign = -1 → forward transform U ; sign = +1 → inverse U^{-1}
    FT  = QTCI.quanticsfouriermpo(R; sign = -1.0, normalize = true)
    FTi = QTCI.quanticsfouriermpo(R; sign = +1.0, normalize = true)

    # Convert the input operator W (an MPO) into a TT-operator (order-4 cores).
    Wtt = TCI.TensorTrain{ComplexF64, 4}(W)
    # println("Max bond dim of MPO before conjugation: ", TCI.rank(Wtt))

    # Perform the sandwich U * W * U^{-1} in TT space.

    # The TCI.reverse calls are used to match the core ordering expected by `contract`.
    ttkspace = TCI.contract(TCI.reverse(Wtt), FT;  algorithm = :naive,
                            tolerance = tol, maxbonddim = maxdim)
    # println("Max bond dim after first operation F*MPO: ", TCI.rank(ttkspace))
    W2 = TCI.contract(TCI.reverse(FTi), ttkspace; algorithm = :naive,
                      tolerance = tol, maxbonddim = maxdim)
    # println("Max bond dim of Fourier transformed F*MPO*Finv: ", TCI.rank(W2))

    return W2
end


function QFTMPO(O::MPO; tol=1e-9, maxdim::Int=100)
    """Transform the MPO O to momentum space. conjugate_by_qft gives a tensor train, which is then converted to an MPO
       indices of OkMPO are arbitrarily assigned, so we replace them with those of O"""
    OkMPO = MPO(conjugate_by_qft(O; tol=tol, maxdim=maxdim)) #Transform the MPO O to momentum space. 
    oldsites = getindex.(siteinds(OkMPO),2) #indices of OkMPO are arbitrarily assigned, so we replace them with those of O
    oldsitesprime = getindex.(siteinds(OkMPO),1)
    newsites = getindex.(siteinds(O),2)
    for i in 1:length(OkMPO)
        OkMPO[i] = OkMPO[i] * delta(oldsites[i], newsites[i]') * delta(oldsitesprime[i], newsites[i])
    end
    return OkMPO
end



function get_spect_k_quantics(deltaoperator, num_k, D::Int = 1)
    # num_k around 100 for D=1 and 70 for D=2 should be fine for most purposes of plotting the spectral function
    # Conjugate real-space delta operator into k-space: Akop = U * deltaoperator * U†
    Akop = QFTMPO(deltaoperator; tol=1e-12, maxdim=250)

    siteskspace = getindex.(siteinds(Akop), 2) #k-space sites  

    # Number of spins/qubits and (recomputed) total sites N = 2^L
    L = length(siteskspace)
    N = 2^L  # number of sites in k-space

    # Diagonal element at momentum k: ⟨k| Akop |k⟩
    f2(k) = inner(
        MPS(siteskspace, to_binary_vector(Int(k), L))',
        Akop,
        MPS(siteskspace, to_binary_vector(Int(k), L)),
    )

    # Sample some k-points in the Brillouin zone using quantics:

    # ==================================================================
    #                           2D CASE 
    # ================================================================== 


    function ilinspace(N::Int, num_x::Int)
        @assert 1 ≤ num_x ≤ N
        num_x == 1 && return [0]
        step = (N - 1) ÷ (num_x - 1)
        return collect(0:step:(step*(num_x-1)))
    end

    if D == 2
        Lx = div(L,2)
        Ly = div(L,2)
        Nx = 2^Lx
        Ny = 2^Ly

        @assert 1 < num_k ≤ N "num_k ($num_k) must be ≤ Nxy ($Nx) to sample unique k-states."

        # Iterate over some momenta k to sample the spectral function A(k)

        kxs = ilinspace(Nx, num_k)     # length == num_k
        kys = ilinspace(Ny, num_k)     # length == num_k

        kidx = vec([ (ky << Lx) | kx for (kx, ky) in zip(kxs, kys) ])  # linear indices of kx=ky  (using zip will give diagonal directly: faster if wanted)
        # kidx = vec([ (ky << Lx) | kx for ky in kys, kx in kxs ])  # linear indices of all momenta combinations

        kvals = 0:N-1

        qtt = QTCI.quanticscrossinterpolate(ComplexF64, f2, kvals; tolerance = 1e-8)[1]

        Ak = qtt.(kidx)  #maybe averaging can be done here as well?
        return real.(Ak)
    elseif D == 1
        Lx = L
        Nx = 2^Lx

        @assert 1 ≤ num_k ≤ Nx "num_k ($num_k) must be in [1, N ($Nx)] to sample unique k-states."

        # Iterate over some momenta k to sample the spectral function A(k)

        kidx = ilinspace(Nx, num_k)     # length == num_k

        kvals = 0:Nx-1
        qtt = QTCI.quanticscrossinterpolate(ComplexF64, f2, kvals; tolerance = 1e-8)[1]

        Ak = qtt.(kidx)  #maybe averaging can be done here as well?
        return real.(Ak)
    end
end


"""
    get_spect_k(deltaoperator, siteskspace) -> Vector

Compute the **momentum-resolved spectral function** for **all momenta** `k = 0, …, N-1`.

Arguments
---------
- `deltaoperator` : an MPO representing the δ-like DOS operator in **real space**.
- `siteskspace`   : vector of unprimed site indices that define the **k-space** ordering.
- `N`             : total number of lattice sites. **Convention:** `N = 2^L`, where
                    `L` is the number of spins/qubits.
                    (Note: inside the function, `N` is recomputed from `L`.)

Algorithm (high level)
----------------------
1. **QFT conjugation:** build `Akop = U * deltaoperator * U†` using the Quantics QFT MPO
   (via `conjugate_by_qft`) to map the operator to momentum space.
2. **Momentum basis states:** for each `k`, form the computational-basis MPS `|k⟩` by
   writing `k` in binary over `L` qubits (with `to_binary_vector`).
3. **Apply operator:** convert `|k⟩` to a TT-vector and contract with the TT-operator
   `Akop` to get `Akop|k⟩`.
4. **Diagonal element:** compute the scalar `⟨k| Akop |k⟩` and store it.
5. **Collect:** return the length-`N` vector `[⟨k| Akop |k⟩]_{k=0}^{N-1}`.


"""
function get_spect_k(deltaoperator, num_k, D::Int = 1; num_spl::Int = 10, averaging::Bool = false, quantics_sampling::Bool = false)

    if quantics_sampling == true
        return get_spect_k_quantics(deltaoperator, num_k, D)
    end

    # num_k around 100 for D=1 and 70 for D=2 should be fine for most purposes of plotting the spectral function
    # Conjugate real-space delta operator into k-space: Akop = U * deltaoperator * U†
    Akop = QFTMPO(deltaoperator; tol=1e-12, maxdim=250)

    siteskspace = getindex.(siteinds(Akop), 2) #k-space sites  

    # Number of spins/qubits and (recomputed) total sites N = 2^L
    L = length(siteskspace)
    N = 2^L  # number of sites in k-space

    # Diagonal element at momentum k: ⟨k| Akop |k⟩
    f2(k) = inner(
        MPS(siteskspace, to_binary_vector(Int(k), L))',
        Akop,
        MPS(siteskspace, to_binary_vector(Int(k), L)),
    )

    # Sample some k-points uniformly in the Brillouin zone: ###### todo: to make sampling better, do random sampling around the points defined by ilinspace, e.g. pm 2% of the distance to the next point

    function ilinspace(N::Int, num_x::Int)
        @assert 1 ≤ num_x ≤ N
        num_x == 1 && return [0]
        step = (N - 1) ÷ (num_x - 1)
        return collect(0:step:(step*(num_x-1)))
    end

    # ---------------- offset helpers ----------------

    @inline function shift_with_fallback(u::Int, du::Int, Umax::Int)
        u_try = u + du
        if 0 ≤ u_try ≤ Umax
            return u_try
        end
        u_try = u - du
        if 0 ≤ u_try ≤ Umax
            return u_try
        end
        return clamp(u, 0, Umax)
    end

    function make_scaled_offsets(step_size::Real, maxd::Int)
        while  maxd > step_size
            @warn "distance between sampling box boundaries ($(2*maxd)) exceeds step size ($(2*step_size)). Reducing `maxd` by half."
            maxd = div(maxd, 2)
        end 
        if maxd < 1
            @warn "maxd < 1; no offsets needed."
            return Int[0], Int[0]
        else
            offs = round.(Int, range(-step_size, step_size; length = 2 * maxd + 1))
            pos  = [o for o in offs if o > 0]
            neg  = [o for o in offs if o < 0]
            return pos, neg
        end
    end

    function make_offset_idx_vectors(xs::Vector{Int}, Lx::Int, Nx::Int,
                                     x_pos_offs::Vector{Int}, x_neg_offs::Vector{Int})
        idxvecs = Dict{Symbol, Vector{Int}}()

        idxvecs[:base] = xs

        for (k, off) in enumerate(x_pos_offs)
            idxvecs[Symbol("x_p", k)] =
                vec([shift_with_fallback(x, off, Nx - 1) for x in xs])
        end

        for (k, off) in enumerate(x_neg_offs)
            idxvecs[Symbol("x_m", k)] =
                vec([shift_with_fallback(x, off, Nx - 1) for x in xs])
        end

        return idxvecs
    end


    # ==================================================================
    #                           2D CASE 
    # ================================================================== 


    if D == 2
        Lx = div(L,2)
        Ly = div(L,2)
        Nx = 2^Lx
        Ny = 2^Ly

        @assert 1 < num_k ≤ N "num_k ($num_k) must be ≤ Nxy ($Nx) to sample unique k-states."

        if averaging == false
            # Iterate over some momenta k to sample the spectral function A(k)

            kxs = ilinspace(Nx, num_k)     # length == num_k
            kys = ilinspace(Ny, num_k)     # length == num_k

            kidx = vec([ (ky << Lx) | kx for (kx, ky) in zip(kxs, kys) ])  # linear indices of kx=ky  (using zip will give diagonal directly: faster if wanted)
            # kidx = vec([ (ky << Lx) | kx for ky in kys, kx in kxs ])  # linear indices of all momenta combinations
        
            # # diagonal k: kx = kys[i], ky = kxs[i] 
            # kidx = [ (kys[i] << Lx) | kxs[i] for i in eachindex(kxs)]

            Ak = [f2(k) for k in kidx]
            return Ak
        elseif averaging == true
            # Step sizes between sample centers divided by 2 (half-step sizes
            # because we want to avoid overlap)
            step_size_k = div(Nx, num_k) / 2

            # Distance offsets: number of samples per point
            maxd = num_spl

            # Sampling positions in 1D
            kxs = ilinspace(Nx, num_k)  # length == num_k
            kys = ilinspace(Ny, num_k)  # length == num_k

            # Positive and negative offsets
            k_pos_offs, k_neg_offs = make_scaled_offsets(step_size_k, maxd)

            # Build index vectors for base and offset points
            idxvecs_kx = make_offset_idx_vectors(kxs, Lx, Nx, k_pos_offs, k_neg_offs)
            idxvecs_ky = make_offset_idx_vectors(kys, Ly, Ny, k_pos_offs, k_neg_offs)

            # Evaluate f(x) on a given index vector
            eval_on = v -> [f2(i) for i in v]

            idx_names_kx = collect(keys(idxvecs_kx))
            idx_names_ky = collect(keys(idxvecs_ky))

            sort!(idx_names_kx, by = String)
            sort!(idx_names_ky, by = String)

            vals = []

            for nx in idx_names_kx, ny in idx_names_ky
                kidx = vec([ (ky << Lx) | kx for (kx, ky) in zip(idxvecs_ky[ny], idxvecs_kx[nx]) ])  # linear indices of kx=ky  (using zip will give diagonal directly: faster if wanted)
                # kidx = vec([ (ky << Lx) | kx for ky in idxvecs_ky[ny], kx in idxvecs_kx[nx] ])  # linear indices of all momenta combinations
                push!(vals, eval_on(kidx))
            end

            # Average over offsets and return as a 1D vector (to be reshaped later)
            return mean(reduce(hcat, vals), dims = 2)[:]
        end
    end

    # ==================================================================
    #                           1D CASE
    # ==================================================================
    # --- 1D Sampling for function f(k) = ⟨k|A|k⟩, where A is MPO -------------------


    @assert 1 ≤ num_k ≤ N "num_k ($num_k) must be in [1, N ($N)] to sample unique k-states."

    if averaging == false
        kxs = ilinspace(N, num_k)  # length == num_k
        kidx = kxs
        Ak = [f2(k) for k in kidx]
        return Ak
    elseif averaging == true
        # Step sizes between sample centers divided by 2 (half-step sizes
        # because we want to avoid too much overlap)
        step_size_k = div(N, num_k) / 2

        # Distance offsets: number of samples per point
        maxd = num_spl

        # Sampling positions in 1D
        ks = ilinspace(N, num_k)  # length == num_k

        # Positive and negative offsets
        k_pos_offs, k_neg_offs = make_scaled_offsets(step_size_k, maxd)

        # Build index vectors for base and offset points
        idxvecs = make_offset_idx_vectors(ks, L, N, k_pos_offs, k_neg_offs)

        # Evaluate f(x) on a given index vector
        eval_on = v -> [f2(i) for i in v]

        idx_names = collect(keys(idxvecs))
        sort!(idx_names, by = String)

        vals = [eval_on(idxvecs[k]) for k in idx_names]

        # Average over offsets and return as a 1D vector
        return mean(reduce(hcat, vals), dims = 2)[:]
    end
end


"""
    akdense(Tn_List, Nmu, e, N) -> Vector

Compute the **momentum-resolved spectral function at a single energy** `e`
(assumed **normalized** to lie in (-1, 1)). Builds the DOS MPO for `e`
from the Chebyshev expansion (`Tn_List`, `Nmu`), conjugates
to k-space, and samples ⟨k|·|k⟩ over all momenta k = 0,…,N-1.

Returns a vector of length `N` (one value per k: amount of real-space points). The element type follows
`get_spect_k` (typically real-valued; apply `real.`/`abs.` as needed).
"""
function akdense(Tn_List, Nmu, e, num_k, D; num_spl = 10, averaging = false, quantics_sampling::Bool = false)
    haha = get_DOS_from_Tn(Tn_List, Nmu, e)[1]; #delta function operator
    return get_spect_k(haha, num_k, D; num_spl = num_spl, averaging = averaging, quantics_sampling = quantics_sampling)
end







end # module
