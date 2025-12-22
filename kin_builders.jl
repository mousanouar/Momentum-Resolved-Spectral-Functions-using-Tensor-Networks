module kin_builders



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


# ================================================================
# Utilities
# ================================================================

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

"""
    binary_exponents(n::Integer) -> Vector{Int}

Return the indices of the set bits of `n`, i.e. `j` such that the `j`-th bit of `n`
(LSB = bit 0) is 1. Useful for power-by-squaring style algorithms.
"""
function binary_exponents(n::Integer)
    @assert n ≥ 0 "binary_exponents expects a non-negative integer"
    return [j for j in 0:(sizeof(n) * 8 - 1) if (n >>> j) & 1 == 1]
end

"""
    compose_power(base::MPO, nn::Integer; side::Symbol = :right,
                  apply_kwargs = NamedTuple()) -> MPO

Compose `base` with itself `nn` times using exponentiation-by-squaring.
This is **much more efficient** and numerically stable than looping `nn`
times or doing `j` nested self-compositions per set bit.

- If `side == :right`, multiplies as `acc = apply(acc, base)` when a bit is set.
- If `side == :left`,  multiplies as `acc = apply(base, acc)` when a bit is set.

`apply_kwargs` are forwarded to `apply` (e.g. `(cutoff=1e-8, maxdim=200)`),
letting you control truncation.

Edge cases:
- `nn == 0` returns an MPO equivalent to the identity on the same sites as `base`.
- `nn == 1` returns `base`.
"""
function compose_power(base, nn::Integer; side::Symbol = :right, apply_kwargs = NamedTuple())
    @assert nn ≥ 0 "nn must be non-negative"
    if nn == 0
        # Build an identity MPO with the same site set as `base`.
        sites = siteinds(base)
        # idsum = OpSum()
        # for (ℓ, _) in enumerate(sites)
        #     idsum += 1.0, "Id", ℓ
        # end
        return MPO(sites, "Id")
    end
    if nn == 1
        return base
    end

    # Power-by-squaring accumulator.
    acc = nothing
    cur = base
    k = nn
    while k > 0
        if (k & 1) == 1
            if acc === nothing
                acc = cur
            else
                acc = side === :right ? apply(acc, cur; apply_kwargs...) : apply(cur, acc; apply_kwargs...)
            end
        end
        k >>>= 1
        if k > 0
            cur = apply(cur, cur; apply_kwargs...)
        end
    end
    return acc::typeof(base)
end

# ================================================================
# 1D kinetic builder
# ================================================================

"""
    kineticNNN(L, sites, hopping, nn) -> MPO

Build a **long-range kinetic operator** on a 1D chain of length `L` (Qubit sites),
composed with a spatially varying `hopping` MPO. `nn` controls the effective
neighbor reach via repeated composition of a nearest-neighbor-like MPO.

Construction outline
--------------------
1. Build a base kinetic string `k_mpo_1` made of a σ⁺ at one position, identities to its left,
   and σ⁻ on a block to its right; sum such strings over all sites. This gives one
   “direction” of hopping.
2. Extend its range using `compose_power(k_mpo_1, nn; side=:right)`.
3. Left-multiply by the provided `hopping` field: `true_hop_1 = apply(hopping, An)`.
4. Build the Hermitian-conjugate direction (swap σ⁺ ↔ σ⁻) → `k_mpo_2`, extend its range
   with `compose_power(k_mpo_2, nn; side=:left)` and right-multiply by `dag(hopping)`.
5. Sum both directions with a small cutoff.

Notes
-----
- Use `apply_kwargs` to control truncation (e.g. `cutoff`, `maxdim`). Defaults are those of ITensor.
- `nn = 1` gives the base MPO (nearest-neighbor-like). Larger `nn` increases the reach approximately.
- The algebra of these strings depends on the spin-1/2 (Qubit) convention that
  `sigma_plus`/`sigma_minus` are raising/lowering operators.
"""
function kineticNNN(L, sites, hopping, nn; apply_kwargs = NamedTuple())
    @assert L == length(sites) "L must equal length(sites)"
    @assert nn ≥ 1 "nn must be ≥ 1"

    kinetic_1 = OpSum()  # σ⁺ … σ⁻ direction
    kinetic_2 = OpSum()  # σ⁻ … σ⁺ direction (Hermitian conjugate)

    # ---- Build the first kinetic direction: σ⁺ with identities left, σ⁻ block right
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_plus", L - (i - 1)
        for j in 1:L - i
            os *= ("Id", j)
        end
        for j in (L + 2 - i):L
            os *= ("sigma_minus", j)
        end
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1, sites)

    # Extend range using power-by-squaring (right-accumulating)
    An = compose_power(k_mpo_1, nn; side = :right, apply_kwargs)

    # Compose with the hopping field (left-multiply)
    true_hop_1 = apply(hopping, An; apply_kwargs...)

    # ---- Build the Hermitian-conjugate direction: swap σ⁺ ↔ σ⁻
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_minus", L - (i - 1)
        for j in 1:L - i
            os *= ("Id", j)
        end
        for j in (L + 2 - i):L
            os *= ("sigma_plus", j)
        end
        kinetic_2 += os
    end
    k_mpo_2 = MPO(kinetic_2, sites)

    # Extend range; left-accumulating mirrors the first branch
    Am = compose_power(k_mpo_2, nn; side = :left, apply_kwargs)

    # Right-multiply by dag(hopping)
    true_hop_2 = apply(Am, dag(hopping); apply_kwargs...)

    # ---- Sum both directions with a small cutoff
    k_mpo = +(true_hop_1, true_hop_2; cutoff = 1e-12)
    return k_mpo
end

# ================================================================
# 2D helpers (row-break / row-select / checkerboard masks as MPOs)
# Conventions:
# - Row-major flattening: linear index i = ix + iy * 2^Lx
# - Bit split (Quantics): lowest Lx bits -> x (= ix), next Ly bits -> y (= iy)
# - All masks are diagonal MPOs with entries in {0.0, 1.0}
# ================================================================

"""
    _row_break_mpo(Lx, Ly, sites; which::Symbol) -> MPO

Diagonal **row-boundary mask** that zeroes out wrap-around couplings when a
`2^Lx × 2^Ly` grid is flattened row-major into a 1D chain.

Boundary conventions (`which`):
- `:xplus`  — break at the **end** of each row (mask 0 on indices where `(ix+1) % 2^Lx == 0`);
- `:xplain` — break at the **start** of each row (mask 0 on indices where `ix % 2^Lx == 0`).

Use this mask to prevent accidental couplings between `(ix = 2^Lx−1, iy)` and
`(ix = 0, iy)` when building kinetic MPOs on the flattened chain.

Arguments
- `Lx, Ly`: numbers of Quantics bits for x and y (system sizes are `2^Lx`, `2^Ly`).
- `sites`: ITensors site indices (Qubit sites expected).
- `which`: one of `:xplus` or `:xplain` (see above).

Returns
- A diagonal MPO of size `Lx + Ly` qubits with entries 0/1.

Notes
- The TT is constructed via `quanticscrossinterpolate` with `tolerance=1e-8`.
- This is a **multiplicative filter**: multiply your MPO by this mask to enforce breaks.
"""
function _row_break_mpo(Lx, Ly, sites; which::Symbol)
    L = Lx + Ly
    xvals = range(0, (2^L - 1); length = 2^L)  # global linear indices (0..2^(Lx+Ly)-1)

    # Choose boundary-breaking predicate on the x (column) index ix
    f = which === :xplus  ? (x -> iszero(mod(x + 1, 2^Lx)) ? 0.0 : 1.0) :
        which === :xplain ? (x -> iszero(mod(x,     2^Lx)) ? 0.0 : 1.0) :
        error("unknown which=:$(which)")

    # TT -> MPS -> diagonal MPO
    qttb  = quanticscrossinterpolate(ComplexF64, f, xvals; tolerance = 1e-8)[1]
    ttb   = TCI.tensortrain(qttb.tci)
    maskmps = MPS(ttb; sites)
    maskmpo = outer(maskmps', maskmps)

    # Project each site tensor to the diagonal (builds the 0/1 diagonal MPO)
    for i in 1:L
        maskmpo.data[i] = Quantics._asdiagonal(maskmps.data[i], sites[i])
    end
    return maskmpo
end

# ================================================================
# 2D kinetic builders
# ================================================================

"""
    kineticintra2DNNN(Lx, Ly, sites, hopping, nn) -> MPO

Long-range **intra-row** hopping on an `Lx × Ly` square lattice flattened in row-major order.
Prevents links between the last site of a row and the first site of the next row by
left- and right-multiplying with a diagonal *row-break* MPO mask.
"""
function kineticintra2DNNN(Lx, Ly, sites, hopping, nn; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites)
    @assert nn ≥ 1

    # Build the 1D-like base in each direction (same as kineticNN)
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i - 1)
        for j in 1:L - i; os *= ("Id", j); end
        for j in (L + 2 - i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1, sites)
    An = compose_power(k_mpo_1, nn; side = :right, apply_kwargs)

    true_hop_1 = apply(hopping, An; apply_kwargs...)

    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i - 1)
        for j in 1:L - i; os *= ("Id", j); end
        for j in (L + 2 - i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k_mpo_2 = MPO(kinetic_2, sites)
    Am = compose_power(k_mpo_2, nn; side = :left, apply_kwargs)

    true_hop_2 = apply(Am, dag(hopping); apply_kwargs...)

    # Break wrap-around between rows
    breakmpo = _row_break_mpo(Lx, Ly, sites; which = :xplus)
    true_hop_1 = apply(breakmpo, true_hop_1; apply_kwargs...)
    true_hop_2 = apply(true_hop_2, breakmpo; apply_kwargs...)

    return +(true_hop_1, true_hop_2; cutoff = 1e-12)
end






end # module
