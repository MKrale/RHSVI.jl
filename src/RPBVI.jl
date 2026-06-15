VERBOSE = false

##################################################################
#                           Solver 
##################################################################

@kwdef struct RPBVISolver <: Solver
    max_iterations::Int64   = 250           # number of iterations
    precision::Float64      = 1e-3          # Unused...
end

function POMDPs.solve(solver::RPBVISolver, env::X) where X<:POMDP

    # B = getBeliefSet(env,solver)
    B = [DiscreteHashedBelief(initialstate(env))]
    Breached = []
    S = states(env)
    ns = length(S)

    Alphas = solve(ZeroAlphas(),env).alphas

    i = 0
    while i < solver.max_iterations
        i += 1
        VERBOSE && println("Performing Backups...")
        Alphas, Bsreached = backup(env,Alphas, B)
        VERBOSE && println("Pruning...")
        Alphas = pruneAlphas(Alphas, B)
        append!(B, expandBeliefSetRPBVI(B,Bsreached))
        VERBOSE && println("Iteration $i complete (Using $(length(Alphas)) vectors and $(length(B)) beliefs).")
    end
    return RobustAlphaVectorPolicy(env, Alphas)
end

##################################################################
#                 Belief expansion & Pruning 
##################################################################

function getUniformBeliefSet(env::X, solver; gridsize=3) where X<:POMDP
    S = states(env)
    ns = length(S)
    Ranges = [0:gridsize for i in 1:ns]
    Gridpoints = collect(Iterators.product(Ranges...))
    Gridpoints = [collect(p) ./ gridsize for p in Gridpoints if sum(p)==gridsize]
    Beliefs = []
    for p in Gridpoints
        mask = p .> 0
        push!(Beliefs, DiscreteHashedBelief(S[mask], p[mask]))
    end
    return Beliefs
end

function expandBeliefSetRPBVI(B, Bsreached; n=nothing, mindist = 0.1)
    n isa Nothing && (n = length(B))
    sampled_bs, ds = [], []
    for bdist in Bsreached
        _o, bp = rand(bdist)
        d = beliefDistanceSet(bp, B)
        if d > mindist
            push!(sampled_bs, bp)
            push!(ds, d)
        end
    end
    best_indexes = sortperm(ds)[1:min(n, length(ds))]
    return sampled_bs[best_indexes]
end

function beliefDistanceSet(bp, B)
    return minimum(map(b -> beliefDistance(bp,b), B))
end

function beliefDistance(bp::DiscreteHashedBelief, b::DiscreteHashedBelief)
    return sum(s -> abs(pdf(bp,s) - pdf(b,s)), support(bp)) # This is supposed to be L1
end

##################################################################
#              Alpha Vectors expansion & Pruning 
##################################################################


### Note: it might be worthwhile to do more thorough pruning every so often...
function pruneAlphas(Alphas::Vector{<:AlphaVector}, B; alphas_protected::Int=1) # Currently: belief-space domination
    na = length(Alphas)
    mask = trues(na)
    for idx1 in 1:na
        !mask[idx1] && continue
        for idx2 in idx1+1:na
            !mask[idx2] && continue
            alpha1, alpha2  = Alphas[idx1], Alphas[idx2]
            # a1_dominant, a2_dominant = beliefspace_dominant(alpha1, alpha2, B)
            a1_dominant, a2_dominant = pointwise_dominant(alpha1, alpha2)

            if a1_dominant
                mask[idx2] = false
            elseif a2_dominant
                mask[idx1] = false
                break
            end
        end
    end
    mask[1:alphas_protected] .= true
    return Alphas[mask]
end


function addDominantAlphas(newAlphas::Vector{<:AlphaVector}, oldAlphas::Vector{<:AlphaVector}, B; alphas_protected::Int=1)
    nnew, nold = length(newAlphas), length(oldAlphas)
    masknew, maskold = trues(nnew), trues(nold)
    new_dominant_alpha = false

    for idx1 in 1:nnew
        !masknew[idx1] && continue
        
        # First check if the new alpha dominates any old vector
        for idx2 in reverse(1:nold)
            !maskold[idx2] && continue
            alpha1, alpha2  = newAlphas[idx1], oldAlphas[idx2]
            # a1_dominant, a2_dominant = beliefspace_dominant(alpha1, alpha2, B)
            a1_dominant, a2_dominant = pointwise_dominant(alpha1, alpha2)

            if a2_dominant
                masknew[idx1] = false
                break
            elseif a1_dominant && idx2 > alphas_protected
                maskold[idx2] = false
                new_dominant_alpha = true
            end
        end

        # Now 'normal' pruning within bnew
        for idx2 in idx1+1:nnew
            !masknew[idx2] && continue
            alpha1, alpha2  = newAlphas[idx1], newAlphas[idx2]
            # a1_dominant, a2_dominant = beliefspace_dominant(alpha1, alpha2, B, delta=0.0)
            a1_dominant, a2_dominant = pointwise_dominant(alpha1, alpha2)

            if a1_dominant
                masknew[idx2] = false
            elseif a2_dominant
                masknew[idx1] = false
                break
            end
        end
    end
    maskold[1:alphas_protected] .= true
    return vcat(oldAlphas[maskold], newAlphas[masknew]), new_dominant_alpha
end

function pointwise_dominant(alpha1::AlphaVector, alpha2::AlphaVector)
    alpha1.action !== alpha2.action && (return false, false)
    a1idx, a2idx = 1, 1
    a1_dominant, a2_dominant = true, true
    n_a1_sup, n_a2_sup = length(alpha1.states), length(alpha2.states)
    while true
        s1, s2 = alpha1.states[a1idx], alpha2.states[a2idx]
        if s1 == s2
            v1, v2 = alpha1.α[a1idx], alpha2.α[a2idx]
            if v1 > v2
                a2_dominant=false
            elseif v2 > v1 
                a1_dominant=false
            end
            a1idx+=1; a2idx+=1
        elseif s1 > s2
            a1_dominant = false 
            a2idx+=1
        elseif s2 > s1
            a2_dominant = false 
            a1idx+=1
        else
            println("Error in Pointwise Domination: states are incomparable")
            println(s1, " ",s2, " ", a1idx, " ", a2idx)
        end
        a1idx > n_a1_sup && a2idx > n_a2_sup && break
        a1idx > n_a1_sup && (a1_dominant = false; break)
        a2idx > n_a2_sup && (a2_dominant = false; break)
        if !a1_dominant && !a2_dominant
            break
        end
    end

    return a1_dominant, a2_dominant
end

# Backups
        

"""
Full robust backup algorithm for RPOMDPs, as defined by Osogami (2015), Alg. 1.
"""
function backup(env::X, Alphas::Vector{<:AlphaVector}, B)::Tuple{Vector{<:AlphaVector},Vector{<:Any}} where X<:POMDP
    newAlphas, newBs = AlphaVector[], []
    for b in B
        alphas, Breached = backup(env,b,Alphas)
        append!(newAlphas, alphas)
        append!(newBs, Breached)
    end
    return newAlphas, newBs
end

"""
Robust backup for a single belief.
"""
function backup(env::X, b::DiscreteHashedBelief, Alphas::Vector{<:AlphaVector})::Tuple{Vector{<:AlphaVector},Vector{<:Any}} where X<:POMDP
    epsilon = 0.05
    bestQ, bestAlphas = -Inf, []
    Breached = []
    for a in actions(env)
        thisQ, thisAlpha, Bdistr = backup(env,b,a,Alphas)
        push!(Breached, Bdistr)
        if abs((thisQ - bestQ)/bestQ) < epsilon
            bestQ = max(thisQ, bestQ)
            push!(bestAlphas, thisAlpha)
        elseif thisQ > bestQ
            bestQ = thisQ
            bestAlphas = [thisAlpha]
        end
    end
    return bestAlphas, Breached
end