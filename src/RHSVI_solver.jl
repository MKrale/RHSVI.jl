VERBOSE = true

#########################################
#            Solver
#########################################


@kwdef struct RHSVISolver <: Solver 
    max_time::Float64       = 60.0              # Timeout time (seconds)
    max_depth::Int          = 50
    init_iters::Int         = 10
    max_iters::Int          = 25_000
    epsilon::Float64        = 0.02              # (Relative) precision
    heuristic_solver        = RFIBSolver()      # Solver used to compute initial
    lowerbound_solver       = ZeroAlphas()
    backup_version          = CONVEX_BACKUP
end

# TODO: add outer loop to build a new tree after this one is sufficiently accurate
function POMDPs.solve(solver::RHSVISolver, env::X; return_pointset = false) where X<:POMDP
    t0 = time()
    VERBOSE && println("Initializing...")

    if !(env isa Index_IPOMDP)
        if env isa IPOMDP{<:Any,<:Any,<:Any}
            VERBOSE && println("Converting environment to Index_IPOMDP...")
            env = Index_IPOMDP(env)
            VERBOSE && println("Done. (Note: policy may require additional wrappers to run on the original env!))")
        elseif !(env isa RPOMDP{<:Any,<:Any,<:Any})
            VERBOSE && println("Converting environment to Index_POMDP...")
            env = Index_POMDP(env)
            VERBOSE && println("Done. (Note: policy may require additional wrappers to run on the original env!))")
        else
            println("Warning: could not convert $(typeof(env)) to known modeltype. This solver might break if environment spaces are not defined as integers.")
        end
    end

    tree::RHSVITree{X} = initialize_RHSVITree(env, solver)

    i,j = 1, 0
    while (rel_value_gap(tree, 1) > solver.epsilon) && (time()-t0 < solver.max_time) && (i < solver.max_iters)
        VERBOSE && println("\nIteration $i (Vs: $(tree.Vlower[1]), $(tree.Vupper[1])):")
        sampled_bidxs = [1]
        bidx::Int64 = 1
        h::Int64 = 0
        while ( value_gap(tree, bidx) * (discount(tree.env)^(h)) >= tree.Vlower[1] * solver.epsilon &&
                h < solver.max_depth &&
                !isterminalbelief(tree.env, tree.B[bidx]))
            bidx = explore(tree, bidx)
            push!(sampled_bidxs, bidx)
            h+=1
        end
        tree_backup(tree, sampled_bidxs; iteration=i)
        VERBOSE && println("Backup done!")
        prune!(tree)
        VERBOSE && println("Pruning done! (Currently $(length(tree.Alphas)) alphas and $(count(tree.B_pointset)) pointset beliefs)")
        i += 1
        if (mod(i,solver.init_iters * floor(1.5^j)) == 0) #|| tree.T_error[1] > 2
            VERBOSE && println("Resetting tree at $i iterations:")
            Alphas = pruneAlphas(tree.Alphas, tree.B[tree.B_pointset])
            tree = initialize_RHSVITree(env,solver;Alphas=Alphas, Vs_init=tree.Vsupper, extra_beliefs=tree.extra_beliefs, extra_upper_values=tree.extra_upper_values)
            j += 1
        end
    end
    Alphas = pruneAlphas(tree.Alphas, tree.B[tree.B_pointset])
    if VERBOSE
        println("Done! (In $i iterations, with Vs: $(tree.Vlower[1]), $(tree.Vupper[1]))")
        println("Printing all $(length(Alphas)) alpha-vectors:")
        for alpha in tree.Alphas
            println(alpha)
        end
    end

    policy = RobustAlphaVectorPolicy(tree.env, tree.Alphas)

    !return_pointset && (return policy)

    beliefs = tree.B[tree.B_pointset]
    values = tree.Vupper[tree.B_pointset]
    return beliefs, values, policy
end

#########################################
#               Belief Tree
#########################################

# struct SucessorBelief
#     bidx::Int
#     prob::Float64
#     oidx::Int
# end

@kwdef mutable struct RHSVITree{M}
    env::M where M<:POMDP
    C::ModelSizes

    B::Vector{DiscreteHashedBelief}             = DiscreteHashedBelief[] 
    Bps::Vector{Vector{Vector{SucessorBelief}}} = Vector{Vector{Vector{SucessorBelief}}}()
    Vupper::Vector{Float64}                     = Float64[]
    Vsupper::Vector{Float64}                    = Float64[]
    Vlower::Vector{Float64}                     = Float64[]
    Qupper::Vector{Vector{Float64}}             = Vector{Vector{Float64}}()
    Qlower::Vector{Vector{Float64}}             = Vector{Vector{Float64}}()
    Uncertainty::Vector{Float64}                = Float64[]

    needs_backup::Vector{Bool}                  = Bool[]

    Alphas::Vector{AlphaVector}                 = AlphaVector[]
    Alphas_protected::Int                            = 1
    B_pointset::BitVector                 = BitVector()
    B_expanded::BitVector                 = BitVector()
    backup_version::Int64                            = CONVEX_BACKUP

    extra_beliefs::Vector{DiscreteHashedBelief} = DiscreteHashedBelief[]
    extra_upper_values::Vector{Float64}         = Float64[]
end

function initialize_RHSVITree(env::X, solver::RHSVISolver; Alphas=[], Vs_init=nothing, extra_beliefs=nothing, extra_upper_values=nothing) where X<:POMDP
    
    constants = ModelSizes(env)
    if Vs_init isa Nothing
        heuristic_policy = solve(solver.heuristic_solver, env)
        Vs_init = get_exterior_values(heuristic_policy)
    end

    if isnothing(extra_beliefs) && isnothing(extra_upper_values)
        if isa(env, RPOMDP)
            VERBOSE && print("Solving non-robust variant...")
            non_robust_pomdp = Index_POMDP(to_mid_POMDP(env))
            prev_verbose = VERBOSE
            global VERBOSE = false
            non_robust_solver = RHSVISolver(max_time=solver.max_time/5, epsilon=solver.epsilon)
            extra_beliefs, extra_upper_values, _policy = solve(non_robust_solver, non_robust_pomdp; return_pointset=true)
            println(" Done! (Pointset of size $(length(extra_beliefs)) added)")
            global VERBOSE = prev_verbose
        else
            extra_beliefs, extra_upper_values = [], []
        end
    end

    append!(Alphas, solve(solver.lowerbound_solver, env).alphas)

    tree::RHSVITree{X} = RHSVITree{X}(
        env=env,
        C=constants,
        Vsupper=Vs_init,
        Alphas = Alphas,
        Alphas_protected = length(Alphas),
        backup_version = solver.backup_version,
        extra_beliefs = extra_beliefs,
        extra_upper_values = extra_upper_values
    )
    b0 = DiscreteHashedBelief(initialstate(tree.env))
    initialize_node(tree, b0)
    return tree
end

#########################################
#               Control Flow
#########################################
"""
The exploration function from HSVI (Alg. 2). 
Heuristicall chooses next sucessor belief of bidx to explore.
"""
function explore(tree::RHSVITree{M}, bidx::Int64) where M<:POMDP
    !(tree.B_expanded[bidx]) && expand_node!(tree, bidx)
    bestQ = maximum(tree.Qupper[bidx])
    aidxs = filter(aidx -> isapprox(tree.Qupper[bidx][aidx], bestQ), 1:length(actions(tree.env)))
    aidx = rand(aidxs)[1]
    # aidx = argmax(tree.Qupper[bidx])
    return uncertain_belief(tree, bidx, aidx)
end
"""
Recomputes upper- and lower bounds for all beliefs visited in counter-chronological order
"""
function tree_backup(tree::RHSVITree{M}, sampled_bidxs::Vector; iteration=0) where M<:POMDP
    bidxlast = sampled_bidxs[end]
    !tree.B_expanded[bidxlast] &&  (expand_node!(tree, sampled_bidxs[end]))
    tree.Uncertainty[bidxlast] = 0.0
    for bidx in reverse(sampled_bidxs)
        tree_backup(tree, bidx; iteration=iteration)
    end
end

"""
Prune beliefs and alpha vectors
"""
function prune!(tree::RHSVITree{M}) where M<:POMDP
    # HSVI only prunes these sporadically, but we prune them after each iteration, 
    # since the complexity of our exploration grows massively with |Alphas|
    prune_beliefs!(tree)
    # tree.Alphas = pruneAlphas(tree.Alphas, tree.B[tree.B_pointset], alphas_protected=tree.Alphas_protected) # using implementation form PBVI 
end

#########################################
#              Nodes Expansion
#########################################

"""
Initializes a belief node with belief b: return the index of the node.
"""
function initialize_node(tree, b)

    push!(tree.B, b)
    bidx = length(tree.B)
    push!(tree.Bps, Vector{SucessorBelief}())
    push!(tree.B_expanded, false)
    push!(tree.B_pointset, false)

    Vlower, Vupper = bounds(tree, bidx)
    push!(tree.Vlower, Vlower)
    push!(tree.Vupper, Vupper)
    push!(tree.Uncertainty, Vupper - Vlower)
    push!(tree.needs_backup, true)
    push!(tree.Qlower, Float64[])
    push!(tree.Qupper, Float64[])

    return bidx
end

"""
Expands a belief node: computes successor beliefs & Q-values.
"""
function expand_node!(tree::RHSVITree{M}, bidx::Int64) where M<:POMDP
    alphas::Vector{AlphaVector} = Vector{AlphaVector}(undef, tree.C.na)
    for a in 1:tree.C.na
        push!(tree.Bps[bidx], Vector{SucessorBelief}())
        Qlower, alpha, Bdist = backup(tree.env, tree.B[bidx], a, tree.Alphas, version=tree.backup_version)
        if Qlower != NoBackup()    
            alphas[a] = alpha
            for ((o,bp),p) in weighted_iterator(Bdist)
                bpidx = initialize_node(tree, bp)
                push!(tree.Bps[bidx][a], SucessorBelief(bpidx, p, o))
            end
            push!(tree.Qlower[bidx], Qlower)
            push!(tree.Qupper[bidx], upperbound(tree, bidx, a))
        end
    end
    tree.B_expanded[bidx] = true
    tree.B_pointset[bidx] = true
    tree.Alphas, isupdated::Bool = addDominantAlphas(alphas, tree.Alphas, tree.B[tree.B_pointset], alphas_protected=1)
    isupdated && (tree.needs_backup = fill(true, length(tree.needs_backup)))
    return isupdated
end

"""
Returns bidx if there is already a node with belief b, and nothing otherwise.
"""
belief_exists(tree, b) = nothing

#########################################
#              Belief Pruning
#########################################

function prune_beliefs!_(tree::RHSVITree{M}) where M<:POMDP
    for (bidx, b) in enumerate(tree.B)
        # Ignore if not expanded, already pruned or initial belief
        (!(tree.B_expanded[bidx]) || !(tree.B_pointset[bidx]) || bidx==1) && continue

        # Condition 1: prune sucessor beliefs if action is suboptimal
        Vlower = tree.Vlower[bidx]
        for a in 1:tree.C.na
            if (tree.Qupper[bidx][a] < Vlower)
                prune_subtree!(tree, bidx, a)
            end
        end
    end
end

function prune_beliefs!(tree::RHSVITree{M}) where M<:POMDP
    epsilon = 0.005
    for (bidx, b) in enumerate(tree.B)
        # Ignore if not expanded, already pruned or initial belief
        (!(tree.B_expanded[bidx]) || !(tree.B_pointset[bidx]) || bidx==1) && continue

        Vupper = tree.Vupper[bidx]
        if (Vupper - sawtooth(tree, bidx)) \ abs(Vupper) > epsilon
            tree.B_pointset[bidx] = false
        end
    end
end


function prune_subtree!(tree, bidx, aidx)
    if tree.B_expanded[bidx] && tree.B_pointset[bidx]
        for succbelief in tree.Bps[bidx][aidx]
            bpidx = succbelief.bidx
            for apidx in 1:tree.C.na
                prune_subtree!(tree, bpidx, apidx)
            end
        end
    end
    tree.B_pointset[bidx] = false
end

#########################################
#              Value bounds
#########################################
bounds(tree, bidx) = (lowerbound(tree,bidx), upperbound(tree, bidx))
"""
Returns the lower value bound for a  belief b.
"""
lowerbound(tree, bidx) = (maximum(alpha -> dot(alpha, tree.B[bidx]), tree.Alphas))

"""
Returns the upper value bound for a belief b.
"""
function upperbound(tree::RHSVITree{M}, bidx::Int) where M<:POMDP
    isterminalbelief(tree.env,tree.B[bidx]) && return 0.0
    Vup::Float64 = sawtooth(tree, bidx)
    tree.B_expanded[bidx] && (Vup = min(Vup, maximum(aidx -> upperbound(tree, bidx, aidx), 1:tree.C.na)))
    # return upperbound_VMDP(tree, bidx)
    return Vup
end

upperbound_VMDP(tree, bidx) = sum(s->pdf(tree.B[bidx],s)*tree.Vsupper[stateindex(tree.env, s)], support(tree.B[bidx]))

# sawtooth(tree, bidx::Int) = sawtooth(tree, tree.B[bidx])
function sawtooth(tree, bidx::Int)
    b = tree.B[bidx]
    alpha_corner = AlphaVector(tree.Vsupper, collect(1:tree.C.ns), nothing)
    Vb = dot(alpha_corner, b)
    Vmin = Vb
    for bint_idx in (1:length(tree.B))[tree.B_pointset]
        bint_idx == bidx && continue
        bint, vint = tree.B[bint_idx], tree.Vupper[bint_idx]
        ratio = min_ratio(b,bint)
        thisV = Inf
        if true #ratio > 0.0 #&& ratio < 1.0 
            thisV = Vb + ratio * (vint - dot(alpha_corner, bint))
        end
        Vmin = min(Vmin, thisV)
    end

    for (bint_idx, bint) in enumerate(tree.extra_beliefs)
        vint = tree.extra_upper_values[bint_idx]
        ratio = min_ratio(b,bint)
        thisV = Inf
        if true #ratio > 0.0 #&& ratio < 1.0 
            thisV = Vb + ratio * (vint - dot(alpha_corner, bint))
        end
        Vmin = min(Vmin, thisV)
    end

    return Vmin
end

function min_ratio(b::DiscreteHashedBelief,bp::DiscreteHashedBelief)
    minratio = Inf
    bidx = 1
    n_sup_b = length(b.state_list)
    n_sup_bp = length(bp.state_list)
    # n_sup_b != n_sup_bp && return 0.0
    bidx, bpidx = 1, 1
    while bidx <= n_sup_b && bpidx <= n_sup_bp
        sb, sbp = b.state_list[bidx], bp.state_list[bpidx]
        if sb == sbp
            minratio = min(minratio, b.probs[bidx] / bp.probs[bpidx])
            bidx += 1; bpidx += 1
        elseif sb < sbp
            bidx += 1
        else
            return 0.0
        end
    end
    bidx >= n_sup_b && bpidx <= n_sup_bp ? (return 0.0) : (return minratio)
end


"""
Returns the upper Q-value bound on belief b and action a.
"""
function upperbound(tree::RHSVITree{M}, bidx::Int64, aidx::Int64) where M<:POMDP
    Qupper::Float64 = beliefreward(tree.env, tree.B[bidx], aidx)
    for succbelief in tree.Bps[bidx][aidx]
        bpidx, p = succbelief.bidx, succbelief.prob
        Qupper += p * discount(tree.env) * tree.Vupper[bpidx]
    end
    return Qupper
end

function uncertainty(tree, bidx)
    uncertainty = 0.0
    for aidx in 1:tree.C.na
        for succbelief in tree.Bps[bidx][aidx]
            bpidx, p = succbelief.bidx, succbelief.prob
            uncertainty += p * discount(tree.env) * (tree.Uncertainty[bpidx])
        end
    end
    return uncertainty
end

"""
Compute the excess uncertainty for a belief-action-observation tuple.
"""
function uncertain_belief(tree, bidx, aidx)
    max_uncertainty = maximum(bp -> bp.prob * tree.Uncertainty[bp.bidx], tree.Bps[bidx][aidx])
    possible_bps = filter(bp -> isapprox( bp.prob * tree.Uncertainty[bp.bidx], max_uncertainty), tree.Bps[bidx][aidx])
    bpidxs = map(bp -> bp.bidx, possible_bps)
    return rand(bpidxs)[1]
end
"""
Return the absolute value gap for a given belief.
"""
value_gap(tree, bidx) = (tree.Vupper[bidx] - tree.Vlower[bidx])
"""
Return the relative value gap for a given belief.
"""
rel_value_gap(tree, bidx) = value_gap(tree, bidx) / abs(max(tree.Vlower[bidx])) #TODO: this breaks if Vupper = 0.0: think of how to fix this nicely!
"""
Update both bounds on  (Q-)values for the given belief, and add 
"""


function tree_backup(tree, bidx; iteration::Int=0, add_alphas=true)
    alphas, Vlower = AlphaVector[], -Inf

    # Update Lower bounds if necessary:
    if add_alphas && tree.needs_backup[bidx]
        for a in 1:tree.C.na
            Qlower, alpha, _Bdist = backup(tree.env, tree.B[bidx],a, tree.Alphas, version=tree.backup_version; prev_value=tree.Qlower[bidx][a])
            if Qlower==NoBackup()
                Vlower = max(Vlower, tree.Qlower[bidx][a])
            else
                tree.Qlower[bidx][a] = Qlower
                Vlower = max(Qlower, Vlower)
                if abs((Qlower - Vlower)/Vlower) < 0.05
                    push!(alphas, alpha)
                elseif Qlower > Vlower
                    alphas = [alpha]
                end
            end
        end
        tree.needs_backup[bidx] = false
        if !isempty(alphas)
            tree.Alphas, isupdated = addDominantAlphas(alphas, tree.Alphas, tree.B[tree.B_pointset], alphas_protected=1)
            isupdated && (tree.needs_backup = trues(length(tree.needs_backup)))
        end
        tree.Vlower[bidx] = Vlower
    else
        # println("Update skipped!")
    end

    # Update upper bounds & uncertainty:
    for a in 1:tree.C.na
        tree.Qupper[bidx][a] = upperbound(tree, bidx, a)
    end
    tree.Vupper[bidx] = min(maximum(tree.Qupper[bidx]), upperbound(tree, bidx))
    tree.Uncertainty[bidx] = min(tree.Uncertainty[bidx], uncertainty(tree, bidx), tree.Vupper[bidx] - tree.Vlower[bidx])

    if tree.Vlower[bidx] > tree.Vupper[bidx] * 1.01
        println("Error: Tree-backup finds invalid bounds!")
        println("b=$(tree.B[bidx]), bidx=$bidx V=[$(tree.Vlower[bidx]), $(tree.Vupper[bidx])")
        println("Qs=$(tree.Qlower[bidx])")
    end
end
