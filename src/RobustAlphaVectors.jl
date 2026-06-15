global CONVEX_BACKUP, OSOGAMI_BACKUP
CONVEX_BACKUP, OSOGAMI_BACKUP = 0,1

##################################################################
#                       Definitions
##################################################################

struct AlphaVector
    α::Vector{Float64}
    states::Vector{Int64} # assumed sorted!
    action::Any
    hash::UInt
end
function AlphaVector(α::Vector{Float64}, states::Vector{Int64}, action)
    idxs = sortperm(states)
    states, α = states[idxs], α[idxs]
    hash = makeDBhash(states, α)
    return AlphaVector(α, states, action, hash)
end

##################################################################
#                       Utility functions
##################################################################

function Base.getindex(a::AlphaVector, s::Int64)
    sidx = findfirst(isequal(s), a.states)
    sidx isa Nothing && return -10^10
    return a.α[sidx]
end
Base.getindex(a::AlphaVector, Ss::AbstractArray) = map(s -> getindex(a,s), Ss)

# dot(env::X, alphavec::AlphaVector, b::D) where {X<:POMDP, D<:Distribution} = dot(env,alphavec,b) # POMDP distributions cannot be typechecked: life is hard...
function dot(alphavec::AlphaVector, b)
    return sum(s -> pdf(b,s) * alphavec[s], support(b))
end

### SOMETHING IS WRONG HERE
function dot(alphavec::AlphaVector, b::DiscreteHashedBelief)
    alphainf = 10^10
    d = 0.0
    aidx, bidx = 1, 1
    n_sup_b = length(b.state_list)
    n_sup_a = length(alphavec.states)
    while bidx <= n_sup_b
        aidx > n_sup_a && return -alphainf
        sa, sb = alphavec.states[aidx], b.state_list[bidx]
        if sb == sa 
            d += b.probs[bidx] * alphavec.α[aidx]
            bidx += 1; aidx += 1
        elseif sb < sa
            return -alphainf 
        elseif sa < sb
            aidx += 1
        end
    end
    return d 
end

function support_has_overlap(a::AlphaVector, S)
    for s in S
        if !(findfirst(isequal(s), a.states) isa Nothing)
            return true
        end
    end
    return false
end

function support_is_subset(a1::AlphaVector, a2::AlphaVector)
    a1_subset = true
    n = 0
    for s in a1.states
        if findfirst(isequal(s), a2.states) isa Nothing
            a1_subset = false 
        else
            n+=1
        end
    end
    return (length(a1.states) == n, a1_subset)
end

function is_valid_alpha(env, alpha, a, o, Sps)
    """Return true only alpha has a value for each possible s ∈ Sps with Pr(o | a,sp) > 0"""
    for sp in Sps
        if pdf(observation(env,a,sp), o) > 0.0 
            if isnothing(findfirst(isequal(sp), alpha.states))
                return false
            end
        end
    end
    return true
end

##################################################################
#                   Alpha Vectors Policies
##################################################################

RANDOMIZE_EPSILON = 1e-3

@kwdef struct RobustAlphaVectorPolicy <: Policy 
    env::X where X<:POMDP           # (R)POMDP model
    alphas::Vector{AlphaVector}     # List of alpha-vectors, assumed sorted according to actions
    aidxs::Vector{Any}              # Indexes of alpha-vectors with a given action (usefull for updates)
    custom_memory_update = nothing
end

function RobustAlphaVectorPolicy(env, alphas; custom_memory_update=nothing)
    alpha_actions = map(alpha -> actionindex(env, alpha.action), alphas)
    mask = sortperm(alpha_actions)
    alpha_actions = alpha_actions[mask]
    alphas = alphas[mask]

    aidxs = []
    for aidx in map( a-> actionindex(env, a),actions(env))
        start, stop = findfirst(isequal(aidx), alpha_actions), findlast(alpha_actions .== aidx)
        (start isa Nothing) ? (push!(aidxs, [])) : (push!(aidxs, start:stop))
    end
    return RobustAlphaVectorPolicy(env, alphas, aidxs, custom_memory_update)
end

"""Returns vector of probabilities for playing the given alpha-vectors"""
function stochastic_choice_alphas(b, alphas)

    if length(alphas) == 1
        return [1.0]
    end
    
    ### Model setup
    # model = Model(Clp.Optimizer; add_bridges=false)
    # model = Model(Gurobi.Optimizer; add_bridges=false)
    model = direct_generic_model(Float64, Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model)
    # set_string_names_on_creation(model, false)

    ### Setup LP:
    nmbr_optimal_actions = length(alphas)
    ns = length(support(b))

    @variable(model, 0 <= alpha_probs[1:nmbr_optimal_actions] <= 1)
    @variable(model, 0 <= exploitability[1:ns, 1:ns])
    @constraint(model, sum(alpha_probs) == 1)

    for (sidx, s) in enumerate(support(b))
        for (spidx, sp) in enumerate(support(b))
            ### Weighted:
            # @constraint(model, exploitability[sidx, spidx] >= sum( aidx -> alpha_probs[aidx] * (pdf(b,s) * alphas[aidx][s] - pdf(b,sp) * alphas[aidx][sp]), 1:nmbr_optimal_actions ) )
            # @constraint(model, exploitability[sidx, spidx] >= sum( aidx -> alpha_probs[aidx] * (pdf(b,sp) * alphas[aidx][sp] - pdf(b,s) * alphas[aidx][s]), 1:nmbr_optimal_actions ) )
            ### Unweighted:
            #TODO: maybe check weighting off full sum? (e.g. pdf(b,s) * pdf(b,sp) * sum(...))
            @constraint(model, exploitability[sidx, spidx] >= 0 )
            @constraint(model, exploitability[sidx, spidx] >= sum( aidx -> alpha_probs[aidx] * (alphas[aidx][sp] - alphas[aidx][s]), 1:nmbr_optimal_actions ) )
        end
    end
    @objective(model, Min, sum(exploitability))
    optimize!(model)
    
    ### Formatting
    probs =  JuMP.value.(alpha_probs)
    return probs
end

function get_optimal_alphas(π::RobustAlphaVectorPolicy, b)
    env = π.env
    na = length(actions(env))
    ### Find the best alpha-vector for each action
    best_values, best_alphas = zeros(na) .- Inf, Vector(undef, na)
    for (alphaidx, alpha) in enumerate(π.alphas)
        this_value = dot(alpha, b)
        aidx = actionindex(env, alpha.action)
        if best_values[aidx] < this_value
            best_values[aidx] = this_value
            best_alphas[aidx] = alpha 
        end         
    end
    
    ### Select alpha-vectors for those actions that could be optimal
    val, best_aidx = findmax(best_values)
    value_bound = val - abs(val*RANDOMIZE_EPSILON)
    mask = map(aidx -> best_values[aidx] >= value_bound, 1:na)
    return best_alphas[mask]
end

# action_value(π::RobustAlphaVectorPolicy, b::X; randomize_epsilon=0.01) where X<:Distribution = action_value(π, b; randomize_epsilon=randomize_epsilon)
@memoize LRU(maxsize=100_000) function action_value_distr(π::RobustAlphaVectorPolicy, b; randomize_epsilon=RANDOMIZE_EPSILON)
    
    ### Get all possibly optimal alpha-vectors
    best_alphas = get_optimal_alphas(π::RobustAlphaVectorPolicy, b)
    val = dot(best_alphas[1], b)
    actions = map(alpha -> alpha.action, best_alphas)

    ### If only one alpha-vector is optimal, or we only have one state, we use a uniform distribution
    if length(best_alphas) == 1 || length(support(b)) == 1
        probs = repeat([1/length(actions)], length(actions))
        return SparseCat(actions, probs), val
    
    ### Otherwise, get a distribution over actions
    else
        probs_alphas = stochastic_choice_alphas(b, best_alphas)

        ### TODO: this seems overkill...
        probs_actions = zeros(length(actions))
        for (aidx, a) in enumerate(actions)
            for (alpha_idx, alpha) in enumerate(best_alphas)
                if alpha.action == a 
                    probs_actions[aidx] += probs_alphas[alpha_idx]
                end
            end
        end
    
        return SparseCat(actions, probs_actions), val
    end

end

function action_value(π::RobustAlphaVectorPolicy, b; randomize_epsilon=RANDOMIZE_EPSILON)
    A_dist, V = action_value_distr(π,b; randomize_epsilon=randomize_epsilon)
    a = rand(A_dist)
    return a, V
end

POMDPs.action(π::RobustAlphaVectorPolicy,b) = first(action_value(π,b))
POMDPs.value(π::RobustAlphaVectorPolicy,b) = last(action_value(π,b))

get_memory_type(π::RobustAlphaVectorPolicy) = DiscreteHashedBelief
function get_initial_memory(π::RobustAlphaVectorPolicy)
    b0 = initialstate(π.env)
    return DiscreteHashedBelief(support(b0), map(s->pdf(b0,s), support(b0)))
end

update_memory(π::RobustAlphaVectorPolicy, b, a, o) = update_memory(π,DiscreteHashedBelief(b),a,o)
@memoize LRU(maxsize=100_000) function update_memory(π::RobustAlphaVectorPolicy, b::DiscreteHashedBelief, a, o)
    !isnothing(π.custom_memory_update) && (return π.custom_memory_update(π,b,a,o))
    Q_, alpha_, Bdistr = backup(π.env,b,a,π.alphas)
    Os, Bos = getindex.(support(Bdistr), 1), getindex.(support(Bdistr),2)
    isempty(Os) && println("Error: impossible observation $o for belief $b and action $a found.")
    idx = findfirst(isequal(o), Os)
    idx isa Nothing && return initialstate(π.env) # Observations with 0 probability are not included in Os, but may be defined.
    return Bos[idx]
end

function get_exterior_values(π::RobustAlphaVectorPolicy)
    Vs = zeros(length(states(π.env)))
    for s in states(π.env)
        Vs[stateindex(π.env, s)] = maximum(alpha -> alpha[s], π.alphas)
    end
    return Vs
end

@memoize LRU(maxsize=10) function get_Qmax(π::RobustAlphaVectorPolicy)
    Qmax = []
    for s in states(π.env)
        push!(Qmax, maximum(alpha -> alpha[s], π.alphas))
    end
    return Qmax
end

@memoize LRU(maxsize=10) function get_Q(π::RobustAlphaVectorPolicy)
    Q = zeros( length(states(π.env)), length(actions(π.env)) )
    for s in states(π.env)
        sidx = stateindex(π.env, s)
        for alpha in π.alphas
            aidx = actionindex(π.env, alpha.action)
            Q[sidx, aidx] = max(Q[sidx, aidx], alpha[s])
        end    
    end
    return Q
end

##################################################################
#                       Domination & Pruning
##################################################################

function beliefspace_dominant(a1::AlphaVector, a2::AlphaVector, B; delta=0.01)
    # Condition 1: different actions (we want to keep these for robustness)
    a1.action !== a2.action && (return false, false)

    # Condition 2: domination must happen for all states in the support
    a1_dominant, a2_dominant = support_is_subset(a1,a2)

    # Condition 3: belief-wise domination (SARSOP)

    for b in B
        !a1_dominant && !a2_dominant && (return (false,false))
        sumsquared, dot_sum = 0.0, 0.0
        for s in support(b)
            if s in a1.states
                diff = a1[s] - a2[s]
                sumsquared += abs2(diff)
                dot_sum += diff*pdf(b,s)
            end
        end
        dV = dot_sum / sqrt(sumsquared)
        dV <= delta && (a1_dominant = false)
        dV >= -delta && (a2_dominant = false)
    end
    return a1_dominant, a2_dominant
end

function beliefspace_dominant(a1::AlphaVector, a2::AlphaVector, B::AbstractVector{DiscreteHashedBelief}; delta=0.01)
    # Condition 1: different actions (we want to keep these for robustness)
    a1.action !== a2.action && (return false, false)

    # Condition 2: domination must happen for all states in the support
    a1_dominant, a2_dominant = support_is_subset(a1,a2)

    # Condition 3: belief-wise domination (SARSOP)

    n_a1_sup, n_a2_sup = length(a1.states), length(a2.states)

    for b in B
        !a1_dominant && !a2_dominant && (return (false,false))
        bidx, a1idx, a2idx = 1, 1, 1
        sumsquared, dot_sum = 0.0, 0.0
        n_b_sup = length(b.state_list)

        while bidx <= n_b_sup
            a1idx > n_a1_sup && break
            sb, sa1 = b.state_list[bidx], a1.states[a1idx]

            if sb == sa1 
                a2idx <= n_a2_sup ? (sa2 = a2.states[a2idx]) : sa2 = nothing
                if sa1 == sa2 && !(sa2 isa Nothing)
                    diff = a1.α[a1idx] - a2.α[a2idx]
                    sumsquared += abs2(diff)
                    dot_sum += diff * b.probs[bidx]
                    bidx += 1; a1idx += 1; a2idx += 1
                elseif sa1 < sa2 || sa2 isa Nothing
                    diff = a1.α[a1idx]
                    sumsquared += abs2(diff)
                    dot_sum += diff * b.probs[bidx]
                elseif sa2 < sa1
                    a2idx += 1
                end

            elseif sb < sa1
                bidx += 1
            elseif sa1 < sb
                a1idx += 1
            end
        end
        dV = dot_sum / sqrt(sumsquared)
        dV <= delta && (a1_dominant = false)
        dV >= -delta && (a2_dominant = false)       
    end
    return a1_dominant, a2_dominant
end


###
#   Helper Functions:
###

struct SucessorBelief
    bidx::Int
    prob::Float64
    oidx::Int
end


##################################################################
#                       Other
##################################################################

#TODO: this is incorrect: redo!
struct ZeroAlphas <: Solver end
function POMDPs.solve(solver::ZeroAlphas, env)
    Rmin, aminidx = findmax( a -> (minimum( s -> reward(env, s, a), states(env))), actions(env))
    Vmin = Rmin / (1.0-discount(env))
    alpha_normal = AlphaVector(zeros(length(states(env))) .+ Vmin, collect(states(env)), actions(env)[aminidx])
    s_terminal = []
    for s in states(env)
        isterminal(env,s) && push!(s_terminal,s)
    end
    return RobustAlphaVectorPolicy(env, [alpha_normal])
end