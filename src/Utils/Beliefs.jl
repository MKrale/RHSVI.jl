#########################################
#          Belief Definitions
#########################################

struct DiscreteHashedBelief <: Distribution{Univariate, Discrete}
    state_list::Vector{Int64}       # assumed sorted!
    probs::Vector{Float64}
    hash::UInt
end

function DiscreteHashedBelief(state_list::AbstractArray, probs::AbstractArray{<:Float64})
    nonzero_els = findall(>(0),probs)
    state_list, probs = state_list[nonzero_els], probs[nonzero_els]
    idxs = sortperm(state_list)
    ordered_state_list, ordered_probs = state_list[idxs], probs[idxs]
    hash = makeDBhash(ordered_state_list, ordered_probs)
    return DiscreteHashedBelief(ordered_state_list, ordered_probs, hash)
end

#TODO: this is bad: we never check if our variable is indeed a belief. 
# I don't see any way to do this though: the beliefs used throughout the POMDP framework do not have a consistent supertype (even though they should all be distributions...)
# Maybe checking for the existance of a support/pdf function would be enough, but the way of doing this in Julia (method_exists()) seems to be removed and is the only thing I can find.
function DiscreteHashedBelief(b) 
    S,P = [], Float64[]
    for (s,p) in weighted_iterator(b)
        if p>0
            push!(S,s)
            push!(P,p)
        end
    end
    return DiscreteHashedBelief(S,P)
end

POMDPs.rand(rng::AbstractRNG, s::Random.SamplerTrivial{DiscreteHashedBelief}) = Base.rand(rng,s)
Base.rand(d::DiscreteHashedBelief) = Base.rand(Random.default_rng(), d)
function Base.rand(rng::AbstractRNG, d::DiscreteHashedBelief)
    r = rand(rng)
    tot = 0.0
    for x in support(d)
        tot += pdf(d,x)
        r < tot && return x
    end
    tot < 1.0 && throw("Trying to sample from non-normalized belief (with total probability $tot)")
    throw("Error: sampling from DiscretizedBelief failed for unknown reason.")
end

POMDPs.pdf(d::DiscreteHashedBelief, s::Real) = thispdf(d,s)
POMDPs.pdf(d::DiscreteHashedBelief, s::Int64) = thispdf(d,s)
function thispdf(d::DiscreteHashedBelief, s) 
    possible_ks = searchsorted(d.state_list, s)
    for k in possible_ks
        d.state_list[k] == s && return d.probs[k]
    end
    return 0
end
POMDPs.support(d::DiscreteHashedBelief) = d.state_list
POMDPTools.weighted_iterator(b::DiscreteHashedBelief) = zip(b.state_list, b.probs)

Base.length(d::DiscreteHashedBelief) = length(d.state_list)
mean(d::DiscreteHashedBelief) = throw("Function not implemented")
mode(d::DiscreteHashedBelief) = throw("Function not implemented")

#########################################
#          Hashing & Comparisons
#########################################

Base.:(==)(x::DiscreteHashedBelief, y::DiscreteHashedBelief) = (x.hash == y.hash) && all( map( s -> isapprox( pdf(x,s), pdf(y,s); atol=10^-3 ),  collect(support(x))))

function Base.:(<)(x::DiscreteHashedBelief, y::DiscreteHashedBelief)
    (x.hash != y.hash) && return (x.hash < y.hash)
    for k in sort(vcat(collect(support(x)), collect(support(y))))
        pdf(x,k) < pdf(y,k) && return true
        pdf(x,k) > pdf(y,k) && return false
    end
    return false
end
Base.isless(x::DiscreteHashedBelief, y::DiscreteHashedBelief) = x < y

makeDBhash(states_list::Vector, probs::Vector{Float64}) = hash(hash_alt(states_list), hash_alt(probs))
hash_alt(v::Vector) = foldr( (x,y) -> hash(x,y), v; init=UInt(0))

Base.hash(x::DiscreteHashedBelief, h::UInt) = hash(x.hash, h)
Base.hash(x::DiscreteHashedBelief) = hash(x,UInt(0))

#########################################
#          Belief Updater
#########################################

"""Struct for updating DiscreteHashedBelief"""
struct DiscreteHashedBeliefUpdater{S,A,O} <: Updater
    model::X where X<:POMDP{S,A,O}
end

"""Given a distribution d, create a DiscreteHashedBelief"""
function initialize_belief(bu::DiscreteHashedBeliefUpdater{Ss}, d) where Ss
    S,P = [], []
    for (s,p) in weighted_iterator(d)
        push!(S,s); push!(P,p)
    end
    return DiscreteHashedBelief(S,P)
end

function POMDPs.update(bu::DiscreteHashedBeliefUpdater{S}, b::DiscreteHashedBelief,a,o) where S
    model = bu.model
    bnext = Dict{S, Float64}()

    ### Collect possible next states with transition probs
    for (s, ps) in weighted_iterator(b)
        for (snext, psnext) in weighted_iterator(transition(model,s,a))
            add_to_dict!(bnext, snext, ps * psnext)
        end
    end

    ### Alter weights according to obs
    bnext_ = Dict{S, Float64}()
    for (snext, psnext) in bnext
        po = pdf(observation(model,a,snext), o)
        bnext_[snext] = psnext * po
    end
    ### Normalize
    states, probs = collect(keys(bnext_)), collect(values(bnext_))
    probs ./= sum(probs)

    return DiscreteHashedBelief(states,probs)
end

#TODO: again, we never type-check b, but I don't know how to do this...
POMDPs.update(bu::DiscreteHashedBeliefUpdater, b, a, o) = update(bu, DiscreteHashedBelief(b),a,o) 

##################################################################
#                       Utilities
##################################################################

function breward(model::POMDP, b::DiscreteHashedBelief,a)
    r = 0.0
    for (s,p) in zip(b.state_list, b.probs)
        s == POMDPTools.ModelTools.TerminalState() || ( r += p * POMDPs.reward(model,s,a) )
    end
    return r
end

function beliefreward(env::X, b::DiscreteHashedBelief, a) where X<:POMDP
    rewards = map(s -> reward(env,s,a), b.state_list)
    return sum(rewards .* b.probs)
end

isterminalbelief(env::POMDP, b::DiscreteHashedBelief) = all(s->isterminal(env,s), support(b))

"""Sets all probabilities < min_val to 0 to prevent floating-point shenanigans."""
function approximate_belief(b::DiscreteHashedBelief; min_val = 1e-6)
    ss, probs = b.state_list, b.probs
    prob_removed = 0.0
    for (sidx, s) in enumerate(ss)
        if probs[sidx] <= min_val
            prob_removed += probs[sidx]
            probs[sidx] = 0.0
        end 
    end
    probs = probs ./ (1-prob_removed)
    return DiscreteHashedBelief(ss, probs)
end

function get_surrounding_beliefs(b; epsilon=0.001)
    """Returns the list of all beliefs b' that are identical to b, but where probability mass epsilon gets added to b(s) for some s ∈ sup(b), and 1/|sup(b)|*epsilon gets removed from the others."""
    N_sup = length(support(b))
    Bs = [b]
    states = b.state_list
    for (sidx,s) in enumerate(support(b))
        probs = copy(b.probs)
        probs = probs .- (1/N_sup * epsilon)
        probs[sidx] += epsilon
        bp = DiscreteHashedBelief(states, probs)
        push!(Bs, bp)
    end
    return Bs
end

relevant_set_cache=LRU{Vector{Int64}, Vector{Int64}}(maxsize=1_000)
"""Returns 1) the support of b, 2) the possible next states given (b,a), and 3) the possible observations given (b,a). LRU cached."""
function get_relevant_sets(env, belief_support::Vector{Int64}, a)
    belief_support = Tuple(belief_support)
    return get(relevant_set_cache, (belief_support,a), compute_relevant_sets(env,belief_support,a))
end

function compute_relevant_sets(env, belief_support, a)
    Ss, Sps, Os = Set{Int64}(), Set{Int64}(), Set{Int64}()
    for s in belief_support
        push!(Ss, s)
        T = transition(env,s,a)
        for sp in support(T)
            if sup(pdf(T,sp)) > 0.0
                push!(Sps, sp)
                O = observation(env,a,sp)
                for o in support(O)
                pdf(O,o) > 0.0 && push!(Os, o)
                end
            end
        end
    end
    return sort(collect(Ss)), sort(collect(Sps)), sort(collect(Os))
end