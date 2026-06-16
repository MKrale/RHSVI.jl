USE_ROBUST_MEM_UPDATE = false

function append_to_dict!(dict, key, value; func=+, minvalue=0)
    if haskey(dict, key)
        push!(dict[key], value)
    else
        dict[key] = [value]
    end
end

#########################################
#               QMDP:
#########################################

@kwdef struct RQMDPSolver <: Solver
    precision::Float64    = 1e-3
    max_time::Float64     = 600
    max_iterations::Int   = 5_000
end

get_max_r(m::X) where X<:POMDP = get_max_r(m,states(m), actions(m))
function get_max_r(m,S, A)
    maxr = 0
    for s in S
        for a in A
            maxr = max(maxr, reward(m,s,a))
        end
    end
    return maxr 
end

# POMDPs.solve(sol::RQMDPSolver, m::X) where X<:POMDP = solve(sol,m)

"""Computes the QMDP table using value iteration"""
function POMDPs.solve(sol::RQMDPSolver, m::X) where X<:POMDP
    t0 = time()
    C = ModelSizes(m)

    Q = zeros(C.ns,C.na)
    Qmax = zeros(C.ns)
    T = Matrix(undef, C.ns, C.na)
    max_r = get_max_r(m,1:C.ns, 1:C.na)
    maxQ = max_r / (1-discount(m))
    Q[:,:] .= maxQ
    Qmax[:] .= maxQ

    i=0
    # Lets iterate!
    factor = discount(m) / (1-discount(m))
    largest_change = Inf
    i=0

    while (factor * largest_change > sol.precision) && (i < sol.max_iterations)
        i+=1
        largest_change = 0
        for s in 1:C.ns
            for a in 1:C.na
                if isterminal(m,s)
                    Qnext = 0.0
                else
                    Qnext, belief = solvestep(sol, m, s, a, Qmax; return_belief=true)
                    Qnext = reward(m,s,a) + discount(m) * Qnext
                    T[s,a] = belief
                end
                largest_change = max(largest_change, abs((Qnext - Q[s,a]) / (Q[s,a]+1e-10) ))
                # println("$si, $ai, $Qnext")
                Q[s,a] = Qnext
            end
            Qmax[s] = maximum(Q[s,:])
        end
        time()-t0 > sol.max_time && break
    end
    Tf = (s,a) -> T[s, a]
    alphas = map(a -> AlphaVector(Q[:,a], collect(1:C.ns), a), 1:C.na)

    if USE_ROBUST_MEM_UPDATE
        return RobustAlphaVectorPolicy(m, alphas)
    else
        return RobustAlphaVectorPolicy(m, alphas, custom_memory_update=memory_update_RQMDP)
    end
end

# function solvestep(_sol::RQMDPSolver, m::POMDP, s, a, Qmax; return_belief=false)
#     if m isa RPOMDP
#         # Call the RPOMDP specific implementation
#         return solvestep(_sol, m::RPOMDP, s, a, Qmax; return_belief=return_belief)
#     else
#         # Call the POMDP implementation
#         return solvestep(_sol, m::POMDP, s, a, Qmax; return_belief=return_belief)
#     end
# end

function solvestep(_sol::RQMDPSolver, m::X, s, a, Qmax; return_belief=false) where X<:RPOMDP
    model = Model(Clp.Optimizer; add_bridges=false)
    set_silent(model)
    set_string_names_on_creation(model, false)

    T = transition(m,s,a)
    Sp = support(transition(m,s,a))
    nSp = length(Sp)
    @variable(model, p_s[1:nSp])
    Qs = []

    for (i,sp) in enumerate(Sp)
        prob_int = pdf(T,sp)
        @constraint(model, p_s[i] <= sup(prob_int))
        @constraint(model, p_s[i] >= inf(prob_int))
        append!(Qs, Qmax[stateindex(m,sp)])
    end

    @constraint(model, sum(p_s) == 1.0)
    @objective(model, Min, sum(Qs .* p_s))
    optimize!(model)
    thisQ = objective_value(model)
    # print(solution_summary(model))

    # TODO: read & output correct data
    if return_belief
        bnext = DiscreteHashedBelief(Sp, JuMP.value.(p_s))
        return thisQ, bnext
    else
        return thisQ
    end
    return objective_value(model)
end

function solvestep(_sol::RQMDPSolver, m::POMDP, s, a, Qmax; return_belief=false)
    thisT = transition(m,s,a)
    Qnext = 0
    for sp in support(thisT)
        Qnext += pdf(thisT, sp) * Qmax[stateindex(m,sp)]
    end
    return_belief ? (return Qnext, thisT) : (return Qnext)
end

function memory_update_RQMDP(π::RobustAlphaVectorPolicy, b, a, o)
    Ss = collect(states(π.env))
    probs = zeros(length(Ss))
    Qmax = get_Qmax(π)

    for s in support(b)
        bp = solvestep(RQMDPSolver(), π.env, s, a, Qmax, return_belief=true)[2]
        for sp in support(bp)
            probs[stateindex(π.env,sp)] += pdf(b,s) * pdf(bp,sp) * pdf(observation(π.env, a,sp), o)
        end
    end
    sum(probs) <= 0.0 && return initialstate(π.env)
    probs = probs ./ sum(probs)
    return DiscreteHashedBelief(Ss, probs)
end


#########################################
#               FIB:
#########################################

@kwdef struct RFIBSolver <: Solver
    precision::Float64    = 1e-5
    max_time::Float64           = 600
    max_iterations::Int         = 5_000
    exclude_interior::Bool      = false
end

# POMDPs.solve(sol::FIBSolver_alt, m::POMDP) = solve(sol,m;Data=nothing)

function POMDPs.solve(sol::RFIBSolver, m::X) where X<:POMDP
    t0 = time()
    
    # Get constants
    C = ModelSizes(m)
    T = Matrix(undef, C.ns, C.na)

    # Use RQMDP as initialization
    rqmdp_solver = RQMDPSolver(sol.precision, sol.max_time, sol.max_iterations)
    rqmdp_planner = solve(rqmdp_solver, m)
    Q = zeros(C.ns, C.na)
    
    for a in C.na
        Q[:,a] = rqmdp_planner.alphas[a].α  # This works only when indexing states & actions consistently
    end 

    
    Qmax = map(sidx -> maximum(Q[sidx,:]), 1:C.ns)

    i=0
    # Lets iterate!
    factor = discount(m) / (1-discount(m))
    largest_change = Inf
    i=0
    is_last_round = false
    while true
        # println(Q)
        i+=1
        largest_change = 0
        for s in 1:C.ns
            for a in 1:C.na
                if isterminal(m,s)
                    Qnext = 0
                    is_last_round && (T[s,a] = Deterministic(s))
                else
                    Qnext = solvestep(sol, m, s, a, Q; return_belief=false)
                    is_last_round && (T[s,a] = solvestep(sol, m, s, a, Q; return_belief=true)[2])
                end
                Qnext = reward(m,s,a) + discount(m) * Qnext
                largest_change = max(largest_change, abs((Qnext - Q[s,a]) / (Q[s,a]+1e-10) ))
                Q[s,a] = Qnext
            end
            Qmax[s] = maximum(Q[s,:])
        end
        if (factor * largest_change < sol.precision) || (i > sol.max_iterations) || time()-t0 > sol.max_time
            is_last_round && break
            is_last_round = true
        end
    end
    Tf = (s,a) -> T[s, a]
    alphas = map(a -> AlphaVector(Q[:,a], collect(1:C.ns), a), 1:C.na)
    if USE_ROBUST_MEM_UPDATE
        return RobustAlphaVectorPolicy(m, alphas)
    else
        return RobustAlphaVectorPolicy(m, alphas, custom_memory_update=memory_update_RFIB)
    end



    # USE_CUSTOM_POLICIES ? (return RQPolicy(m,Q,Qmax,Tf,C,S_dict)) : (return RobustAlphaVectorPolicy(m, alphas))
end

function solvestep(_sol::RFIBSolver, m::POMDP,s,a,Q; return_belief=false)
    thisQ = 0
    for o in observations(m)
        bnext = update(DiscreteHashedBeliefUpdater(m), DiscreteHashedBelief([s], [1.0]), a, o)
        isempty(support(bnext)) && continue
        prob_o = sum(sp -> pdf(transition(m,s,a), sp) * pdf(observation(m,a,sp), o), support(bnext) )
        Qo = zeros(length(actions(m)))
        for s in support(bnext)
            Qo += pdf(bnext,s) * Q[stateindex(m,s), :]
        end
        thisQ += prob_o * maximum(Qo)
    end
    return_belief ? (return thisQ, transition(m,s,a)) : (return thisQ)   
    # for oi in Data.SAOs[si, ai]
    #     bnext_idx = Data.B_idx[si,ai,oi]
    #     bnext = Data.B[bnext_idx]
    #     Qo = zeros(Data.constants.na)
    #     for s in support(bnext)
    #         Qo = Qo .+ ( pdf(bnext, s) .* Q[Data.S_dict[s], :])
    #     end
    #     thisQ += Data.SAO_probs[oi,si,ai] * maximum(Qo)
    # end
    # TODO: this is dumb: we are just copying the transition function. Can we do this more nicely?
    return_belief ? (return thisQ, transition(m,s,a)) : (return thisQ)   
end

function solvestep(_sol::RFIBSolver, m::IPOMDP,s,a,Q; return_belief=false)
    # model = direct_generic_model(Float64,Gurobi.Optimizer(GRB_ENV[]))#(GRB_ENV[]))
    # model = direct_generic_model(Float64,Clp.Optimizer())
    # s, a = Data.constants[si], Data.constants[ai]
    model = Model(Clp.Optimizer; add_bridges=false)
    set_silent(model)
    set_string_names_on_creation(model, false)

    # Defining variables
    T = transition(m,s,a)
    Sp = support(T)
    nSp = length(Sp)
    Sp_idxs = map(s->stateindex(m,s), Sp)
    O = observations(m)
    nO = length(O)
    Q = Q[Sp_idxs,:]

    # Define LP variables Pr(sp), Pr(o) and b_o(sp), with elementary constraints
    @variable(model, 0.0 <= prob_sp[1:nSp] <= 1.0)
    @constraint(model, sum(prob_sp) == 1.0)

    @variable(model, 0.0 <= belief[1:nO, 1:nSp] <= 1.0)
    for (spidx, sp) in enumerate(Sp)
        @constraint(model, sum(belief[:,spidx]) == prob_sp[spidx])
    end

    # Constraint: probabilities for sp's fall within intervals
    for (spidx,sp) in enumerate(Sp)
        prob_int = pdf(T,sp)
        @constraint(model, prob_sp[spidx] <= sup(prob_int))
        @constraint(model, prob_sp[spidx] >= inf(prob_int))
    end

    # Contraint: probability of observations follows from sp's
    for (oidx, o) in enumerate(O)
        for (spidx, sp) in enumerate(Sp)
            prob_o_given_sp = pdf(observation(m,a,sp), o)
            @constraint(model, belief[oidx,spidx] == prob_o_given_sp * prob_sp[spidx])
            # push!(prob_o_given_sp, pdf(observation(m,a,sp), o))
        end
        # @constraint(model, prob_o[oidx] >= sum(prob_o_given_sp .* prob_sp)) #???
    end

    # Constraint: Each Qo must be computed using the best possible action
    @variable(model, Qo[1:nO])
    for (oidx,o) in enumerate(O)
        for aidx in 1:length(actions(m))
            @constraint(model, Qo[oidx] >= sum(belief[oidx,:].* Q[:,aidx])) #Qo = Q-values for all sps given actions chosen after o
        end
    end
    
    @objective(model, Min, sum(Qo))
    optimize!(model)
    thisQ = objective_value(model)

    # If multiple next beliefs are possible, we pick one heuristically
    # Idea: try to maximize the spread in optimal actions.
    if return_belief
        bnext = DiscreteHashedBelief(Sp, JuMP.value.(prob_sp))
        return thisQ, bnext
    else
        return thisQ
    end
    # thisQ = objective_value(model)
    # return_belief ? (thisQ, bnext) : thisQ
    # return objective_value(model) #, parameter_value(model) #TODO: this is not correct...
end

function memory_update_RFIB(π::RobustAlphaVectorPolicy, b, a, o)
    Ss = collect(states(π.env))
    probs = zeros(length(Ss))
    Q = get_Q(π)

    for s in support(b)
        bp = solvestep(RFIBSolver(), π.env, s, a, Q, return_belief=true)[2]
        for sp in support(bp)
            probs[stateindex(π.env,sp)] += pdf(b,s) * pdf(bp,sp) * pdf(observation(π.env, a,sp), o)
        end
    end
    sum(probs) <= 0.0 && return initialstate(π.env)
    probs = probs ./ sum(probs)
    return DiscreteHashedBelief(Ss, probs)
end