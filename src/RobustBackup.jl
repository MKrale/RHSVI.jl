function backup(env::M, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector}; version::Int=CONVEX_BACKUP, prev_value=-Inf) where M<:IPOMDP
    """Performs a robust backup."""
    version == OSOGAMI_BACKUP && return osogami_backup(env,b,a,Alphas)
    version != CONVEX_BACKUP && println("Error: backup version not recognized! Using Convex Backup.")
    return robust_backup(env,b,a,Alphas; prev_value=prev_value)
end

global EPSILON_OPTIMALITY = 0.001

##################################################################
#                       Osogami Backup
##################################################################

function osogami_backup(env::IPOMDP, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector})
    ### Simplify
    b = approximate_belief(b)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    Alphas = Alphas[map(a -> support_has_overlap(a, Sps), Alphas)]

    ### If terminal, return 0's
    if isterminalbelief(env,b)
        return 0.0, AlphaVector(zeros(length(Ss)), Ss, a), SparseCat([(observations(env)[1], b)], [1.0])
    end

    ### Compute worst-case transition
    _bestQ, Qos, T = osogami_get_nature(env,b,a,Alphas)

    ### Construct corresponding alpha-vector
    return osogami_get_alpha(env,b,a,Alphas,T,Qos)
end

"""
Computes a worst-case (but non least-permissive) transition based on Osogami (2015).
"""
function osogami_get_nature(env::IPOMDP, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector})
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)

    ### Build base model
    model, ps = get_model_base(env,b,a)

    ### Add Optimality contraint: Qo corresponds to optimal next action
    # ∀α,o: Qo := P(o|b) * Q(bp|o) ≥ ∑ b(s) * ∑ P(sp,o|s) * α[sp]
    @variable(model, Qo[1:length(Os)])
    for (oidx, o) in enumerate(Os)
        for alpha in Alphas
            if is_valid_alpha(env,alpha,a,o,Sps)
                @constraint(model, Qo[oidx] >=  sum(sidx -> pdf(b,Ss[sidx]) .* sum(spidx -> ps[sidx,oidx,spidx] .* alpha[Sps[spidx]], 1:length(Sps)), 1:length(Ss) ))
            end
        end
    end

    # Solve LP and extract relevant
    @objective(model, Min, sum(Qo))
    optimize!(model)
    bestQ = objective_value(model)
    Qos = JuMP.value.(Qo)
    T = JuMP.value.(ps)[:,:,:]

    return bestQ, Qos, T
end

function get_model_base(env::IPOMDP, b::DiscreteHashedBelief, a)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    As = actions(env)

    # model = Model(Clp.Optimizer; add_bridges=false)
    model = direct_generic_model(Float64, Gurobi.Optimizer(GRB_ENV[]))
    # model = Model(Gurobi.Optimizer; add_bridges=false)
    set_silent(model)
    set_string_names_on_creation(model, false)
    # set_optimizer_attribute(model, "PoolSearchMode", 2)
    # set_optimizer_attribute(model, "PoolSolutions", 10)
    # model = direct_generic_model(Float64, Gurobi.Optimizer(GRB_ENV[]))
    # set_silent(model)

    # Set up LP (Osogami, Eq. 5)
    @variable(model, 0.0 <= ps[1:length(Ss), 1:length(Os), 1:length(Sps)] <= 1.0)   # P(o,sp|s)
    for (sidx, s) in enumerate(Ss)
        @constraint(model, sum(ps[sidx,:,:]) == 1.0)
    end

    # Constraint 1: ps fall within intervals
    # Constraint 2: observation probabililies are correct
    for (sidx, s) in enumerate(Ss)
        thisT = transition(env,s,a)
        for (spidx, sp) in enumerate(Sps)
            thisInt = pdf(thisT, sp)
            @constraint(model, inf(thisInt) <= sum(ps[sidx, :, spidx]))
            @constraint(model, sum(ps[sidx, :, spidx]) <= sup(thisInt))
            for (oidx,o) in enumerate(Os)
                prob_o_given_sp = pdf(observation(env,a,sp), o) # Future work: this could also be an interval!
                @constraint(model, ps[sidx,oidx,spidx] == prob_o_given_sp * sum(ps[sidx,:,spidx]))
            end
        end
    end
    return model, ps
end

function osogami_get_alpha(env::IPOMDP, b::DiscreteHashedBelief, a, Alphas, T, Qos)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    alphastar = zeros(length(Ss))
    bos = []
    bo_probs = []
    for (oidx, o) in enumerate(Os)
        # Compute and store belief given worst-case dynamics
        bo_vector = map(spidx -> sum(sidx -> pdf(b,Ss[sidx]) * T[sidx,oidx,spidx], 1:length(Ss)), 1:length(Sps))
        prob = sum(bo_vector)
        bo = DiscreteHashedBelief(Sps, bo_vector ./ prob)
        push!(bos, (o, bo))
        push!(bo_probs, prob)

        # Find alpha-vector corresponding to observation
        this_alpha, this_min_error = nothing, Inf
        for alpha in Alphas
            error = abs(prob*dot(alpha, bo) - Qos[oidx])
            if error <= this_min_error
                this_alpha = alpha
                this_min_error = error 
            end
        end

        # Update alpha*
        for (sidx, s) in enumerate(Ss)
            alphastar[sidx] += sum(T[sidx,oidx,:] .* this_alpha[Sps])
        end
    end

    alphastar = discount(env) .* alphastar .+ map(s -> reward(env,s,a), Ss)
    alphastar = AlphaVector(alphastar, Ss, a)
    return dot(alphastar,b), alphastar, SparseCat(bos, bo_probs)
end

##################################################################
#                       Robust Backup
##################################################################

function robust_backup(env::IPOMDP, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector}; prev_value = -Inf)
    
    ### 1 - Setup   
    b = approximate_belief(b)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    Alphas = Alphas[map(alpha -> support_has_overlap(alpha, Sps), Alphas)]

    if isterminalbelief(env,b)
        return 0.0, AlphaVector(zeros(length(Ss)), Ss, a), SparseCat([(observations(env)[1], b)], [1.0])
    end

    ### 2 - Get initial worst-case nature via Osogami
    bestQ, Qos, T = osogami_get_nature(env,b,a,Alphas)

    ### 3: Skip if value equals previous value
    val_q = (sum(s -> pdf(b,s) * reward(env,s,a), Ss) + discount(env) * bestQ)
    if isapprox(val_q, prev_value; rtol=EPSILON_OPTIMALITY)
        return val_q, nothing, nothing
    end
    
    ### 5 - Construct optimal alpha-vector for each observation
    # Prune all alphas that cannot be part of the minimal optimal set
    possible_optimal_alphas = get_possibly_optimal_alphas(env, b, a, T, Qos, Alphas, epsilon=EPSILON_OPTIMALITY)
    # With all remaining alphas, construct minimal optimal set
    if length(possible_optimal_alphas) > 1
        optimal_alphas, new_T = make_smallest_optimal_subset(env,b,a,possible_optimal_alphas, bestQ)
        !isnothing(new_T) && (T = new_T)
    else
        optimal_alphas = possible_optimal_alphas
    end
    length(optimal_alphas) == 0 && println("Error: no optimal alpha-vectors found!")
    # println("Q=$bestQ, a=$a, b=$b, alphas=$optimal_alphas")

    ### 6 - Recompute worst-case dynamics
    bestQ, Qos, T = get_nature_robust(env, b, a, optimal_alphas, Alphas)

    ### 7 - (Re)compute beliefs & combined alpha-vector
    alpha = get_alpha_robust(env, b, a, optimal_alphas, bestQ, T, Qos)
    return alpha
end

function make_smallest_optimal_subset(env::IPOMDP, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector}, val::Float64)
    b = approximate_belief(b)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    L = length(Alphas)
    epsilon = 0.005
    slackify = (v -> min(v*(1+epsilon), v*(1-epsilon)))
    bigM = 100_000

    ### Heuristically sort alphas
    alphas_sorted = Dict(s => Int[] for s in Sps)
    diffs = Float64[]
    for (idx, alpha) in enumerate(Alphas)
        min_val = minimum(s -> alpha[s], Sps)
        max_val = maximum(s -> alpha[s], Sps)
        max_s = argmax(s -> alpha[s], Sps)
        push!(alphas_sorted[max_s], idx)
        push!(diffs, max_val - min_val)
    end
    # sort each bucket by diff
    alphas_combined = []
    for s in Sps
        aidxs = alphas_sorted[s]
        perm = sortperm(diffs[aidxs])
        push!(alphas_combined, Alphas[aidxs[perm]])
    end
    # order buckets so the one with the smallest diff comes first
    vs = sort(alphas_combined, by = v -> isempty(v) ? Inf : (maximum(s -> v[1][s], Sps) - minimum(s -> v[1][s], Sps)))
    # round-robin merge
    Alphas = [v[i] for i in 1:maximum(length.(vs)) for v in vs if length(v) ≥ i]

    ### Build Model:
    model, ps = get_model_base(env, b, a)
    @variable(model, Qoalpha[1:length(Os), 1:length(Alphas)])
    @variable(model, alpha_valid[1:length(Alphas)], Bin)
    @variable(model, Qo[1:length(Os)])
    constraints = []
    for (oidx, o) in enumerate(Os)
        for (alphaidx, alpha) in enumerate(Alphas)
            @constraint(model, Qoalpha[oidx, alphaidx] >= sum(sidx -> pdf(b,Ss[sidx]) .* sum(spidx -> ps[sidx,oidx,spidx] .* alpha[Sps[spidx]], 1:length(Sps)), 1:length(Ss) ))
            push!(constraints, @constraint(model, Qo[oidx] >= Qoalpha[oidx, alphaidx] - bigM * (1-alpha_valid[alphaidx])))
        end
    end
    @objective(model, Min, sum(Qo))
    i = 0
    for nmbr_alphas in 1:(L-1)
        for indexes in collect(combinations(1:L, nmbr_alphas))
            i+=1
            # build model, restricted to smaller set of alpha-vectors
            @constraint(model, temp_constraint[i = indexes], alpha_valid[i] == 1)
            optimize!(model)
            
            # return first set that achieves optimal result
            if objective_value(model) > slackify(val)
                return Alphas[indexes], nothing
            end
            
            #cleanup for next iteration
            for c in values(temp_constraint) 
                delete(model, c)
            end
            unregister(model, :temp_constraint)
        end
    end
    return Alphas, nothing
end

# Note: prunes per observation (otherwise we can't compare to expected value)
function prune_possibly_optimal_alphas(Sps, Qo, Alphas, epsilon=0.005)
    Qo = min(Qo * (1+epsilon), Qo * (1-epsilon))
    alphas_mask = trues(length(Alphas))
    for (aidx1, alpha1) in enumerate(Alphas)
        !alphas_mask[aidx1] && continue

            vals1 = alpha1[Sps] .- Qo
            if all(vals1 .>= 0)
                return [alpha1]
            end
            mask = vals1 .>= 0
            relevant_states = Sps[mask]
            if isempty(relevant_states)
                alphas_mask[aidx1] = false
                continue
            end

        for (aidx2, alpha2) in enumerate(Alphas)
            aidx1 == aidx2 && continue
            !alphas_mask[aidx2] && continue

            vals1, vals2 = alpha1[relevant_states], alpha2[relevant_states]
            best_state_idx = argmax(vals1)
            if all(vals1 .<= vals2 .* (1+epsilon) ) && (vals1[best_state_idx] .< vals2[best_state_idx])
                alphas_mask[aidx1] = false
                break
            end

        end
    end
    return Alphas[alphas_mask]
end

function prune_epsilon_close_alphas(Alphas, epsilon)
    alphas_mask = trues(length(Alphas))
    for (aidx1, alpha1) in enumerate(Alphas)
        !alphas_mask[aidx1] && continue
        for (aidx2, alpha2) in enumerate(Alphas)
            aidx1 == aidx2 && continue
            !alphas_mask[aidx2] && continue

            if  (alpha1.states == alpha2.states)
                states = alpha1.states
                if maximum(abs.(alpha1[states] .- alpha2[states])) < epsilon
                    max_value_diff = minimum(alpha1[states]) - minimum(alpha2[states])
                    max_value_diff > 0 ? (alphas_mask[aidx1] = false; break) : (alphas_mask[aidx2] = false)
                end
            end
        end
    end
    return Alphas[alphas_mask]        
end

function get_possibly_optimal_alphas(env::IPOMDP, b::DiscreteHashedBelief, a, T, Qos, Alphas::Vector{<:AlphaVector}; epsilon = 0.001)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    bos = []
    bo_probs = []
    all_optimal_alphas = Set{AlphaVector}([])
    for (oidx, o) in enumerate(Os)
        ### Compute worst-case belief given observation
        bo_vector = map(spidx -> sum(sidx -> pdf(b,Ss[sidx]) * T[sidx,oidx,spidx], 1:length(Ss)), 1:length(Sps))
        prob = sum(bo_vector)
        bo = DiscreteHashedBelief(Sps, bo_vector ./ prob)
        push!(bos, (o, bo))
        push!(bo_probs, prob)

        ### Find (epsilon-) optimal alpha-vectors
        Qo = Qos[oidx] / prob
        alphas, min_error = [], Inf
        for (alphaidx, alpha) in enumerate(Alphas)
            error = abs((dot(alpha, bo) - Qo) / (Qo + epsilon))
            if error <= min_error - epsilon
                alphas = []
                push!(alphas, alpha)
                min_error = error
            elseif error <= min_error + epsilon
                push!(alphas, alpha)
                min_error = min(min_error, error)
            end
        end

        ### Prune set of alphas
        alphas = prune_possibly_optimal_alphas(bo.state_list, Qo, alphas, epsilon)
        union!(all_optimal_alphas, Set(alphas))
    end
    all_optimal_alphas = prune_epsilon_close_alphas(collect(all_optimal_alphas), epsilon)
    # pruned_optimal_alphas = Set()

    return all_optimal_alphas
end

### 
#   Backup 
###

function get_alpha_robust(env::IPOMDP, b::DiscreteHashedBelief, a, optimal_alphas, bestQ, T, Qos)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    alphastar = zeros(length(Ss))
    bos = []
    bo_probs = []
    for (oidx, o) in enumerate(Os)
        ### Compute and store belief given worst-case dynamics
        bo_vector = map(spidx -> sum(sidx -> pdf(b,Ss[sidx]) * T[sidx,oidx,spidx], 1:length(Ss)), 1:length(Sps))
        prob = sum(bo_vector)
        bo = DiscreteHashedBelief(Sps, bo_vector ./ prob)
        push!(bos, (o, bo))
        push!(bo_probs, prob)

        ### best alpha-vector for this belief
        if length(optimal_alphas) == 1
            this_alpha = optimal_alphas[1]
        else
            best_Qo = maximum(alpha -> dot(alpha, bo), optimal_alphas)
            best_Qo_slackified = min(best_Qo * (1+EPSILON_OPTIMALITY), best_Qo * (1-EPSILON_OPTIMALITY))
            this_optimal_alphas = filter(alpha -> dot(alpha, bo) >= best_Qo_slackified, optimal_alphas)
            # println("o=$o, optimal_alphas=$this_optimal_alphas")
            probs = stochastic_choice_alphas(bo, this_optimal_alphas)
            this_alpha = zeros(length(Sps))
            for (aidx, alpha) in enumerate(this_optimal_alphas)
                this_alpha += probs[aidx] .* alpha[Sps]
            end
            this_alpha = AlphaVector(this_alpha, Sps, a)
        end
        
        ### Update alpha*
        for (sidx, s) in enumerate(Ss)
            for (spidx, sp) in enumerate(Sps)
                p = T[sidx,oidx,spidx]
                # println("p=$p, val=$(this_alpha[sp])")
                (p > 0.001) && (alphastar[sidx] += p * this_alpha[sp])
            end
        end
        # println("\n")
    end
    ### Refactor
    alphastar = discount(env) .* alphastar .+ map(s -> reward(env,s,a), Ss)
    alphastar = AlphaVector(alphastar, Ss, a)

    val_q = (sum(s -> pdf(b,s) * reward(env,s,a), Ss) + discount(env) * bestQ)
    if abs((val_q - dot(alphastar, b)) / val_q) > 0.05 && val_q > 0.01
        println("Error: value of α* ($(dot(alphastar, b))) is not equal to worst-case value ($(val_q))")
        println("b=$b, alphastar=$(alphastar), $b")
        println("================")
    end

    return dot(alphastar,b), alphastar, SparseCat(bos, bo_probs)
end

function get_nature_robust(env::IPOMDP, b::DiscreteHashedBelief, a, optimal_alphas, Alphas)
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    min_val = minimum(alpha -> minimum(alpha.α), Alphas)
    model, ps = get_model_base(env,b,a)
    @variable(model, Qo[1:length(Os)] >= min_val)
    @variable(model, Qo_suboptimal[1:length(Os)] >= min_val) # value of the best sub-optimal alpha-vector for each observation.
    @variable(model, max_Qo_suboptimal >= min_val)
    for (oidx, o) in enumerate(Os)
        @constraint(model, max_Qo_suboptimal >= Qo_suboptimal[oidx])
        for alpha in Alphas
            @constraint(model, Qo[oidx] >= sum(sidx -> pdf(b,Ss[sidx]) .* sum(spidx -> ps[sidx,oidx,spidx] .* alpha[Sps[spidx]], 1:length(Sps)), 1:length(Ss) ))
            if !(alpha in optimal_alphas)
                @constraint(model, Qo_suboptimal[oidx] >= sum(sidx -> pdf(b,Ss[sidx]) .* sum(spidx -> ps[sidx,oidx,spidx] .* alpha[Sps[spidx]], 1:length(Sps)), 1:length(Ss) ))
            end
        end
    end
    bigM = 1_000
    @objective(model, Min, sum(bigM .* Qo) + sum(Qo_suboptimal))
    optimize!(model)
    T = JuMP.value.(ps)[:,:,:]
    Qos = JuMP.value.(Qo)
    return sum(Qos), Qos, T
end

##################################################################
#                       POMDP Backup
##################################################################

function backup(env::POMDP, b::DiscreteHashedBelief, a, Alphas::Vector{<:AlphaVector})
    # Get relevant variables:
    Ss, Sps, Os = get_relevant_sets(env,support(b), a)
    Alphas = Alphas[map(a -> support_has_overlap(a, Sps), Alphas)]

    if isterminalbelief(env,b)
        return 0.0, AlphaVector(zeros(length(Ss)), Ss, a), SparseCat([(observations(env)[1], b)], [1.0])
    end

    Prob_o_given_sp = [pdf(observation(env,a,sp), o) for (o,sp) in Iterators.product(Os, Sps)]
    Prob_sp_given_s = [pdf(transition(env,s,a), sp) for (sp,s) in Iterators.product(Sps, Ss)]
    
    Bs, B_probs = [], []
    alphastar = map(s -> reward(env,s,a), Ss)
    for (oidx,o) in enumerate(Os)
        bo_vector = map(spidx -> sum(sidx -> pdf(b, Ss[sidx]) * Prob_sp_given_s[spidx,sidx] * Prob_o_given_sp[oidx,spidx], 1:length(Ss)), 1:length(Sps))
        prob = sum(bo_vector)
        bo_vector = bo_vector ./ prob
        bo = DiscreteHashedBelief(Sps, bo_vector)
        push!(Bs, (o,bo))
        push!(B_probs, prob)
        alpha_o =  argmax(alpha -> dot(alpha,bo), Alphas)
        for (sidx, s) in enumerate(Ss)
            alphastar[sidx] += discount(env) * sum(Prob_sp_given_s[:,sidx] .* Prob_o_given_sp[oidx,:] .* alpha_o[Sps])
        end
    end

    alpha = AlphaVector(alphastar, Ss, a)

    return dot(alpha, b), alpha, SparseCat(Bs, B_probs)
end