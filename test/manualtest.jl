using RHSVI,RPOMDPs
using POMDPs, .RPOMDPs, ArgParse, POMDPTools, JSON, POMDPModels, RockSample, MCTS, Statistics
# The following clash: uncomment only one!
using Profile, FlameGraphs, ProfileSVG
using D3Trees

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        # required = true
        default = "Tiger"

    "--rtype"
        help = "Type of robustness used. Options: full, mid, rmdp, or all"
        default = "full"

    "--solvers"
        help = "Solver to be run. Availble options: RQMDP, RFIB. (default: run All)"
        default = "all"

    "--path"
        help = "File path for data output."
        default = "Data/Tests/"

    "--filename"
        help = "Filename (default: generated automatically)"
        default = ""

    "--intervaltype"
        help = "Method of converting POMDP to IPOMDP. Only used when env is not already robust. Options: add_rel, add_abs, mult"
        default = "add_rel"

    "--intervaldist"
        help = "Distance used to convert POMDP to IPOMDP. Only used when env is not already robust."
        arg_type = Float64
        default = 0.5
    
    "--evaltime"
        help = "Max time for evalutation stage"
        arg_type = Float64
        default = 30.0

    "--evalnmbr"
        help = "Nmbr of times evaluation is run"
        arg_type = Int 
        default = 1

    "--timeout", "-t"
        help = "Time untill timeout."
        arg_type = Float64
        default = 30.0

    "--discount"
        help = "Discount factor"
        arg_type = Float64
        default = 0.95

    "--precompile"
        help = "Option to precomile all code by running at low horizon. Particularly relevant for small environments. (default: true)"
        arg_type = Bool 
        default = false
end

### For running from CL: ###

parsed_args = parse_args(ARGS, s)
env_names = [parsed_args["env"]]
rtypes = [parsed_args["rtype"]]
rtypes == ["all"] && (rtypes = ["full", "maxent", "mid", "rmdp"])
solver_names = [parsed_args["solvers"]]
solver_names == ["all"] && (solver_names = ["RFIB", "RHSVI"])
path = parsed_args["path"]
filename = parsed_args["filename"]
evaltime = parsed_args["evaltime"]
evalnmbr = parsed_args["evalnmbr"]
intervaldist = parsed_args["intervaldist"]
intervaltype_str = parsed_args["intervaltype"]
timeout = parsed_args["timeout"]
discount = parsed_args["discount"]
discount_str = string(discount)[3:end]
precompile = parsed_args["precompile"]

### For runnning in REPL: ###

# env_names = ["Test_Random"]
# rtypes = ["full"]
# solver_names = ["RHSVI"]
# path = "./Data"
# filename = ""
# evaltime = 15.0
# evalnmbr = 1
# intervaldist = 0.5
# intervaltype_str = "add_rel"
# timeout = 2.0

# discount = 0.95 #TODO: pipe this to all relevant places!




##################################################################
#                           Set Solvers 
##################################################################

solvers, solverargs, precomp_solverargs = [], [], []
bounds_steps, precision = 10_000, 2e-2

if "RQMDP" in solver_names
    push!(solvers, RQMDPSolver)
    push!(solverargs, (name="RQMDP", sargs=(max_iterations=bounds_steps, precision=precision, max_time=timeout), pargs=(),))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "RFIB" in solver_names
    push!(solvers, RFIBSolver)
    push!(solverargs, (name="RFIB", sargs=(max_iterations=bounds_steps, precision=precision, max_time=timeout), pargs=(),))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "RPBVI" in solver_names
    push!(solvers, RPBVISolver)
    push!(solverargs, (name="RPBVI", sargs=(max_iterations=25, precision=precision,), pargs=(),)) #TODO: add maxtime
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "RHSVI" in solver_names
    push!(solvers, RHSVISolver)
    push!(solverargs, (name="RHSVI", sargs=(max_time=timeout,epsilon=precision), pargs=()))
    push!(precomp_solverargs, ( sargs=(max_time=5.0,), pargs=()))
end

isempty(solvers) && println("Warning: no solver recognized!")

##################################################################
#                       Selecting env 
##################################################################

POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
POMDPs.discount(M::RockSample.RockSamplePOMDP) = discount
intervaltype_str == "add_rel" && (intervaltype = AdditiveRel())
intervaltype_str == "add_abs" && (intervaltype = AdditiveAbs())
intervaltype_str == "mult" && (intervaltype = Multiplicative())

envs, envsargs = [], []

if "Toy" in env_names
    push!(envs, Toy())         # Ignores discount: always ~1.0
    push!(envsargs, (name="Toy",))
end
if "Toy_mid" in env_names
    push!(envs, Toy_mid())         # Ignores discount: always ~1.0
    push!(envsargs, (name="Toy_mid",))
end
if "Toy_rmdp" in env_names
    push!(envs, Toy_rmdp())         # Ignores discount: always ~1.0
    push!(envsargs, (name="Toy",))
end
if "ChainInf" in env_names
    push!(envs, ChainInf(discount=discount))
    push!(envsargs, (name="ChainInf",))
end
if "Chain5" in env_names
    push!(envs, ChainN(N=5, discount=discount))
    push!(envsargs, (name="Chain10",))
end
if "Chain10" in env_names
    push!(envs, ChainN(N=10, discount=discount))
    push!(envsargs, (name="Chain10",))
end
if "Chain15" in env_names
    push!(envs, ChainN(N=15, discount=discount))
    push!(envsargs, (name="Chain15",))
end
if "Chain20" in env_names
    push!(envs, ChainN(N=20, discount=discount))
    push!(envsargs, (name="Chain20",))
end
if "Chain25" in env_names
    push!(envs, ChainN(N=25, discount=discount))
    push!(envsargs, (name="Chain25",))
end
if "Machine" in env_names
    push!(envs, Machine(discount=0.99))      # Ignores discount: always 0.99
    push!(envsargs, (name="Machine",))
end
if "Tiger" in env_names
    base = POMDPModels.TigerPOMDP()
    base.discount_factor = discount
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="Tiger",))
end
if "RockSample5" in env_names
    map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
    base = RockSample.RockSamplePOMDP(map_size, rock_pos)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="RockSample5",))
end
if "HeavenOrHell5" in env_names
    # base = HeavenOrHellClassic()
    base = HeavenOrHell(size=5)#discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="HeavenOrHell5",))
end
if "HeavenOrHell10" in env_names
    # base = HeavenOrHellClassic()
    base = HeavenOrHell(size=10)#discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="HeavenOrHell10",))
end
if "Aloha10" in env_names
    base = Sparse_aloha10(discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="Aloha10",))
end
if "K_out_of_N1" in env_names
    base = K_out_of_N(N=1, K=1, discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="K_out_of_N1",))
end
if "K_out_of_N2" in env_names
    base = K_out_of_N(N=2, K=2, discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="K_out_of_N2",))
end
if "MiniHallway" in env_names
    base = MiniHallway() # discount cannot be changed...
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="MiniHallway",))
end
if "FrozenLake" in env_names
    base = FrozenLakeSmall(discount=discount)
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="FrozenLake",))
end
if "CNC_Detection" in env_names
    base = CNC_Detection(discount=discount)
    push!(envs, base)
    push!(envsargs, (name="CNC_Detection",))
end
if "Replacement" in env_names
    base = MachineReplacement(discount=discount) # Default discount is 0.9
    push!(envs, ConfidencePOMDP(base, intervaldist, intervaltype))
    push!(envsargs, (name="Replacement",))
end
if "Test_Backup" in env_names
    push!(envs, Test_Backup())         # Ignores discount: always ~1.0
    push!(envsargs, (name="Test_Backup",))
end
if "Test_Random" in env_names
    push!(envs, Test_Random())         # Ignores discount: always ~1.0
    push!(envsargs, (name="Test_Random",))
end

isempty(envs) && println("Warning: $env_names environment not recognized!")

##################################################################
#                      Other
##################################################################

function get_simplified_model(env, rtype)
    if rtype == "full"
        return env
    elseif rtype == "mid"
        return to_mid_POMDP(env)
    elseif rtype == "rmdp"
        return to_rmdp_POMDP(env)
    elseif rtype == "maxent"
        return to_maxent_POMDP(env)
    end
    println("Error: Rtype not recognized!")
end

##################################################################
#                           Run 
##################################################################

sims, steps = 1_000, 1_000
verbose = true

for (m_idx,(env, envargs)) in enumerate(zip(envs, envsargs))
    verbose && println("Testing in $(envargs.name) environment")



    for (s_idx,(solver, solverarg)) in enumerate(zip(solvers, solverargs))
        for rtype in rtypes

            ### Precompile
            verbose && println("\nPrecompiling...")
            precompsolver = solver(;precomp_solverargs[s_idx].sargs...)
            precomp_env_solver = get_simplified_model(env,rtype)
            _policy, _info = solve_info(precompsolver, precomp_env_solver; precomp_solverargs[s_idx].pargs...) #Force precompile

            # Run policy
            verbose && println("\nRunning $(solverarg.name) using $rtype robustness")
            thissolver = solver(;solverarg.sargs...)
            env_solver = get_simplified_model(env,rtype)

            Profile.clear()
            t_solve = @elapsed begin
                @profile policy, info = POMDPTools.solve_info(thissolver, env_solver; solverarg.pargs...)
                # policy, info = POMDPTools.solve_info(thissolver, env_solver; solverarg.pargs...)
            end

            # function make_left_heavy!(node)
            #     # Sort children so heaviest (by .count) is on the left
            #     node.children = sort(node.children; by = c -> -c.count)
            #     # Recurse
            #     for child in node.children
            #         make_left_heavy!(child)
            #     end
            #     return node
            # end

            fg = flamegraph(Profile.fetch(); norepl=true, combine=true)
            ProfileSVG.save("flamegraph.svg", fg; width=3600, fontsize=10, maxdepth=40, maxframes=10_000)

            (info isa Nothing) ? val = POMDPs.value(policy, POMDPs.initialstate(env_solver)) : val = info.value
            verbose && println("Value $val (computed in $t_solve seconds)")

            # Test Policy
            val_adv = 0.0
            env_adv = get_model_adversary(env,policy)
            # t_build = @elapsed begin
            #     env_adv = get_model_adversary(env,policy)
            # end
            binit_adv = initialstate(env_adv)
            total_iterations = 10_000_000
            # solver_adv = MCTSSolver(max_time=evaltime, n_iterations=total_iterations, depth=250, exploration_constant=sqrt(2), estimate_value=-val, init_Q=-val, init_N=2)
            f_leaf = (env, sn, _depth) -> -POMDPs.value(policy, sn.x)
            f_init = (env, sn, _a) -> -POMDPs.value(policy, sn.x)
            # f_init = (env, sn, a) -> -POMDPs.value(policy)
            # solver_adv = MCTSSolver(max_time=evaltime, n_iterations=total_iterations, depth=250, exploration_constant=2.0, estimate_value=f_leaf, init_Q=f_init, init_N=0, enable_tree_vis=true)

            # policy_adv = solve(solver_adv, env_adv)
            t_eval = @elapsed begin
                vals_adv = []
                for i in 1:evalnmbr
                    this_val_adv = 0.0
                    for (sinit_adv, prob) in weighted_iterator(binit_adv)
                        this_max_time = evaltime*prob
                        this_n_iterations = Int(ceil(total_iterations*prob))
                        # solver_adv = MCTSSolver(max_time=this_max_time, n_iterations=this_n_iterations, depth=100, exploration_constant=25.0, estimate_value=f_leaf, init_Q=f_init, init_N=0)
                        solver_adv = MCTSSolver(max_time=this_max_time, n_iterations=this_n_iterations, depth=100, exploration_constant=5.0, estimate_value=f_leaf, init_Q=f_init, init_N=0, enable_tree_vis=true)
                        policy_adv = solve(solver_adv, env_adv)
                        a = action(policy_adv, sinit_adv)
                        this_val_adv -= prob * value(policy_adv, sinit_adv)
                        inchrome(D3Tree(policy_adv, sinit_adv))
                    end
                    push!(vals_adv, this_val_adv)
                    println(this_val_adv)
                end
            end
            val_adv, std_adv = minimum(vals_adv), Statistics.std(vals_adv)
            verbose && println("Adversarial value $(minimum(vals_adv)) (computed in $t_eval seconds)")
            data_dict = Dict(
                # "env" => env_name,
                "env_full" => envargs.name,
                "rtype" => rtype,
                "states" => length(states(env)),
                "actions" => length(actions(env)),
                "observations" => length(observations(env)),
                
                "intervaltype" => intervaltype_str,
                "intervaldist" => intervaldist,

                "solver" => solverarg.name,
                "solvertime" => t_solve,
                "evaltime"  => t_eval,
                "value_sol" => val,
                "value_adv" => val_adv,
                "std_adv" => std_adv
            )
            json_str = JSON.json(data_dict)
            if filename == ""
                thisfilename =  path * "RPolicyTest_$(envargs.name)_$(solver_names[s_idx])_$(rtype).json"
            else
                thisfilename = path * filename * solverarg.name
            end
            open(thisfilename, "w") do file
                write(file, json_str)
            end
        end
    end
end
