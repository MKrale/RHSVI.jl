module RHSVI
    using POMDPs, POMDPTools, Distributions, Random, Memoize, LRUCache, Combinatorics, IntervalArithmetic
    using JuMP, Clp, Gurobi
    using RPOMDPs      

    # Setting up Gurobi and suppressing printing
    const GRB_ENV = Ref{Gurobi.Env}()
    function __init__()
        oldstd = stdout
        redirect_stdout(devnull)
        GRB_ENV[] = Gurobi.Env()
        redirect_stdout(oldstd)
        return
    end

    # Utils
    include("Utils/Beliefs.jl")

    # Solver
    include("RobustAlphaVectors.jl")
    export AlphaVector, AlphaVectorPolicy
    include("RobustBackup.jl")
    include("RQMDP.jl")
    export RQMDPSolver, RQMDPPlanner, RFIBSolver, RFIBPlanner
    include("RPBVI.jl")
    export RPBVISolver, backup
    include("RHSVI_solver.jl")
    export RHSVISolver
end