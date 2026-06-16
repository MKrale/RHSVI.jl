using RHSVI
using Test
using RPOMDPs, POMDPs, POMDPTools
include("./tiger.jl")

@testset "RHSVI.jl" begin
    rpomdps = [TigerPOMDP(), Index_IPOMDP(RPOMDPs.ToyRPOMDP()), Index_IPOMDP(Test_Random()), Index_IPOMDP(ConfidencePOMDP(TigerPOMDP(), 0.1, AdditiveAbs()))]
    values_min = [69, 0.38, 19.3]
    values_max = [70, 0.4, 20]
    for idx in eachindex(rpomdps)
        policy, info = POMDPTools.solve_info(
            RHSVISolver(max_iters=100, max_depth=100), 
            rpomdps[idx]
        )
        value = POMDPs.value(policy, POMDPs.initialstate(rpomdps[idx]))
        @test (value >= values_min[idx])
        @test (value <= values_max[idx])
    end
end