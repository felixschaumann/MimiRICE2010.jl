########################################################################################################
# Quick check: what are max costs for flow-only adaptation (no stock)?
# Run unconstrained to find the natural cost level, then we can set a binding cap.
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using CSV
using DataFrames

include("create_rice.jl")
include(joinpath(@__DIR__, "../src/", "optimisation_functions.jl"))
include(joinpath(@__DIR__, "../src/", "multistart_optimisation.jl"))

ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

println("Running flow-only unconstrained (SBPLX, 600s)...")
result = optimize_rice(
    :LN_SBPLX, n_opt_periods, 600, 1e-12, backstop_prices;
    run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=false, cbudget=nothing, cost_cap=nothing,
    ext_starting_points=nothing
)

model = result[7]
utility = result[9]
costs = model[:neteconomy, :TOTAL_COST]
em = model[:emissions, :EIND]
cca = model[:emissions, :CCA][end]

println("\nUtility: $(round(utility, sigdigits=8))")
println("CCA: $(round(cca, digits=1))")
println("Max cost (any region/period): $(round(maximum(costs), sigdigits=4))")
println("\nCost time series (max across regions per period):")
for t in 1:size(costs, 1)
    row_max = maximum(costs[t, :])
    if row_max > 0.001
        println("  t=$t: rich=$(round(costs[t,1], sigdigits=4))  poor=$(round(costs[t,2], sigdigits=4))  max=$(round(row_max, sigdigits=4))")
    end
end

println("\nFor comparison, stock+flow unconstrained max cost ≈ 0.0097")
println("Suggested cap levels to test: $(round(maximum(costs) * 0.9, sigdigits=3)), $(round(maximum(costs) * 0.8, sigdigits=3)), $(round(maximum(costs) * 0.7, sigdigits=3))")
