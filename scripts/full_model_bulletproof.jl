########################################################################################################
# Bulletproof full_model run: 4 configs with warm-starting and multi-run verification
#
# - Config 1: mitigation only (3 runs)
# - Config 2f: warm-started from Config 1 solution (pad with zeros for adaptation) + default start (3 runs)
# - Config 3f: warm-started from Config 2f best (3 runs)
# - Config 4f: warm-started from Config 3f best (3 runs)
#
# Key fix: Negishi weights computed once from Config 1 (standard model) and shared across all configs.
# This ensures Config 2f utility >= Config 1 (adding adaptation can't hurt when welfare is identical).
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using DataFrames
using Statistics
using Dates

include("create_rice.jl")
include(joinpath(@__DIR__, "../src/", "optimisation_functions.jl"))

#------------------------------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------------------------------
ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
optimization_algorithm = :LN_AUGLAG
stop_time = 2400       # generous time per run
tolerance = 1e-15
carbon_budget = 1022.5950757761052
cost_cap = 0.008
n_runs = 3             # runs per config for verification

results_base = joinpath(@__DIR__, "../results", "FullModel_Bulletproof")
mkpath(results_base)

# Get backstop prices and compute base Negishi weights ONCE from standard model
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000
base_alpha = compute_negishi_weights(backstop_rice)
println("Base Negishi weights computed from standard (no-adaptation) model")
println("  α[1,:] = $(round.(base_alpha[1,:], sigdigits=6))")
println("  α[10,:] = $(round.(base_alpha[10,:], sigdigits=6))")

function compute_share_rich(model)
    eind = model[:emissions, :EIND]
    total_rich = sum(eind[:, 1])
    total_poor = sum(eind[:, 2])
    return total_rich / (total_rich + total_poor) * 100.0
end

#------------------------------------------------------------------------------------------------------
# Config 1: Mitigation only
#------------------------------------------------------------------------------------------------------
println("\n" * "="^70)
println("CONFIG 1: Mitigation only ($n_runs runs)")
println("="^70)

c1_results = []
for i in 1:n_runs
    t0 = time()
    result = optimize_rice(
        optimization_algorithm, n_opt_periods, stop_time, tolerance, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=false, stock_ad=false, cbudget=nothing, cost_cap=nothing,
        alpha_override=base_alpha
    )
    elapsed = round(time() - t0, digits=1)
    opt_model = result[7]
    share = compute_share_rich(opt_model)
    utility = result[9]
    println("  Run $i: utility=$(round(utility, sigdigits=8)), share_rich=$(round(share, digits=3))%, convergence=$(result[8]) ($(elapsed)s)")
    push!(c1_results, (utility=utility, share=share, policy=result[1], model=opt_model, convergence=result[8]))
end

# Best Config 1
best_c1 = argmax([r.utility for r in c1_results])
c1_policy = c1_results[best_c1].policy
c1_utility = c1_results[best_c1].utility
c1_share = c1_results[best_c1].share
println("\nBest Config 1: utility=$(round(c1_utility, sigdigits=8)), share_rich=$(round(c1_share, digits=3))%")
println("Spread: utility=[$(round(minimum(r.utility for r in c1_results), sigdigits=8)), $(round(maximum(r.utility for r in c1_results), sigdigits=8))], share=[$(round(minimum(r.share for r in c1_results), digits=3)), $(round(maximum(r.share for r in c1_results), digits=3))]")

#------------------------------------------------------------------------------------------------------
# Config 2f: Flow adaptation — warm-start from Config 1 + default start
#------------------------------------------------------------------------------------------------------
println("\n" * "="^70)
println("CONFIG 2f: + Flow adaptation ($n_runs runs, warm-started from Config 1)")
println("="^70)

# Create warm-start: pad Config 1 mitigation with zeros for adaptation
n_regions = 2
warm_start_2f = vcat(c1_policy, zeros(n_opt_periods * n_regions))

c2_results = []
for i in 1:n_runs
    # Alternate between warm-start from C1 and default
    sp = (i <= 2) ? warm_start_2f : nothing
    label = (i <= 2) ? "warm-start" : "default"

    t0 = time()
    result = optimize_rice(
        optimization_algorithm, n_opt_periods, stop_time, tolerance, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=false, cbudget=nothing, cost_cap=nothing,
        ext_starting_points=sp, alpha_override=base_alpha
    )
    elapsed = round(time() - t0, digits=1)
    opt_model = result[7]
    share = compute_share_rich(opt_model)
    utility = result[9]
    max_cost = maximum(opt_model[:neteconomy, :TOTAL_COST])
    println("  Run $i ($label): utility=$(round(utility, sigdigits=8)), share_rich=$(round(share, digits=3))%, max_cost=$(round(max_cost, sigdigits=4)) ($(elapsed)s)")

    # Sanity check: utility must be >= Config 1
    if utility < c1_utility - 0.01
        println("  ⚠ WARNING: utility < Config 1 ($(round(c1_utility, sigdigits=8))) — convergence issue!")
    end

    push!(c2_results, (utility=utility, share=share, policy=result[1], model=opt_model, convergence=result[8]))
end

best_c2 = argmax([r.utility for r in c2_results])
c2_policy = c2_results[best_c2].policy
c2_utility = c2_results[best_c2].utility
c2_share = c2_results[best_c2].share
println("\nBest Config 2f: utility=$(round(c2_utility, sigdigits=8)), share_rich=$(round(c2_share, digits=3))%")
println("Spread: utility=[$(round(minimum(r.utility for r in c2_results), sigdigits=8)), $(round(maximum(r.utility for r in c2_results), sigdigits=8))], share=[$(round(minimum(r.share for r in c2_results), digits=3)), $(round(maximum(r.share for r in c2_results), digits=3))]")

#------------------------------------------------------------------------------------------------------
# Config 3f: + Cost cap — warm-start from Config 2f
#------------------------------------------------------------------------------------------------------
println("\n" * "="^70)
println("CONFIG 3f: + Cost cap 0.8% ($n_runs runs, warm-started from Config 2f)")
println("="^70)

c3_results = []
for i in 1:n_runs
    sp = (i <= 2) ? c2_policy : nothing
    label = (i <= 2) ? "warm-start" : "default"

    t0 = time()
    result = optimize_rice(
        optimization_algorithm, n_opt_periods, stop_time, tolerance, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=false, cbudget=nothing, cost_cap=cost_cap,
        ext_starting_points=sp, alpha_override=base_alpha
    )
    elapsed = round(time() - t0, digits=1)
    opt_model = result[7]
    share = compute_share_rich(opt_model)
    utility = result[9]
    max_cost = maximum(opt_model[:neteconomy, :TOTAL_COST])
    feasible = max_cost <= cost_cap + 1e-4
    println("  Run $i ($label): utility=$(round(utility, sigdigits=8)), share_rich=$(round(share, digits=3))%, max_cost=$(round(max_cost, sigdigits=4)), feasible=$feasible ($(elapsed)s)")
    push!(c3_results, (utility=utility, share=share, policy=result[1], model=opt_model, convergence=result[8], feasible=feasible, max_cost=max_cost))
end

# Best feasible Config 3f
feasible_c3 = filter(r -> r.feasible, c3_results)
if !isempty(feasible_c3)
    best_c3_idx = argmax([r.utility for r in feasible_c3])
    c3_policy = feasible_c3[best_c3_idx].policy
    c3_utility = feasible_c3[best_c3_idx].utility
    c3_share = feasible_c3[best_c3_idx].share
    println("\nBest feasible Config 3f: utility=$(round(c3_utility, sigdigits=8)), share_rich=$(round(c3_share, digits=3))%")
    println("Feasible: $(length(feasible_c3))/$n_runs")
else
    println("\n⚠ No feasible Config 3f solutions! Using best overall.")
    best_c3_idx = argmax([r.utility for r in c3_results])
    c3_policy = c3_results[best_c3_idx].policy
    c3_utility = c3_results[best_c3_idx].utility
    c3_share = c3_results[best_c3_idx].share
end

#------------------------------------------------------------------------------------------------------
# Config 4f: + Carbon budget — warm-start from Config 3f
#------------------------------------------------------------------------------------------------------
println("\n" * "="^70)
println("CONFIG 4f: + Carbon budget 1022 GtC ($n_runs runs, warm-started from Config 3f)")
println("="^70)

c4_results = []
for i in 1:n_runs
    sp = (i <= 2) ? c3_policy : nothing
    label = (i <= 2) ? "warm-start" : "default"

    t0 = time()
    result = optimize_rice(
        optimization_algorithm, n_opt_periods, stop_time, tolerance, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=false, cbudget=carbon_budget, cost_cap=cost_cap,
        ext_starting_points=sp, alpha_override=base_alpha
    )
    elapsed = round(time() - t0, digits=1)
    opt_model = result[7]
    share = compute_share_rich(opt_model)
    utility = result[9]
    max_cost = maximum(opt_model[:neteconomy, :TOTAL_COST])
    cca = opt_model[:emissions, :CCA][end]
    feasible = max_cost <= cost_cap + 1e-4 && cca <= carbon_budget + 1.0
    println("  Run $i ($label): utility=$(round(utility, sigdigits=8)), share_rich=$(round(share, digits=3))%, max_cost=$(round(max_cost, sigdigits=4)), CCA=$(round(cca, digits=1)), feasible=$feasible ($(elapsed)s)")
    push!(c4_results, (utility=utility, share=share, policy=result[1], model=opt_model, convergence=result[8], feasible=feasible, max_cost=max_cost, cca=cca))
end

feasible_c4 = filter(r -> r.feasible, c4_results)
if !isempty(feasible_c4)
    best_c4_idx = argmax([r.utility for r in feasible_c4])
    c4_utility = feasible_c4[best_c4_idx].utility
    c4_share = feasible_c4[best_c4_idx].share
    println("\nBest feasible Config 4f: utility=$(round(c4_utility, sigdigits=8)), share_rich=$(round(c4_share, digits=3))%")
    println("Feasible: $(length(feasible_c4))/$n_runs")
else
    println("\n⚠ No feasible Config 4f solutions!")
end

#------------------------------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------------------------------
println("\n\n" * "="^70)
println("SUMMARY: Full Model with Shared Negishi Weights")
println("="^70)
println("Config     | Share_rich | Utility     | Spread(share) | Notes")
println("-"^70)

# Config 1
c1_spread = round(maximum(r.share for r in c1_results) - minimum(r.share for r in c1_results), digits=3)
println("Config 1   | $(lpad(round(c1_share, digits=3), 7))%  | $(round(c1_utility, sigdigits=8)) | ±$(c1_spread) pp    |")

# Config 2f
c2_spread = round(maximum(r.share for r in c2_results) - minimum(r.share for r in c2_results), digits=3)
println("Config 2f  | $(lpad(round(c2_share, digits=3), 7))%  | $(round(c2_utility, sigdigits=8)) | ±$(c2_spread) pp    | must be ≥ C1 utility")

# Config 3f
if !isempty(feasible_c3)
    c3_spread = round(maximum(r.share for r in feasible_c3) - minimum(r.share for r in feasible_c3), digits=3)
    println("Config 3f  | $(lpad(round(c3_share, digits=3), 7))%  | $(round(c3_utility, sigdigits=8)) | ±$(c3_spread) pp    | $(length(feasible_c3))/$n_runs feasible")
end

# Config 4f
if !isempty(feasible_c4)
    c4_spread = round(maximum(r.share for r in feasible_c4) - minimum(r.share for r in feasible_c4), digits=3)
    println("Config 4f  | $(lpad(round(c4_share, digits=3), 7))%  | $(round(c4_utility, sigdigits=8)) | ±$(c4_spread) pp    | $(length(feasible_c4))/$n_runs feasible")
end

println("\nShifts:")
println("  1→2f (adaptation):    $(round(c2_share - c1_share, digits=3)) pp")
if !isempty(feasible_c3)
    println("  2f→3f (cost cap):     $(round(c3_share - c2_share, digits=3)) pp")
end
if !isempty(feasible_c4)
    println("  3f→4f (carbon budget): $(round(c4_share - c3_share, digits=3)) pp")
    println("  1→4f (total):          $(round(c4_share - c1_share, digits=3)) pp")
end

println("\nFinished at: $(now())")
