########################################################################################################
# STRESS TEST: Config 4 (cost_cap=0.008 + cbudget=1022 GtC) global optimality verification
#
# Strategy: Run many diverse starting points through AUGLAG optimization with dual constraints.
# If the homotopy solution is a global optimum, all feasible solutions should converge to
# approximately the same utility and emissions allocation.
#
# Starting point strategies:
#   1-3: Large perturbations of homotopy solution (10%, 20%, 30%)
#   4:   Swapped regions (rich↔poor mitigation and adaptation)
#   5:   High mitigation / low adaptation
#   6:   Low mitigation / high adaptation
#   7:   Aggressive early action
#   8:   Delayed action
#   9:   Equal regions (same profile for both)
#  10:   Config 3 solution directly (let AUGLAG find carbon budget feasibility)
#  11-12: LHS ramp starts
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using CSV
using DataFrames
using Dates
using Statistics

include("create_rice.jl")
include(joinpath(@__DIR__, "../src/", "optimisation_functions.jl"))
include(joinpath(@__DIR__, "../src/", "multistart_optimisation.jl"))

#------------------------------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------------------------------

ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
n_regions = 2
cost_cap = 0.008
cbudget = 1022.0
time_per_start = 900  # 15 minutes per start — generous

# Output directory
results_dir = joinpath(@__DIR__, "../results/stresstest")
mkpath(results_dir)

# Get backstop prices
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

#------------------------------------------------------------------------------------------------------
# Load homotopy solution (reference)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("LOADING REFERENCE SOLUTIONS")
println("="^80)

homotopy_pv = CSV.read(joinpath(@__DIR__, "../results/homotopy/config4_policy_vector.csv"), DataFrame).x
println("Loaded homotopy Config 4 policy vector: $(length(homotopy_pv)) elements")

# Also load Config 3 solution for comparison
config3_pv = CSV.read(joinpath(@__DIR__, "../results/convergence_test/3_stock_flow_ad_cap008/policy_vector.csv"), DataFrame).x
println("Loaded Config 3 policy vector: $(length(config3_pv)) elements")

#------------------------------------------------------------------------------------------------------
# Generate diverse starting points
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("GENERATING STARTING POINTS")
println("="^80)

starting_points = Tuple{Vector{Float64}, String}[]

# --- 1-3: Large perturbations of homotopy solution ---
for (i, mag) in enumerate([0.10, 0.20, 0.30])
    sp = perturb_solution(homotopy_pv, mag, n_opt_periods, n_regions, true)
    push!(starting_points, (sp, "perturb-$(Int(mag*100))pct"))
end

# --- 4: Swapped regions (rich↔poor) ---
# Swap mitigation for rich and poor, and similarly for adaptation
function swap_regions(x, n_opt_periods, n_regions)
    swapped = copy(x)
    for block in 0:2  # mitigation, flow adapt, stock adapt
        offset = block * n_opt_periods * n_regions
        r1 = x[(offset+1):(offset+n_opt_periods)]
        r2 = x[(offset+n_opt_periods+1):(offset+2*n_opt_periods)]
        swapped[(offset+1):(offset+n_opt_periods)] = r2
        swapped[(offset+n_opt_periods+1):(offset+2*n_opt_periods)] = r1
    end
    return swapped
end
push!(starting_points, (swap_regions(homotopy_pv, n_opt_periods, n_regions), "swapped-regions"))

# --- 5: High mitigation / low adaptation ---
sp5 = vcat(
    ones(n_opt_periods * n_regions) .* 0.95,      # high mitigation
    ones(n_opt_periods * n_regions) .* 0.05,       # low flow adaptation
    ones(n_opt_periods * n_regions) .* 0.05        # low stock adaptation
)
push!(starting_points, (sp5, "high-mit-low-ad"))

# --- 6: Low mitigation / high adaptation ---
sp6 = vcat(
    ones(n_opt_periods * n_regions) .* 0.3,        # low mitigation
    ones(n_opt_periods * n_regions) .* 0.5,        # high flow adaptation
    ones(n_opt_periods * n_regions) .* 0.5         # high stock adaptation
)
push!(starting_points, (sp6, "low-mit-high-ad"))

# --- 7: Aggressive early action ---
# MIU starts at 0.6, rises quickly to 1.0 by period 10
sp7_mit = zeros(n_opt_periods * n_regions)
for r in 1:n_regions
    for t in 1:n_opt_periods
        offset = (r-1) * n_opt_periods + t
        sp7_mit[offset] = min(1.0, 0.6 + 0.04 * t)
    end
end
sp7 = vcat(sp7_mit, ones(n_opt_periods * n_regions) .* 0.2, ones(n_opt_periods * n_regions) .* 0.2)
push!(starting_points, (sp7, "aggressive-early"))

# --- 8: Delayed action ---
# MIU starts at 0.1, rises slowly
sp8_mit = zeros(n_opt_periods * n_regions)
for r in 1:n_regions
    for t in 1:n_opt_periods
        offset = (r-1) * n_opt_periods + t
        sp8_mit[offset] = min(1.0, 0.1 + 0.03 * t)
    end
end
sp8 = vcat(sp8_mit, ones(n_opt_periods * n_regions) .* 0.15, ones(n_opt_periods * n_regions) .* 0.15)
push!(starting_points, (sp8, "delayed-action"))

# --- 9: Equal regions ---
# Both regions get the average of the homotopy rich/poor profiles
avg_profile = copy(homotopy_pv)
for block in 0:2
    offset = block * n_opt_periods * n_regions
    r1 = homotopy_pv[(offset+1):(offset+n_opt_periods)]
    r2 = homotopy_pv[(offset+n_opt_periods+1):(offset+2*n_opt_periods)]
    avg = (r1 + r2) / 2
    avg_profile[(offset+1):(offset+n_opt_periods)] = avg
    avg_profile[(offset+n_opt_periods+1):(offset+2*n_opt_periods)] = avg
end
push!(starting_points, (avg_profile, "equal-regions"))

# --- 10: Config 3 solution (let AUGLAG find carbon budget feasibility) ---
push!(starting_points, (config3_pv, "config3-direct"))

# --- 11-12: LHS ramp starts ---
ramp_starts = generate_ramp_starting_points(2, n_opt_periods, n_regions, true)
for (i, sp) in enumerate(ramp_starts)
    push!(starting_points, (sp, "ramp-$i"))
end

println("Generated $(length(starting_points)) starting points:")
for (i, (sp, label)) in enumerate(starting_points)
    println("  $i. $label")
end

#------------------------------------------------------------------------------------------------------
# Run stress test
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("RUNNING STRESS TEST ($(length(starting_points)) starts × $(time_per_start)s each)")
println("Estimated time: ~$(round(length(starting_points) * time_per_start / 60, digits=0)) minutes")
println("="^80)

stress_results = []

for (i, (sp, label)) in enumerate(starting_points)
    println("\n" * "-"^60)
    println("Start $i/$(length(starting_points)): $label")
    println("-"^60)

    t_start = time()

    try
        result = optimize_rice(
            :LN_AUGLAG, n_opt_periods, time_per_start, 1e-15, backstop_prices;
            run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
            opt_ad=true, stock_ad=true, cbudget=cbudget, cost_cap=cost_cap,
            ext_starting_points=sp
        )

        model = result[7]
        utility = result[9]
        cca_val = model[:emissions, :CCA][end]
        max_cost_val = maximum(model[:neteconomy, :TOTAL_COST])
        feasible = check_feasibility(model, cbudget, cost_cap)

        # Emissions by region (sum of EIND * 10 for GtCO2, but raw for shares)
        emissions = model[:emissions, :EIND]
        cum_rich = sum(emissions[:, 1])
        cum_poor = sum(emissions[:, 2])
        cum_total = cum_rich + cum_poor
        share_rich = cum_rich / cum_total * 100

        # MIU smoothness
        miu = model[:emissions, :MIU]
        smoothness = maximum(abs.(diff(miu[1:min(20, size(miu,1)), :], dims=1)))

        elapsed = round(time() - t_start, digits=1)

        push!(stress_results, (
            label=label,
            utility=utility,
            cca=cca_val,
            max_cost=max_cost_val,
            feasible=feasible,
            cum_rich=cum_rich,
            cum_poor=cum_poor,
            share_rich=share_rich,
            smoothness=smoothness,
            elapsed=elapsed,
            policy_vector=result[1]
        ))

        println("  Utility:    $(round(utility, sigdigits=8))")
        println("  CCA:        $(round(cca_val, sigdigits=6)) GtC (budget=$cbudget)")
        println("  Max cost:   $(round(max_cost_val, sigdigits=4)) (cap=$cost_cap)")
        println("  Feasible:   $feasible")
        println("  Share rich: $(round(share_rich, digits=1))%")
        println("  Smoothness: $(round(smoothness, sigdigits=3))")
        println("  Time:       $(elapsed)s")
    catch e
        println("  FAILED: $e")
        push!(stress_results, (
            label=label, utility=NaN, cca=NaN, max_cost=NaN, feasible=false,
            cum_rich=NaN, cum_poor=NaN, share_rich=NaN, smoothness=NaN,
            elapsed=round(time() - t_start, digits=1), policy_vector=Float64[]
        ))
    end
end

#------------------------------------------------------------------------------------------------------
# Also evaluate the homotopy solution (no optimization, just model run)
#------------------------------------------------------------------------------------------------------

println("\n" * "-"^60)
println("Reference: Homotopy solution (no re-optimization)")
println("-"^60)

ref_result = optimize_rice(
    :LN_AUGLAG, n_opt_periods, 60, 1e-10, backstop_prices;
    run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=true, cbudget=cbudget, cost_cap=cost_cap,
    ext_starting_points=homotopy_pv
)
ref_model = ref_result[7]
ref_utility = ref_result[9]
ref_cca = ref_model[:emissions, :CCA][end]
ref_max_cost = maximum(ref_model[:neteconomy, :TOTAL_COST])
ref_feasible = check_feasibility(ref_model, cbudget, cost_cap)
ref_emissions = ref_model[:emissions, :EIND]
ref_cum_rich = sum(ref_emissions[:, 1])
ref_cum_poor = sum(ref_emissions[:, 2])
ref_share_rich = ref_cum_rich / (ref_cum_rich + ref_cum_poor) * 100

println("  Utility:    $(round(ref_utility, sigdigits=8))")
println("  CCA:        $(round(ref_cca, sigdigits=6)) GtC")
println("  Max cost:   $(round(ref_max_cost, sigdigits=4))")
println("  Feasible:   $ref_feasible")
println("  Share rich: $(round(ref_share_rich, digits=1))%")

#------------------------------------------------------------------------------------------------------
# Summary table
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("STRESS TEST SUMMARY")
println("="^80)

# Sort by utility (descending), feasible first
feasible_results = filter(r -> r.feasible, stress_results)
infeasible_results = filter(r -> !r.feasible, stress_results)

println("\n--- FEASIBLE SOLUTIONS ($(length(feasible_results)) / $(length(stress_results))) ---")
header = rpad("Label", 20) * rpad("Utility", 16) * rpad("CCA", 10) * rpad("MaxCost", 10) * rpad("Share_R", 10) * rpad("Smooth", 8) * rpad("Time", 8)
println(header)
println("-"^length(header))

# Reference first
println(rpad("HOMOTOPY (ref)", 20) * rpad(string(round(ref_utility, sigdigits=8)), 16) *
        rpad(string(round(ref_cca, sigdigits=6)), 10) * rpad(string(round(ref_max_cost, sigdigits=4)), 10) *
        rpad(string(round(ref_share_rich, digits=1)) * "%", 10) * rpad("—", 8) * rpad("—", 8))

for r in sort(collect(feasible_results), by=x -> -x.utility)
    delta = r.utility - ref_utility
    delta_str = delta >= 0 ? "+$(round(delta, sigdigits=3))" : "$(round(delta, sigdigits=3))"
    println(rpad(r.label, 20) * rpad("$(round(r.utility, sigdigits=8)) ($delta_str)", 16) *
            rpad(string(round(r.cca, sigdigits=6)), 10) * rpad(string(round(r.max_cost, sigdigits=4)), 10) *
            rpad(string(round(r.share_rich, digits=1)) * "%", 10) * rpad(string(round(r.smoothness, sigdigits=3)), 8) *
            rpad(string(r.elapsed), 8))
end

if !isempty(infeasible_results)
    println("\n--- INFEASIBLE SOLUTIONS ($(length(infeasible_results))) ---")
    for r in sort(collect(infeasible_results), by=x -> -x.utility)
        println(rpad(r.label, 20) * rpad(string(round(r.utility, sigdigits=8)), 16) *
                rpad(string(round(r.cca, sigdigits=6)), 10) * rpad(string(round(r.max_cost, sigdigits=4)), 10) *
                rpad(string(round(r.share_rich, digits=1)) * "%", 10) * rpad(string(round(r.smoothness, sigdigits=3)), 8) *
                rpad(string(r.elapsed), 8))
    end
end

#------------------------------------------------------------------------------------------------------
# Emissions share change analysis (all configs)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("EMISSIONS SHARE ANALYSIS (all configurations)")
println("="^80)

# Load emissions for all configs
config1_em = CSV.read(joinpath(@__DIR__, "../results/convergence_test/1_mitigation_only/Emissions_single.csv"), DataFrame)
config2_em = CSV.read(joinpath(@__DIR__, "../results/convergence_test/2_stock_flow_ad_no_constr/Emissions_single.csv"), DataFrame)
config3_em = CSV.read(joinpath(@__DIR__, "../results/convergence_test/3_stock_flow_ad_cap008/Emissions_refined.csv"), DataFrame)

# Config 4: use homotopy result (from model, already computed above)
# Also compute from stress test best feasible for comparison

# Helper function
function compute_shares(em_df)
    cum_rich = sum(em_df[:, 1])
    cum_poor = sum(em_df[:, 2])
    total = cum_rich + cum_poor
    return (cum_rich=cum_rich, cum_poor=cum_poor, total=total,
            share_rich=cum_rich/total*100, share_poor=cum_poor/total*100)
end

c1 = compute_shares(config1_em)
c2 = compute_shares(config2_em)
c3 = compute_shares(config3_em)

# Config 4: from reference evaluation
c4 = (cum_rich=ref_cum_rich, cum_poor=ref_cum_poor, total=ref_cum_rich+ref_cum_poor,
      share_rich=ref_share_rich, share_poor=100-ref_share_rich)

# Compute relative changes (as in analyse_ad_results.jl)
# Change = (scenario_share - baseline_share) / baseline_share × 100
function share_change(scenario, baseline)
    change_rich = (scenario.share_rich - baseline.share_rich) / baseline.share_rich * 100
    change_poor = (scenario.share_poor - baseline.share_poor) / baseline.share_poor * 100
    return (change_rich=change_rich, change_poor=change_poor)
end

# NOTE: Emissions units are EIND (model raw), cumulated across all 60 periods.
# The *10 conversion to GtCO2 cancels in the share calculation.
# CCA as reported by the model = cum_total * 10 (to convert to GtC).

println("\n--- Cumulative Emissions (raw EIND, summed over all periods) ---")
println(rpad("Config", 40) * rpad("Rich", 12) * rpad("Poor", 12) * rpad("Total", 12) * rpad("Rich%", 10) * rpad("Poor%", 10))
println("-"^96)

for (label, c) in [
    ("1: Mitigation only", c1),
    ("2: Stock+flow adapt, unconstrained", c2),
    ("3: Stock+flow + cost cap 0.8%", c3),
    ("4: Stock+flow + cap + CB 1022 (homotopy)", c4)
]
    println(rpad(label, 40) *
            rpad(string(round(c.cum_rich, digits=2)), 12) *
            rpad(string(round(c.cum_poor, digits=2)), 12) *
            rpad(string(round(c.total, digits=2)), 12) *
            rpad(string(round(c.share_rich, digits=2)) * "%", 10) *
            rpad(string(round(c.share_poor, digits=2)) * "%", 10))
end

println("\n--- Relative Change in Cumulative Emissions Share vs Baseline (Config 1) ---")
println("    (as computed in analyse_ad_results.jl)")
println(rpad("Config", 50) * rpad("Rich Δ%", 15) * rpad("Poor Δ%", 15))
println("-"^80)

for (label, c) in [
    ("2: Unconstrained adapt → Baseline", c2),
    ("3: + Cost cap 0.8% → Baseline", c3),
    ("4: + Cap + CB 1022 (homotopy) → Baseline", c4)
]
    sc = share_change(c, c1)
    rich_str = (sc.change_rich >= 0 ? "+" : "") * string(round(sc.change_rich, digits=2)) * "%"
    poor_str = (sc.change_poor >= 0 ? "+" : "") * string(round(sc.change_poor, digits=2)) * "%"
    println(rpad(label, 50) * rpad(rich_str, 15) * rpad(poor_str, 15))
end

# Also show Config 3→2 and Config 4→3 changes
println("\n--- Changes between consecutive configs ---")
sc32 = share_change(c3, c2)
sc43 = share_change(c4, c3)
sc42 = share_change(c4, c2)
println(rpad("3 vs 2 (effect of cost cap)", 50) * rpad((sc32.change_rich >= 0 ? "+" : "") * string(round(sc32.change_rich, digits=2)) * "%", 15) * rpad((sc32.change_poor >= 0 ? "+" : "") * string(round(sc32.change_poor, digits=2)) * "%", 15))
println(rpad("4 vs 3 (effect of carbon budget)", 50) * rpad((sc43.change_rich >= 0 ? "+" : "") * string(round(sc43.change_rich, digits=2)) * "%", 15) * rpad((sc43.change_poor >= 0 ? "+" : "") * string(round(sc43.change_poor, digits=2)) * "%", 15))
println(rpad("4 vs 2 (effect of both constraints)", 50) * rpad((sc42.change_rich >= 0 ? "+" : "") * string(round(sc42.change_rich, digits=2)) * "%", 15) * rpad((sc42.change_poor >= 0 ? "+" : "") * string(round(sc42.change_poor, digits=2)) * "%", 15))

# If stress test found any better feasible solution, compute its shares too
if !isempty(feasible_results)
    best_stress = sort(collect(feasible_results), by=x -> -x.utility)[1]
    if best_stress.utility > ref_utility + 0.001
        println("\n--- WARNING: Stress test found BETTER feasible solution! ---")
        c4_stress = (cum_rich=best_stress.cum_rich, cum_poor=best_stress.cum_poor,
                     total=best_stress.cum_rich+best_stress.cum_poor,
                     share_rich=best_stress.share_rich, share_poor=100-best_stress.share_rich)
        sc_stress = share_change(c4_stress, c1)
        println("  Label: $(best_stress.label)")
        println("  Utility: $(round(best_stress.utility, sigdigits=8)) (delta=$(round(best_stress.utility - ref_utility, sigdigits=4)))")
        println("  Rich share change: $(round(sc_stress.change_rich, digits=2))%")
        println("  Poor share change: $(round(sc_stress.change_poor, digits=2))%")
    else
        println("\n✓ No stress test start found a feasible solution better than homotopy.")
        println("  Best stress: $(best_stress.label) at $(round(best_stress.utility, sigdigits=8)) (delta=$(round(best_stress.utility - ref_utility, sigdigits=4)))")
    end
end

# Save stress test results
stress_df = DataFrame(
    label = [r.label for r in stress_results],
    utility = [r.utility for r in stress_results],
    cca = [r.cca for r in stress_results],
    max_cost = [r.max_cost for r in stress_results],
    feasible = [r.feasible for r in stress_results],
    cum_rich = [r.cum_rich for r in stress_results],
    cum_poor = [r.cum_poor for r in stress_results],
    share_rich = [r.share_rich for r in stress_results],
    smoothness = [r.smoothness for r in stress_results],
    elapsed = [r.elapsed for r in stress_results]
)
CSV.write(joinpath(results_dir, "stresstest_results.csv"), stress_df)
println("\nStress test results saved to: $(joinpath(results_dir, "stresstest_results.csv"))")

println("\n" * "="^80)
println("STRESS TEST COMPLETE")
println("="^80)
