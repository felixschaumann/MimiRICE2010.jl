########################################################################################################
# HEAVY REFINEMENT: Cap=0.75%, Config 3 + Config 4
# Finer homotopy (7 steps), 10 long refinement passes (3600s), save PVs
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using CSV
using DataFrames
using Statistics

include("create_rice.jl")
include(joinpath(@__DIR__, "../src/", "optimisation_functions.jl"))
include(joinpath(@__DIR__, "../src/", "multistart_optimisation.jl"))

ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
n_regions = 2

results_dir = joinpath(@__DIR__, "../results/cap_sweep/refine_075")
mkpath(results_dir)

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

CONFIG1_SHARE_RICH = 31.5499
cap = 0.0075
target_cb = 1022.0

function run_and_extract(algo, n_opt, time_s, tol, backstop;
                         opt_ad, stock_ad, cbudget, cost_cap, start_pv)
    result = optimize_rice(
        algo, n_opt, time_s, tol, backstop;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=opt_ad, stock_ad=stock_ad, cbudget=cbudget, cost_cap=cost_cap,
        ext_starting_points=start_pv
    )
    model = result[7]
    utility = result[9]
    em = model[:emissions, :EIND]
    cca = model[:emissions, :CCA][end]
    max_cost = maximum(model[:neteconomy, :TOTAL_COST])
    feasible = check_feasibility(model, cbudget, cost_cap)
    cum_rich = sum(em[:, 1])
    cum_poor = sum(em[:, 2])
    total = cum_rich + cum_poor
    share_rich = cum_rich / total * 100
    return (utility=utility, cca=cca, max_cost=max_cost, feasible=feasible,
            cum_rich=cum_rich, cum_poor=cum_poor, share_rich=share_rich,
            policy_vector=result[1])
end

#------------------------------------------------------------------------------------------------------
# Phase 1: Config 3 — 3 passes × 1500s to get solid starting PV
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE 1: Config 3 @ cap=0.75% — 3 passes × 1500s")
println("="^80)

c3_results = []
global current_pv_c3 = nothing

for i in 1:3
    println("\n  Pass $i/3...")
    t0 = time()
    start = if i == 1
        nothing
    else
        (i % 2 == 0) ? current_pv_c3 : perturb_solution(current_pv_c3, 0.02, n_opt_periods, n_regions, true)
    end
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 1500, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=cap,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c3_results, r)
    feasible_c3 = filter(r -> r.feasible, c3_results)
    if !isempty(feasible_c3)
        global current_pv_c3 = sort(feasible_c3, by=r->-r.utility)[1].policy_vector
    elseif i == 1
        global current_pv_c3 = r.policy_vector
    end
    feas_str = r.feasible ? "✓" : "✗"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  CCA=$(round(r.cca, digits=1))  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c3 = filter(r -> r.feasible, c3_results)
c3_best = sort(feas_c3, by=r->-r.utility)[1]
c3_best_pv = c3_best.policy_vector
c3_best_cca = c3_best.cca
println("\n  Best Config 3: utility=$(round(c3_best.utility, sigdigits=8))  CCA=$(round(c3_best_cca, digits=1))")

# Save Config 3 PV
CSV.write(joinpath(results_dir, "config3_policy_vector.csv"), DataFrame(x=c3_best_pv))

#------------------------------------------------------------------------------------------------------
# Phase 2: Fine homotopy (7 steps from CCA → 1022)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE 2: Fine homotopy — 7 steps to CB=1022")
println("="^80)

# Create 7 evenly spaced steps from slack to target
gap = c3_best_cca - target_cb
steps_cbudget = [round(c3_best_cca + 10, digits=0)]  # start slack
for frac in [0.8, 0.6, 0.4, 0.2, 0.0]
    push!(steps_cbudget, round(target_cb + gap * frac, digits=0))
end
push!(steps_cbudget, target_cb)  # exact target

homotopy_steps = [(cost_cap=cap, cbudget=cb) for cb in steps_cbudget]
println("  Path: $(join(steps_cbudget, " → "))")

homotopy_results = homotopy_solve(
    homotopy_steps, c3_best_pv,
    :LN_AUGLAG, n_opt_periods, backstop_prices;
    time_per_step=900, first_step_time=1200, final_step_time=1800,
    first_step_starts=4, final_step_starts=5,
    tolerance=1e-15, ρ=ρ, η=η, remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=true
)

homotopy_pv = homotopy_results[end].policy_vector
homotopy_util = homotopy_results[end].utility
homotopy_feas = homotopy_results[end].feasible
println("\n  Homotopy endpoint: utility=$(round(homotopy_util, sigdigits=8))  feasible=$homotopy_feas")

# Save homotopy PV
CSV.write(joinpath(results_dir, "homotopy_policy_vector.csv"), DataFrame(x=homotopy_pv))

#------------------------------------------------------------------------------------------------------
# Phase 3: 10 long refinement passes (3600s each)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE 3: 10 refinement passes × 3600s from homotopy solution")
println("="^80)

c4_results = []
global current_pv_c4 = homotopy_pv
global best_utility_c4 = -Inf
global best_pv_c4 = homotopy_pv

for i in 1:10
    println("\n  Refinement $i/10...")
    t0 = time()

    # Strategy: alternate between direct refinement, small perturbation, and larger perturbation
    start = if i % 3 == 1
        perturb_solution(current_pv_c4, 0.01, n_opt_periods, n_regions, true)
    elseif i % 3 == 2
        current_pv_c4  # direct refinement
    else
        perturb_solution(current_pv_c4, 0.03, n_opt_periods, n_regions, true)
    end

    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 3600, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=target_cb, cost_cap=cap,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c4_results, r)

    # Update best feasible
    if r.feasible && r.utility > best_utility_c4
        global best_utility_c4 = r.utility
        global best_pv_c4 = r.policy_vector
        global current_pv_c4 = r.policy_vector
        # Save improved PV
        CSV.write(joinpath(results_dir, "best_config4_policy_vector.csv"), DataFrame(x=best_pv_c4))
    else
        # Still use best known for next iteration
        feasible_c4 = filter(r -> r.feasible, c4_results)
        if !isempty(feasible_c4)
            global current_pv_c4 = sort(feasible_c4, by=r->-r.utility)[1].policy_vector
        end
    end

    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)), cca=$(round(r.cca, sigdigits=6)))"
    best_str = (r.feasible && r.utility >= best_utility_c4) ? " ★ NEW BEST" : ""
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)$best_str")

    # Running summary
    feas_so_far = filter(r -> r.feasible, c4_results)
    if !isempty(feas_so_far)
        shares = [r.share_rich for r in feas_so_far]
        println("    Running: n=$(length(feas_so_far))  mean=$(round(mean(shares), digits=3))  std=$(round(std(shares), digits=4))  range=[$(round(minimum(shares), digits=3)), $(round(maximum(shares), digits=3))]")
    end
end

#------------------------------------------------------------------------------------------------------
# Final summary
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("FINAL RESULTS — Cap=0.75%, Config 4 (cap + CB 1022)")
println("="^80)

feas_c4 = filter(r -> r.feasible, c4_results)
c4_shares = [r.share_rich for r in feas_c4]
c4_utils = [r.utility for r in feas_c4]

println("\n  All feasible runs ($(length(feas_c4))/10):")
for (i, r) in enumerate(feas_c4)
    println("    $(i): utility=$(round(r.utility, sigdigits=8))  share_rich=$(round(r.share_rich, digits=3))%")
end

c4_mean = mean(c4_shares)
c4_std = std(c4_shares)
signal = c4_mean - CONFIG1_SHARE_RICH
snr = c4_std > 1e-10 ? abs(signal) / c4_std : Inf

println("\n  Summary:")
println("    Config 1 baseline: $(CONFIG1_SHARE_RICH)%")
println("    Config 4 mean:     $(round(c4_mean, digits=3)) ± $(round(c4_std, digits=4))%")
println("    Δ(4 vs 1):         $(round(signal, digits=3)) pp  SNR=$(round(snr, digits=2))")
println("    Direction:         $(signal > 0 ? "rich↑ (poor gets LESS)" : "rich↓ (poor gets MORE)")")

# Best solution analysis
best_idx = argmax(c4_utils)
best = feas_c4[best_idx]
println("\n  Best solution (utility=$(round(best.utility, sigdigits=8))):")
println("    share_rich = $(round(best.share_rich, digits=3))%")
println("    Δ vs Config 1: $(round(best.share_rich - CONFIG1_SHARE_RICH, digits=3)) pp")

# Save all results
results_df = DataFrame(
    run = 1:length(c4_results),
    utility = [r.utility for r in c4_results],
    share_rich = [r.share_rich for r in c4_results],
    feasible = [r.feasible for r in c4_results],
    cca = [r.cca for r in c4_results],
    max_cost = [r.max_cost for r in c4_results],
)
CSV.write(joinpath(results_dir, "config4_refinement_results.csv"), results_df)
println("\nSaved to: $(results_dir)")

println("\n" * "="^80)
println("DONE")
println("="^80)
