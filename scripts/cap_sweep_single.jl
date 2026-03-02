########################################################################################################
# SINGLE-CAP SWEEP: Run Config 3 + Config 4 for one cost cap level
# Usage: julia --project=scripts scripts/cap_sweep_single.jl <cap_value>
# Example: julia --project=scripts scripts/cap_sweep_single.jl 0.0075
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

cap = parse(Float64, ARGS[1])
cap_pct = round(cap * 100, digits=2)
cap_tag = replace(string(cap_pct), "." => "")  # e.g. "075" for 0.75%

ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
n_regions = 2
n_runs = 5

results_dir = joinpath(@__DIR__, "../results/cap_sweep")
mkpath(results_dir)

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

CONFIG1_SHARE_RICH = 31.5499

#------------------------------------------------------------------------------------------------------
# Helper
#------------------------------------------------------------------------------------------------------

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
# Config 3: cap only (no carbon budget)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 3 @ cap=$(cap_pct)%: $n_runs runs × 1800s")
println("="^80)

c3_results = []
local current_pv_c3

for i in 1:n_runs
    println("\n  Pass $i/$n_runs...")
    t0 = time()

    if i == 1
        cached_path = joinpath(@__DIR__, "../results/convergence_test/3_stock_flow_ad_cap008/policy_vector.csv")
        if cap == 0.008 && isfile(cached_path)
            start = CSV.read(cached_path, DataFrame).x
            println("    Using cached Config 3 PV")
        else
            start = nothing
        end
    else
        start = (i % 2 == 0) ? current_pv_c3 : perturb_solution(current_pv_c3, 0.02, n_opt_periods, n_regions, true)
    end

    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 1800, 1e-15, backstop_prices;
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

    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  CCA=$(round(r.cca, digits=1))  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c3 = filter(r -> r.feasible, c3_results)
c3_shares = [r.share_rich for r in feas_c3]
println("\n  Config 3 summary (feasible=$(length(feas_c3))/$n_runs): share_rich = $(round(mean(c3_shares), digits=3)) ± $(round(std(c3_shares), digits=4))%")

#------------------------------------------------------------------------------------------------------
# Config 4: cap + CB 1022 (homotopy from Config 3 best)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 4 @ cap=$(cap_pct)%: homotopy + $n_runs refinement passes × 2400s")
println("="^80)

if !isempty(feas_c3)
    c3_best_pv = sort(feas_c3, by=r->-r.utility)[1].policy_vector
    c3_best_cca = sort(feas_c3, by=r->-r.utility)[1].cca
else
    c3_best_pv = c3_results[end].policy_vector
    c3_best_cca = c3_results[end].cca
end

target_cb = 1022.0
if c3_best_cca > target_cb + 5
    mid1 = round(c3_best_cca + 15, digits=0)
    mid2 = round(c3_best_cca + 5, digits=0)
    mid3 = round(c3_best_cca, digits=0)
    mid4 = round((c3_best_cca + target_cb) / 2, digits=0)
    homotopy_steps = [
        (cost_cap=cap, cbudget=mid1),
        (cost_cap=cap, cbudget=mid2),
        (cost_cap=cap, cbudget=mid3),
        (cost_cap=cap, cbudget=mid4),
        (cost_cap=cap, cbudget=target_cb),
    ]
    println("  Homotopy path: $(mid1) → $(mid2) → $(mid3) → $(mid4) → $(target_cb)")
elseif c3_best_cca > target_cb + 1
    mid = round((c3_best_cca + target_cb) / 2, digits=0)
    homotopy_steps = [
        (cost_cap=cap, cbudget=c3_best_cca + 10),
        (cost_cap=cap, cbudget=mid),
        (cost_cap=cap, cbudget=target_cb),
    ]
    println("  Short homotopy: $(c3_best_cca + 10) → $(mid) → $(target_cb)")
else
    homotopy_steps = [
        (cost_cap=cap, cbudget=target_cb + 10),
        (cost_cap=cap, cbudget=target_cb),
    ]
    println("  Direct: $(target_cb + 10) → $(target_cb)")
end

homotopy_results = homotopy_solve(
    homotopy_steps, c3_best_pv,
    :LN_AUGLAG, n_opt_periods, backstop_prices;
    time_per_step=600, first_step_time=900, final_step_time=1200,
    first_step_starts=3, final_step_starts=3,
    tolerance=1e-15, ρ=ρ, η=η, remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=true
)

final_homotopy = homotopy_results[end]
homotopy_pv = final_homotopy.policy_vector

c4_results = []
global current_pv_c4 = homotopy_pv

for i in 1:n_runs
    println("\n  Refinement $i/$n_runs...")
    t0 = time()
    start = (i % 2 == 0) ? current_pv_c4 : perturb_solution(current_pv_c4, 0.02, n_opt_periods, n_regions, true)
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 2400, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=target_cb, cost_cap=cap,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c4_results, r)

    feasible_c4 = filter(r -> r.feasible, c4_results)
    if !isempty(feasible_c4)
        global current_pv_c4 = sort(feasible_c4, by=r->-r.utility)[1].policy_vector
    end

    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)), cca=$(round(r.cca, sigdigits=6)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c4 = filter(r -> r.feasible, c4_results)
c4_shares = [r.share_rich for r in feas_c4]

#------------------------------------------------------------------------------------------------------
# Summary for this cap
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("RESULTS @ cap=$(cap_pct)%")
println("="^80)

c3_mean = isempty(c3_shares) ? NaN : mean(c3_shares)
c3_std_val = length(c3_shares) > 1 ? std(c3_shares) : 0.0
c4_mean = isempty(c4_shares) ? NaN : mean(c4_shares)
c4_std_val = length(c4_shares) > 1 ? std(c4_shares) : 0.0

println("  Config 1 baseline: $(CONFIG1_SHARE_RICH)%")
println("  Config 3 (cap only):   $(round(c3_mean, digits=3)) ± $(round(c3_std_val, digits=4))%  [$(length(feas_c3))/$n_runs feasible]")
println("  Config 4 (cap + CB):   $(round(c4_mean, digits=3)) ± $(round(c4_std_val, digits=4))%  [$(length(feas_c4))/$n_runs feasible]")

signal = c4_mean - CONFIG1_SHARE_RICH
snr = c4_std_val > 1e-10 ? abs(signal) / c4_std_val : Inf
dir = signal > 0 ? "rich↑" : "rich↓"
println("  Δ(4 vs 1): $(round(signal, digits=3)) pp ($dir)  SNR=$(round(snr, digits=2))")

rel_rich = signal / CONFIG1_SHARE_RICH * 100
rel_poor = -signal / (100 - CONFIG1_SHARE_RICH) * 100
println("  Relative: Rich $(rel_rich >= 0 ? "+" : "")$(round(rel_rich, digits=2))%  Poor $(rel_poor >= 0 ? "+" : "")$(round(rel_poor, digits=2))%")

# Save per-cap results
cap_df = DataFrame(
    config = vcat(fill(3, length(c3_results)), fill(4, length(c4_results))),
    cap = cap,
    utility = vcat([r.utility for r in c3_results], [r.utility for r in c4_results]),
    share_rich = vcat([r.share_rich for r in c3_results], [r.share_rich for r in c4_results]),
    feasible = vcat([r.feasible for r in c3_results], [r.feasible for r in c4_results]),
    cca = vcat([r.cca for r in c3_results], [r.cca for r in c4_results]),
    max_cost = vcat([r.max_cost for r in c3_results], [r.max_cost for r in c4_results]),
)
CSV.write(joinpath(results_dir, "cap_$(cap_tag)_results.csv"), cap_df)
println("\nSaved to: $(joinpath(results_dir, "cap_$(cap_tag)_results.csv"))")
println("\n" * "="^80)
println("DONE — cap=$(cap_pct)%")
println("="^80)
