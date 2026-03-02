########################################################################################################
# FLOW-ONLY CONVERGENCE v2
# Fixes from v1: longer homotopy times, stricter cap (0.8%), more refinement time
# Config 2f: unconstrained (SBPLX), Config 3f: cap=0.8%, Config 4f: cap + CB 1022
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
n_runs = 5
cap = 0.008
target_cb = 1022.0

results_dir = joinpath(@__DIR__, "../results/flow_only_convergence")
mkpath(results_dir)

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

CONFIG1_SHARE_RICH = 31.5499

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
# Config 2f: Flow-only, unconstrained, SBPLX — 5 × 600s
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 2f: Flow-only adaptation, unconstrained — SBPLX — $n_runs × 600s")
println("="^80)

c2f_results = []
for i in 1:n_runs
    println("\n  Run $i/$n_runs...")
    t0 = time()
    start = nothing
    if i > 1
        prev_best = sort(c2f_results, by=r->-r.utility)[1]
        start = perturb_solution(prev_best.policy_vector, 0.03, n_opt_periods, n_regions, true)
    end
    r = run_and_extract(:LN_SBPLX, n_opt_periods, 600, 1e-12, backstop_prices;
                        opt_ad=true, stock_ad=false, cbudget=nothing, cost_cap=nothing,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c2f_results, r)
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  CCA=$(round(r.cca, digits=1))  ($(elapsed)s)")
end
c2f_shares = [r.share_rich for r in c2f_results]
println("\n  Config 2f: share_rich = $(round(mean(c2f_shares), digits=3)) ± $(round(length(c2f_shares) > 1 ? std(c2f_shares) : 0.0, digits=4))%")

#------------------------------------------------------------------------------------------------------
# Config 3f: Flow-only + cost cap 0.8% — AUGLAG — 5 iterative × 1200s (longer than v1)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 3f: Flow-only + cap=0.8% — AUGLAG — $n_runs iterative × 1200s")
println("="^80)

c3f_results = []
global current_pv_c3f = nothing

for i in 1:n_runs
    println("\n  Pass $i/$n_runs...")
    t0 = time()
    start = if i == 1
        nothing
    else
        (i % 2 == 0) ? current_pv_c3f : perturb_solution(current_pv_c3f, 0.02, n_opt_periods, n_regions, true)
    end
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 1200, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=false, cbudget=nothing, cost_cap=cap,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c3f_results, r)
    feasible = filter(r -> r.feasible, c3f_results)
    if !isempty(feasible)
        global current_pv_c3f = sort(feasible, by=r->-r.utility)[1].policy_vector
    elseif i == 1
        global current_pv_c3f = r.policy_vector
    end
    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  CCA=$(round(r.cca, digits=1))  Feasible=$feas_str  ($(elapsed)s)")
end
feas_c3f = filter(r -> r.feasible, c3f_results)
c3f_shares = [r.share_rich for r in feas_c3f]
if !isempty(c3f_shares)
    println("\n  Config 3f (feasible=$(length(feas_c3f))/$n_runs): share_rich = $(round(mean(c3f_shares), digits=3)) ± $(round(length(c3f_shares) > 1 ? std(c3f_shares) : 0.0, digits=4))%")
else
    println("\n  Config 3f: NO FEASIBLE RUNS — aborting")
    println("DONE (incomplete)")
    exit(1)
end

#------------------------------------------------------------------------------------------------------
# Config 4f: Flow-only + cap + CB 1022 — homotopy + 5 refinement × 2400s
# Key fix: much longer homotopy times so AUGLAG can calibrate the cost constraint
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 4f: Flow-only + cap=0.8% + CB=1022 — homotopy + $n_runs × 2400s")
println("="^80)

if !isempty(feas_c3f)
    c3f_best = sort(feas_c3f, by=r->-r.utility)[1]
    c3f_best_pv = c3f_best.policy_vector
    c3f_best_cca = c3f_best.cca
else
    c3f_best_pv = c3f_results[end].policy_vector
    c3f_best_cca = c3f_results[end].cca
end

println("  Config 3f best CCA: $(round(c3f_best_cca, digits=1))")

# Design homotopy — fine steps
gap = c3f_best_cca - target_cb
if gap > 5
    steps_cbudget = [round(c3f_best_cca + 10, digits=0)]
    for frac in [0.8, 0.6, 0.4, 0.2, 0.0]
        push!(steps_cbudget, round(target_cb + gap * frac, digits=0))
    end
    push!(steps_cbudget, target_cb)
elseif gap > 1
    mid = round((c3f_best_cca + target_cb) / 2, digits=0)
    steps_cbudget = [round(c3f_best_cca + 10, digits=0), mid, target_cb]
else
    steps_cbudget = [target_cb + 10, target_cb]
end

homotopy_steps = [(cost_cap=cap, cbudget=cb) for cb in steps_cbudget]
println("  Homotopy: $(join(steps_cbudget, " → "))")

# Key fix: generous time budgets for homotopy
homotopy_results = homotopy_solve(
    homotopy_steps, c3f_best_pv,
    :LN_AUGLAG, n_opt_periods, backstop_prices;
    time_per_step=900, first_step_time=1200, final_step_time=1800,
    first_step_starts=4, final_step_starts=5,
    tolerance=1e-15, ρ=ρ, η=η, remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=false
)

homotopy_pv = homotopy_results[end].policy_vector
homotopy_feas = homotopy_results[end].feasible
println("  Homotopy endpoint: utility=$(round(homotopy_results[end].utility, sigdigits=8))  feasible=$homotopy_feas")

# Refinement passes — 2400s each, only small perturbations
c4f_results = []
global current_pv_c4f = homotopy_pv

for i in 1:n_runs
    println("\n  Refinement $i/$n_runs...")
    t0 = time()
    # Only small perturbations (0.01) or direct — no large perturbations that kick out of basin
    start = (i % 2 == 1) ? perturb_solution(current_pv_c4f, 0.01, n_opt_periods, n_regions, true) : current_pv_c4f
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 2400, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=false, cbudget=target_cb, cost_cap=cap,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c4f_results, r)
    feasible = filter(r -> r.feasible, c4f_results)
    if !isempty(feasible)
        global current_pv_c4f = sort(feasible, by=r->-r.utility)[1].policy_vector
    end
    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)), cca=$(round(r.cca, sigdigits=6)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c4f = filter(r -> r.feasible, c4f_results)
c4f_shares = [r.share_rich for r in feas_c4f]

#------------------------------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("FLOW-ONLY CONVERGENCE v2 — cap=0.8%")
println("="^80)

println("\n  Config                    | n_feas | Mean ± Std        | Range")
println("  --------------------------|--------|-------------------|--------")

for (label, shares, n_total) in [
    ("1:  Mitigation only (ref)", [CONFIG1_SHARE_RICH], 1),
    ("2f: + Flow adapt (SBPLX)", c2f_shares, n_runs),
    ("3f: + Cost cap 0.8%", c3f_shares, n_runs),
    ("4f: + Cap + CB 1022", c4f_shares, n_runs),
]
    n = length(shares)
    m = mean(shares)
    s = n > 1 ? std(shares) : 0.0
    range_str = n > 1 ? "[$(round(minimum(shares), digits=3)), $(round(maximum(shares), digits=3))]" : "—"
    println("  $(rpad(label, 26))| $(n)/$(n_total)    | $(round(m, digits=3)) ± $(round(s, digits=4))%  | $range_str")
end

# Signal analysis
println("\n--- Pairwise signals ---")
c1_share = CONFIG1_SHARE_RICH
c2f_mean = mean(c2f_shares)
c3f_mean = mean(c3f_shares)
c4f_mean = !isempty(c4f_shares) ? mean(c4f_shares) : NaN

for (label, new_s, base_s) in [
    ("2f vs 1 (adaptation)", c2f_mean, c1_share),
    ("3f vs 1 (+ cap)", c3f_mean, c1_share),
    ("4f vs 1 (+ cap + CB)", c4f_mean, c1_share),
    ("3f vs 2f (cap effect)", c3f_mean, c2f_mean),
    ("4f vs 3f (CB effect)", c4f_mean, c3f_mean),
]
    signal = new_s - base_s
    rel = signal / base_s * 100
    dir = signal > 0 ? "rich↑" : "rich↓"
    println("  $(rpad(label, 28)) $(round(signal, digits=3)) pp  ($(round(rel, digits=2))% rel)  $dir")
end

# Save
all_results = DataFrame(
    config = vcat(fill("2f", length(c2f_results)), fill("3f", length(c3f_results)), fill("4f", length(c4f_results))),
    utility = vcat([r.utility for r in c2f_results], [r.utility for r in c3f_results], [r.utility for r in c4f_results]),
    share_rich = vcat([r.share_rich for r in c2f_results], [r.share_rich for r in c3f_results], [r.share_rich for r in c4f_results]),
    feasible = vcat([r.feasible for r in c2f_results], [r.feasible for r in c3f_results], [r.feasible for r in c4f_results]),
    cca = vcat([r.cca for r in c2f_results], [r.cca for r in c3f_results], [r.cca for r in c4f_results]),
    max_cost = vcat([r.max_cost for r in c2f_results], [r.max_cost for r in c3f_results], [r.max_cost for r in c4f_results]),
)
CSV.write(joinpath(results_dir, "flow_only_v2.csv"), all_results)
println("\nSaved to: $(joinpath(results_dir, "flow_only_v2.csv"))")

println("\n" * "="^80)
println("DONE")
println("="^80)
