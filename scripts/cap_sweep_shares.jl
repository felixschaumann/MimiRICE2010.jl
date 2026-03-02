########################################################################################################
# COST CAP SWEEP: How does tightening the cost cap affect emissions shares?
#
# For each cost cap level, run Config 3 (cap only) and Config 4 (cap + CB 1022)
# with multiple refinement passes. Compare all to Config 1 (31.55%, zero noise).
#
# Also run Config 2 with SBPLX (no constraints, no AUGLAG overhead).
#
# Cost caps: 0.008 (current), 0.006, 0.005, 0.004
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

results_dir = joinpath(@__DIR__, "../results/cap_sweep")
mkpath(results_dir)

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

# Config 1 baseline (rock solid from previous runs)
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
# Config 2: Unconstrained adaptation with SBPLX (no AUGLAG overhead)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 2: Unconstrained adaptation — SBPLX — $n_runs runs × 1200s")
println("="^80)

c2_results = []
for i in 1:n_runs
    println("\n  Run $i/$n_runs...")
    t0 = time()
    start = nothing
    if i > 1
        prev_best = sort(c2_results, by=r->-r.utility)[1]
        start = perturb_solution(prev_best.policy_vector, 0.03, n_opt_periods, n_regions, true)
    end
    r = run_and_extract(:LN_SBPLX, n_opt_periods, 1200, 1e-12, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=nothing,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c2_results, r)
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  ($(elapsed)s)")
end
c2_mean = mean([r.share_rich for r in c2_results])
c2_std = std([r.share_rich for r in c2_results])
println("\n  Config 2 (SBPLX): share_rich = $(round(c2_mean, digits=3)) ± $(round(c2_std, digits=4))%")

#------------------------------------------------------------------------------------------------------
# Cost cap sweep: for each cap, run Config 3 and Config 4
#------------------------------------------------------------------------------------------------------

cost_caps = [0.008, 0.0075, 0.007]
all_sweep_results = []

for cap in cost_caps
    cap_pct = round(cap * 100, digits=1)

    #--- Config 3: cap only (no carbon budget) ---
    println("\n" * "="^80)
    println("CONFIG 3 @ cap=$(cap_pct)%: $n_runs runs × 1800s")
    println("="^80)

    c3_results = []
    local current_pv_c3  # explicit local to avoid scoping issues

    # First run from default or cached solution
    for i in 1:n_runs
        println("\n  Pass $i/$n_runs...")
        t0 = time()

        if i == 1
            # Try to load cached PV for this cap, or use default
            cached_path = joinpath(@__DIR__, "../results/convergence_test/3_stock_flow_ad_cap008/policy_vector.csv")
            if cap == 0.008 && isfile(cached_path)
                start = CSV.read(cached_path, DataFrame).x
                println("    Using cached Config 3 PV")
            else
                start = nothing  # default
            end
        else
            # Alternate: direct refinement vs perturbed
            start = (i % 2 == 0) ? current_pv_c3 : perturb_solution(current_pv_c3, 0.02, n_opt_periods, n_regions, true)
        end

        r = run_and_extract(:LN_AUGLAG, n_opt_periods, 1800, 1e-15, backstop_prices;
                            opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=cap,
                            start_pv=start)
        elapsed = round(time() - t0, digits=1)
        push!(c3_results, r)

        # Track best feasible
        feasible_c3 = filter(r -> r.feasible, c3_results)
        if !isempty(feasible_c3)
            current_pv_c3 = sort(feasible_c3, by=r->-r.utility)[1].policy_vector
        elseif i == 1
            current_pv_c3 = r.policy_vector  # use even if infeasible as starting point
        end

        feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)))"
        println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  CCA=$(round(r.cca, digits=1))  Feasible=$feas_str  ($(elapsed)s)")
    end

    feas_c3 = filter(r -> r.feasible, c3_results)
    c3_shares = [r.share_rich for r in feas_c3]

    #--- Config 4: cap + CB 1022 (homotopy from Config 3 best) ---
    println("\n" * "="^80)
    println("CONFIG 4 @ cap=$(cap_pct)%: homotopy + $n_runs refinement passes × 2400s")
    println("="^80)

    # Get best Config 3 solution for this cap as homotopy starting point
    if !isempty(feas_c3)
        c3_best_pv = sort(feas_c3, by=r->-r.utility)[1].policy_vector
        c3_best_cca = sort(feas_c3, by=r->-r.utility)[1].cca
    else
        c3_best_pv = c3_results[end].policy_vector
        c3_best_cca = c3_results[end].cca
    end

    # Design homotopy steps based on Config 3's CCA
    target_cb = 1022.0
    if c3_best_cca > target_cb + 5
        # Need homotopy: start from slack, tighten
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
        # Short homotopy
        mid = round((c3_best_cca + target_cb) / 2, digits=0)
        homotopy_steps = [
            (cost_cap=cap, cbudget=c3_best_cca + 10),
            (cost_cap=cap, cbudget=mid),
            (cost_cap=cap, cbudget=target_cb),
        ]
        println("  Short homotopy: $(c3_best_cca + 10) → $(mid) → $(target_cb)")
    else
        # CCA already near target, direct
        homotopy_steps = [
            (cost_cap=cap, cbudget=target_cb + 10),
            (cost_cap=cap, cbudget=target_cb),
        ]
        println("  Direct: $(target_cb + 10) → $(target_cb)")
    end

    # Run homotopy
    homotopy_results = homotopy_solve(
        homotopy_steps, c3_best_pv,
        :LN_AUGLAG, n_opt_periods, backstop_prices;
        time_per_step=600, first_step_time=900, final_step_time=1200,
        first_step_starts=3, final_step_starts=3,
        tolerance=1e-15, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=true
    )

    # Get homotopy final solution
    final_homotopy = homotopy_results[end]
    homotopy_pv = final_homotopy.policy_vector

    # Now run refinement passes from homotopy solution
    c4_results = []
    local current_pv_c4 = homotopy_pv

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
            current_pv_c4 = sort(feasible_c4, by=r->-r.utility)[1].policy_vector
        end

        feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)), cca=$(round(r.cca, sigdigits=6)))"
        println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)")
    end

    feas_c4 = filter(r -> r.feasible, c4_results)
    c4_shares = [r.share_rich for r in feas_c4]

    # Store results for this cap
    push!(all_sweep_results, (
        cap=cap,
        c3_shares=c3_shares,
        c3_mean=isempty(c3_shares) ? NaN : mean(c3_shares),
        c3_std=length(c3_shares) > 1 ? std(c3_shares) : 0.0,
        c3_n_feasible=length(feas_c3),
        c4_shares=c4_shares,
        c4_mean=isempty(c4_shares) ? NaN : mean(c4_shares),
        c4_std=length(c4_shares) > 1 ? std(c4_shares) : 0.0,
        c4_n_feasible=length(feas_c4),
        c3_cca=isempty(feas_c3) ? NaN : sort(feas_c3, by=r->-r.utility)[1].cca,
        c4_cca=isempty(feas_c4) ? NaN : sort(feas_c4, by=r->-r.utility)[1].cca,
    ))
end

#------------------------------------------------------------------------------------------------------
# Final comparison table
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("COST CAP SWEEP — FINAL RESULTS")
println("="^80)

println("\nConfig 1 baseline: share_rich = $(CONFIG1_SHARE_RICH)% (zero noise)")
println("Config 2 (SBPLX):  share_rich = $(round(c2_mean, digits=3)) ± $(round(c2_std, digits=4))%")

println("\n" * "-"^100)
println(rpad("Cap", 8) * rpad("Config 3", 35) * "| " * rpad("Config 4 (+ CB 1022)", 35) * "| " * "Δ(4 vs 1)")
println(rpad("", 8) * rpad("mean ± std  [n_feas]  CCA", 35) * "| " * rpad("mean ± std  [n_feas]  CCA", 35) * "| " * "signal  SNR")
println("-"^100)

for s in all_sweep_results
    cap_str = "$(round(s.cap*100, digits=1))%"
    c3_str = "$(round(s.c3_mean, digits=2)) ± $(round(s.c3_std, digits=3))  [$(s.c3_n_feasible)/5]  $(round(s.c3_cca, digits=0))"
    c4_str = "$(round(s.c4_mean, digits=2)) ± $(round(s.c4_std, digits=3))  [$(s.c4_n_feasible)/5]  $(round(s.c4_cca, digits=0))"

    # Signal: Config 4 vs Config 1
    signal_41 = s.c4_mean - CONFIG1_SHARE_RICH
    snr_41 = s.c4_std > 1e-10 ? abs(signal_41) / s.c4_std : Inf
    dir = signal_41 > 0 ? "rich↑" : "rich↓"
    signal_str = "$(round(signal_41, digits=3))pp  $(round(snr_41, digits=2)) ($dir)"

    println(rpad(cap_str, 8) * rpad(c3_str, 35) * "| " * rpad(c4_str, 35) * "| " * signal_str)
end

println("\n--- Relative change in rich share (Config 4 mean vs Config 1) ---")
for s in all_sweep_results
    cap_str = "$(round(s.cap*100, digits=1))%"
    signal = s.c4_mean - CONFIG1_SHARE_RICH
    rel_rich = signal / CONFIG1_SHARE_RICH * 100
    rel_poor = -signal / (100 - CONFIG1_SHARE_RICH) * 100
    snr = s.c4_std > 1e-10 ? abs(signal) / s.c4_std : Inf
    println("  Cap $(cap_str):  Rich $(rel_rich >= 0 ? "+" : "")$(round(rel_rich, digits=2))%   Poor $(rel_poor >= 0 ? "+" : "")$(round(rel_poor, digits=2))%   [SNR=$(round(snr, digits=2))]")
end

# Save
sweep_df = DataFrame(
    cap = [s.cap for s in all_sweep_results],
    c3_mean = [s.c3_mean for s in all_sweep_results],
    c3_std = [s.c3_std for s in all_sweep_results],
    c3_n = [s.c3_n_feasible for s in all_sweep_results],
    c4_mean = [s.c4_mean for s in all_sweep_results],
    c4_std = [s.c4_std for s in all_sweep_results],
    c4_n = [s.c4_n_feasible for s in all_sweep_results],
    signal_41 = [s.c4_mean - CONFIG1_SHARE_RICH for s in all_sweep_results],
    snr_41 = [s.c4_std > 1e-10 ? abs(s.c4_mean - CONFIG1_SHARE_RICH) / s.c4_std : Inf for s in all_sweep_results],
)
CSV.write(joinpath(results_dir, "cap_sweep_results.csv"), sweep_df)
println("\nSaved to: $(joinpath(results_dir, "cap_sweep_results.csv"))")

println("\n" * "="^80)
println("DONE")
println("="^80)
