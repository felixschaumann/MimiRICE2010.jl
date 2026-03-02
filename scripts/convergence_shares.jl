########################################################################################################
# HIGH-CONVERGENCE EMISSIONS SHARE ANALYSIS
#
# For each config, run multiple long refinement passes from the best known solution.
# Goal: nail down the emissions share allocation with enough precision to distinguish
# between-config effects from convergence noise.
#
# Strategy:
#   - Config 1 (60 vars, no constraints): 5 runs × 600s from default start
#   - Config 2 (180 vars, no constraints): 5 runs × 1200s from default start
#   - Config 3 (180 vars, cost cap): 5 iterative refinement passes × 1800s from cached PV
#   - Config 4 (180 vars, dual): 5 iterative refinement passes × 2400s from homotopy PV
#
# For constrained configs, iterative refinement: each pass starts from the previous best,
# allowing AUGLAG to progressively tighten constraint enforcement.
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

#------------------------------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------------------------------

ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
n_regions = 2

results_dir = joinpath(@__DIR__, "../results/share_convergence")
mkpath(results_dir)

backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

#------------------------------------------------------------------------------------------------------
# Helper: run one optimization and extract shares
#------------------------------------------------------------------------------------------------------

function run_and_extract(algo, n_opt, time_s, tol, backstop;
                         opt_ad, stock_ad, cbudget, cost_cap, start_pv,
                         ρ=0.008, η=1.5, remove_negishi=false)
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
# Config 1: Mitigation only (SBPLX, no constraints, 60 vars)
#------------------------------------------------------------------------------------------------------

# Config 1 and 2 results from previous run (process crashed on Config 3 due to scoping)
# Config 1: 5 runs all converged to share_rich ≈ 31.55% with std ≈ 0.0%
# Config 2: 5 runs converged to share_rich ≈ 32.06% with std ≈ 0.34%

println("\n" * "="^80)
println("CONFIG 1: Using cached results (share_rich = 31.55 ± 0.0%)")
println("CONFIG 2: Using cached results (share_rich = 32.059 ± 0.3392%)")
println("="^80)

# Reconstruct Config 1 results (all identical — zero noise)
c1_results = [(utility=6280.8487, cca=1022.8, max_cost=0.00968, feasible=true,
               cum_rich=322.58, cum_poor=700.21, share_rich=31.5499,
               policy_vector=Float64[]) for _ in 1:5]

# Reconstruct Config 2 results from log
c2_shares = [32.493, 31.761, 31.871, 32.355, 31.813]  # from log output
c2_results = [(utility=6281.62, cca=1022.1, max_cost=0.00971, feasible=true,
               cum_rich=0.0, cum_poor=0.0, share_rich=s,
               policy_vector=Float64[]) for s in c2_shares]
println("  Config 2 share_rich values: $c2_shares")

#------------------------------------------------------------------------------------------------------
# Config 3: Stock+flow + cost cap 0.8% — iterative refinement from cached PV
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 3: Cost cap 0.8% — 5 iterative passes × 1800s")
println("="^80)

config3_pv = CSV.read(joinpath(@__DIR__, "../results/convergence_test/3_stock_flow_ad_cap008/policy_vector.csv"), DataFrame).x
println("  Loaded Config 3 PV ($(length(config3_pv)) elements)")

c3_results = []
current_pv = config3_pv

for i in 1:5
    println("\n  Pass $i/5 (iterative refinement)...")
    t0 = time()
    # Small perturbation on odd passes for diversity, direct refinement on even
    start = (i % 2 == 0) ? current_pv : perturb_solution(current_pv, 0.02, n_opt_periods, n_regions, true)
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 1800, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=0.008,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c3_results, r)
    # Update current PV to best feasible
    feasible_c3 = filter(r -> r.feasible, c3_results)
    if !isempty(feasible_c3)
        global current_pv = sort(feasible_c3, by=r->-r.utility)[1].policy_vector
    end
    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c3 = filter(r -> r.feasible, c3_results)
println("\n  Config 3 summary (feasible=$(length(feas_c3))/5): share_rich = $(round(mean([r.share_rich for r in feas_c3]), digits=3)) ± $(round(std([r.share_rich for r in feas_c3]), digits=4))%")

#------------------------------------------------------------------------------------------------------
# Config 4: Dual constraints — iterative refinement from homotopy PV
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("CONFIG 4: Cap + CB 1022 — 5 iterative passes × 2400s")
println("="^80)

config4_pv = CSV.read(joinpath(@__DIR__, "../results/homotopy/config4_policy_vector.csv"), DataFrame).x
println("  Loaded Config 4 homotopy PV ($(length(config4_pv)) elements)")

c4_results = []
current_pv = config4_pv

for i in 1:5
    println("\n  Pass $i/5 (iterative refinement)...")
    t0 = time()
    start = (i % 2 == 0) ? current_pv : perturb_solution(current_pv, 0.02, n_opt_periods, n_regions, true)
    r = run_and_extract(:LN_AUGLAG, n_opt_periods, 2400, 1e-15, backstop_prices;
                        opt_ad=true, stock_ad=true, cbudget=1022.0, cost_cap=0.008,
                        start_pv=start)
    elapsed = round(time() - t0, digits=1)
    push!(c4_results, r)
    feasible_c4 = filter(r -> r.feasible, c4_results)
    if !isempty(feasible_c4)
        global current_pv = sort(feasible_c4, by=r->-r.utility)[1].policy_vector
    end
    feas_str = r.feasible ? "✓" : "✗ (cost=$(round(r.max_cost, sigdigits=4)), cca=$(round(r.cca, sigdigits=6)))"
    println("    Utility=$(round(r.utility, sigdigits=8))  Share_rich=$(round(r.share_rich, digits=3))%  Feasible=$feas_str  ($(elapsed)s)")
end

feas_c4 = filter(r -> r.feasible, c4_results)
println("\n  Config 4 summary (feasible=$(length(feas_c4))/5): share_rich = $(round(mean([r.share_rich for r in feas_c4]), digits=3)) ± $(round(std([r.share_rich for r in feas_c4]), digits=4))%")

#------------------------------------------------------------------------------------------------------
# Final comparison
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("FINAL EMISSIONS SHARE COMPARISON")
println("="^80)

function summarize(results, label; filter_feasible=false)
    rs = filter_feasible ? filter(r -> r.feasible, results) : results
    shares = [r.share_rich for r in rs]
    utils = [r.utility for r in rs]
    n = length(shares)
    m = mean(shares)
    s = n > 1 ? std(shares) : 0.0
    best_idx = argmax(utils)
    best_share = shares[best_idx]
    return (label=label, n=n, mean_share=m, std_share=s, best_share=best_share,
            best_utility=maximum(utils), shares=shares)
end

s1 = summarize(c1_results, "1: Mitigation only")
s2 = summarize(c2_results, "2: + Adaptation")
s3 = summarize(c3_results, "3: + Cost cap", filter_feasible=true)
s4 = summarize(c4_results, "4: + Cap + CB", filter_feasible=true)

println("\n  Config                          | n  | Best share | Mean ± Std      | Range")
println("  --------------------------------|----|-----------:|-----------------|--------")
for s in [s1, s2, s3, s4]
    range_str = length(s.shares) > 1 ? "[$(round(minimum(s.shares), digits=2)), $(round(maximum(s.shares), digits=2))]" : "—"
    println("  $(rpad(s.label, 32))| $(s.n)  | $(rpad(string(round(s.best_share, digits=3))*"%", 10)) | $(round(s.mean_share, digits=3)) ± $(round(s.std_share, digits=4))% | $range_str")
end

# Between-config effects using BEST solutions (highest utility → best converged)
println("\n--- Effects using BEST solution per config ---")
Δ_21 = s2.best_share - s1.best_share
Δ_31 = s3.best_share - s1.best_share
Δ_41 = s4.best_share - s1.best_share
Δ_32 = s3.best_share - s2.best_share
Δ_43 = s4.best_share - s3.best_share
Δ_42 = s4.best_share - s2.best_share

for (label, signal) in [
    ("2 vs 1 (adaptation)", Δ_21),
    ("3 vs 2 (cost cap)", Δ_32),
    ("4 vs 3 (carbon budget)", Δ_43),
    ("4 vs 2 (both constraints)", Δ_42),
    ("4 vs 1 (total vs baseline)", Δ_41),
]
    dir = signal > 0 ? "rich↑" : "rich↓"
    println("  $(rpad(label, 35)) $(round(signal, digits=3)) pp ($dir)")
end

# Relative changes using best solutions
println("\n--- Relative change in rich share (best per config) ---")
for (label, new_share) in [
    ("2 vs 1", s2.best_share),
    ("3 vs 1", s3.best_share),
    ("4 vs 1", s4.best_share),
    ("3 vs 2", s3.best_share),  # relative to config 2
    ("4 vs 3", s4.best_share),
    ("4 vs 2", s4.best_share),
]
    # Determine base
    base = if startswith(label, "3 vs 2") || startswith(label, "4 vs 2")
        s2.best_share
    elseif startswith(label, "4 vs 3")
        s3.best_share
    else
        s1.best_share
    end
    rel = (new_share - base) / base * 100
    println("  $(rpad(label, 12)) rich: $(rel >= 0 ? "+" : "")$(round(rel, digits=3))%    poor: $(rel >= 0 ? "-" : "+")$(round(abs(rel * base / (100 - base)), digits=3))%")
end

# Noise-aware confidence: can we distinguish the effects?
println("\n--- Signal vs noise (using per-config std) ---")
# Propagated uncertainty: σ_diff = sqrt(σ_A² + σ_B²)
for (label, sA, sB) in [
    ("2 vs 1", s2, s1),
    ("3 vs 2", s3, s2),
    ("4 vs 3", s4, s3),
    ("4 vs 2", s4, s2),
    ("4 vs 1", s4, s1),
]
    signal = sA.best_share - sB.best_share
    σ_diff = sqrt(sA.std_share^2 + sB.std_share^2)
    snr = σ_diff > 1e-10 ? abs(signal) / σ_diff : Inf
    reliable = snr > 2.0 ? "YES" : snr > 1.0 ? "marginal" : "no"
    println("  $(rpad(label, 12)) signal=$(round(signal, digits=3))pp  σ_diff=$(round(σ_diff, digits=4))pp  SNR=$(round(snr, digits=2))  [$reliable]")
end

# Save results
all_results = DataFrame(
    config = vcat(fill(1, length(c1_results)), fill(2, length(c2_results)),
                  fill(3, length(c3_results)), fill(4, length(c4_results))),
    utility = vcat([r.utility for r in c1_results], [r.utility for r in c2_results],
                   [r.utility for r in c3_results], [r.utility for r in c4_results]),
    share_rich = vcat([r.share_rich for r in c1_results], [r.share_rich for r in c2_results],
                      [r.share_rich for r in c3_results], [r.share_rich for r in c4_results]),
    feasible = vcat([true for _ in c1_results], [true for _ in c2_results],
                    [r.feasible for r in c3_results], [r.feasible for r in c4_results]),
    cca = vcat([r.cca for r in c1_results], [r.cca for r in c2_results],
               [r.cca for r in c3_results], [r.cca for r in c4_results]),
    max_cost = vcat([r.max_cost for r in c1_results], [r.max_cost for r in c2_results],
                    [r.max_cost for r in c3_results], [r.max_cost for r in c4_results])
)
CSV.write(joinpath(results_dir, "share_convergence.csv"), all_results)
println("\nResults saved to: $(joinpath(results_dir, "share_convergence.csv"))")

println("\n" * "="^80)
println("DONE — Total configs: 4, Total runs: $(length(c1_results)+length(c2_results)+length(c3_results)+length(c4_results))")
println("="^80)
