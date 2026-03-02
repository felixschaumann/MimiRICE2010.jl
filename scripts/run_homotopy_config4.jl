########################################################################################################
# Homotopy approach for Config 4 (cost_cap=0.008 + cbudget=1022 GtC).
#
# Problem: Config 4 fails to converge from cold start (0/6 multi-starts feasible).
# Solution: Trace the solution path from Config 3 (cost_cap only, converges solidly)
# through progressively tighter carbon budgets until reaching the target.
#
# Config 3 CCA ≈ 1031 GtC, target is 1022. The 5-step homotopy path:
#   Step 1: cbudget=1045  (slack — activates AUGLAG carbon budget machinery)
#   Step 2: cbudget=1038  (still slack, AUGLAG calibrates multipliers)
#   Step 3: cbudget=1031  (just binding)
#   Step 4: cbudget=1026  (tighten by 5 GtC)
#   Step 5: cbudget=1022  (final target)
#
# Time budget: ~4500s (~75 min), plus Config 3 if not cached (~3000s).
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using CSV
using DataFrames
using Dates
using PythonPlot
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

# Output directories
results_dir = joinpath(@__DIR__, "../results/homotopy")
convergence_dir = joinpath(@__DIR__, "../results/convergence_test")
obsidian_dir = "/Users/fsch/Documents/Obsidian_Vault/PhD/Inefficient mitigation/Figures"
mkpath(results_dir)

# Get backstop prices
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

# Plot settings
time_axis = collect(2015:10:2605)
n_plot = 20
region_labels = ["rich", "poor"]
region_colors = ["#2166AC", "#D6604D"]

#------------------------------------------------------------------------------------------------------
# Phase A: Obtain Config 3 solution (cost_cap=0.008 only, no carbon budget)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE A: Obtain Config 3 solution")
println("="^80)

config3_pv_path = joinpath(convergence_dir, "3_stock_flow_ad_cap008", "policy_vector.csv")
config3_solution = nothing

if isfile(config3_pv_path)
    println("Loading cached Config 3 policy vector from: $config3_pv_path")
    pv_df = CSV.read(config3_pv_path, DataFrame)
    config3_solution = pv_df.x
    println("  Loaded $(length(config3_solution))-element policy vector")

    # Validate: run the model with this policy and check feasibility
    println("  Validating...")
    validation_result = optimize_rice(
        :LN_AUGLAG, n_opt_periods, 120, 1e-10, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=0.008,
        ext_starting_points=config3_solution
    )
    val_model = validation_result[7]
    val_utility = validation_result[9]
    val_cca = val_model[:emissions, :CCA][end]
    val_max_cost = maximum(val_model[:neteconomy, :TOTAL_COST])
    println("  Validation: utility=$(round(val_utility, sigdigits=8)), CCA=$(round(val_cca, sigdigits=6)), max_cost=$(round(val_max_cost, sigdigits=4))")

    # Use the validated (possibly refined) solution
    config3_solution = validation_result[1]
    config3_utility = val_utility
    config3_cca = val_cca
    config3_max_cost = val_max_cost
else
    println("No cached Config 3 solution found. Running Config 3 from scratch...")
    println("  (This will take ~3000s)")

    config3_result = multistart_optimize_rice(
        5, :LN_AUGLAG, n_opt_periods, 3000, 1e-15, backstop_prices;
        run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
        opt_ad=true, stock_ad=true, cbudget=nothing, cost_cap=0.008
    )

    config3_solution = config3_result[1]
    config3_model = config3_result[7]
    config3_utility = config3_result[9]
    config3_cca = config3_model[:emissions, :CCA][end]
    config3_max_cost = maximum(config3_model[:neteconomy, :TOTAL_COST])

    # Save policy vector for future reuse
    pv_dir = joinpath(convergence_dir, "3_stock_flow_ad_cap008")
    mkpath(pv_dir)
    CSV.write(config3_pv_path, DataFrame(x=config3_solution))
    println("  Saved Config 3 policy vector to: $config3_pv_path")
end

println("\nConfig 3 baseline:")
println("  Utility:  $(round(config3_utility, sigdigits=8))")
println("  CCA:      $(round(config3_cca, sigdigits=6)) GtC")
println("  Max cost: $(round(config3_max_cost, sigdigits=4))")

#------------------------------------------------------------------------------------------------------
# Phase B: Homotopy path (5 steps from slack to target carbon budget)
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE B: Homotopy path")
println("="^80)

homotopy_steps = [
    (cost_cap=0.008, cbudget=1045.0),  # Step 1: slack (CCA≈1031 < 1045)
    (cost_cap=0.008, cbudget=1038.0),  # Step 2: still slack, AUGLAG calibrates
    (cost_cap=0.008, cbudget=1031.0),  # Step 3: just binding
    (cost_cap=0.008, cbudget=1026.0),  # Step 4: tighten by 5 GtC
    (cost_cap=0.008, cbudget=1022.0),  # Step 5: final target
]

homotopy_results = homotopy_solve(
    homotopy_steps,
    config3_solution,
    :LN_AUGLAG,
    n_opt_periods,
    backstop_prices;
    time_per_step=600,
    first_step_time=1200,
    final_step_time=1500,
    first_step_starts=4,
    final_step_starts=4,
    tolerance=1e-15,
    ρ=ρ, η=η,
    remove_negishi=remove_negishi,
    opt_ad=true, stock_ad=true
)

# Save homotopy results to CSV
homotopy_df = DataFrame(
    step = [r.step for r in homotopy_results],
    cost_cap = [r.cost_cap for r in homotopy_results],
    cbudget = [r.cbudget for r in homotopy_results],
    utility = [r.utility for r in homotopy_results],
    cca = [r.cca for r in homotopy_results],
    max_cost = [r.max_cost for r in homotopy_results],
    feasible = [r.feasible for r in homotopy_results],
    smoothness = [r.smoothness for r in homotopy_results],
    elapsed = [r.elapsed for r in homotopy_results]
)
CSV.write(joinpath(results_dir, "homotopy_results.csv"), homotopy_df)
println("\nHomotopy results saved to: $(joinpath(results_dir, "homotopy_results.csv"))")

# Save final policy vector
final_step = homotopy_results[end]
CSV.write(joinpath(results_dir, "config4_policy_vector.csv"), DataFrame(x=final_step.policy_vector))
println("Final Config 4 policy vector saved.")

#------------------------------------------------------------------------------------------------------
# Phase C: Diagnostics
#------------------------------------------------------------------------------------------------------

println("\n" * "="^80)
println("PHASE C: Diagnostics")
println("="^80)

# --- C1: Homotopy convergence plot ---
fig, axes = PythonPlot.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Homotopy Convergence: Config 4", fontsize=14, fontweight="bold")

cbudgets = [r.cbudget for r in homotopy_results]
utilities = [r.utility for r in homotopy_results]
ccas = [r.cca for r in homotopy_results]
max_costs = [r.max_cost for r in homotopy_results]

# Utility vs step
ax = axes[0]
ax.plot(cbudgets, utilities, "o-", color="#2166AC", linewidth=2, markersize=8)
ax.set_xlabel("Carbon budget (GtC)")
ax.set_ylabel("Utility")
ax.set_title("Utility along homotopy path")
ax.invert_xaxis()
ax.grid(true, alpha=0.3)

# CCA vs step
ax = axes[1]
ax.plot(cbudgets, ccas, "o-", color="#D6604D", linewidth=2, markersize=8)
ax.plot(cbudgets, cbudgets, "--", color="gray", alpha=0.5, label="budget = CCA")
ax.set_xlabel("Carbon budget (GtC)")
ax.set_ylabel("CCA (GtC)")
ax.set_title("Cumulative emissions vs budget")
ax.invert_xaxis()
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

# Max cost vs step
ax = axes[2]
ax.plot(cbudgets, max_costs, "o-", color="#4DAF4A", linewidth=2, markersize=8)
ax.axhline(0.008, color="red", linestyle="--", alpha=0.7, label="cost cap")
ax.set_xlabel("Carbon budget (GtC)")
ax.set_ylabel("Max TOTAL_COST")
ax.set_title("Cost constraint along path")
ax.invert_xaxis()
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

fig.tight_layout()
fig.savefig(joinpath(results_dir, "homotopy_convergence.png"), dpi=150, bbox_inches="tight")
fig.savefig(joinpath(obsidian_dir, "homotopy_convergence.png"), dpi=150, bbox_inches="tight")
PythonPlot.close("all")
println("  Homotopy convergence plot saved.")

# --- C2: Final Config 4 time series (6-panel) ---
final_result = final_step.result
final_model = final_result[7]

ts = Dict{String, Any}()
ts["Emissions"] = final_model[:emissions, :EIND] .* 10  # GtCO2/yr
ts["MIU"] = final_model[:emissions, :MIU]
ts["Temperature"] = reshape(final_model[:climatedynamics, :TATM], :, 1)
ts["TOTAL_COST"] = final_model[:neteconomy, :TOTAL_COST]
ts["T_AD_FLOW"] = final_model[:damages, :T_AD_FLOW]
ts["T_AD_COMBINED"] = final_model[:damages, :T_AD_COMBINED]
ts["I_AD"] = final_model[:damages, :I_AD]
ts["CPC"] = final_model[:neteconomy, :CPC]

t = time_axis[1:n_plot]

fig, axes = PythonPlot.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Config 4 (Homotopy): cost_cap=0.008, cbudget=1022", fontsize=14, fontweight="bold")

# 1. Emissions
ax = axes[0, 0]
for r in 1:2
    ax.plot(t, ts["Emissions"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
end
ax.set_ylabel("GtCO₂/yr")
ax.set_title("EIND — Industrial Emissions")
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

# 2. MIU
ax = axes[0, 1]
for r in 1:2
    ax.plot(t, ts["MIU"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
end
ax.set_ylabel("Fraction")
ax.set_title("MIU — Mitigation Rate")
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

# 3. Temperature
ax = axes[0, 2]
ax.plot(t, ts["Temperature"][1:n_plot, 1], color="black")
ax.set_ylabel("°C above 1900")
ax.set_title("TATM — Atmospheric Temperature")
ax.grid(true, alpha=0.3)

# 4. TOTAL_COST
ax = axes[1, 0]
for r in 1:2
    ax.plot(t, ts["TOTAL_COST"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
end
ax.axhline(0.008, color="red", linestyle="--", alpha=0.7, label="cost cap")
ax.set_ylabel("Fraction of GDP")
ax.set_title("TOTAL_COST")
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

# 5. Adaptation levels
ax = axes[1, 1]
for r in 1:2
    ax.plot(t, ts["T_AD_FLOW"][1:n_plot, r], label="T_AD_FLOW $(region_labels[r])", color=region_colors[r])
    ax.plot(t, ts["T_AD_COMBINED"][1:n_plot, r], label="T_AD_COMBINED $(region_labels[r])", color=region_colors[r], linestyle="--")
end
ax.set_ylabel("°C adapted")
ax.set_title("Adaptation Levels")
ax.legend(fontsize=7)
ax.grid(true, alpha=0.3)

# 6. Stock adaptation investment
ax = axes[1, 2]
for r in 1:2
    ax.plot(t, ts["I_AD"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
end
ax.set_ylabel("Investment level")
ax.set_title("I_AD — Stock Adaptation Investment")
ax.legend(fontsize=8)
ax.grid(true, alpha=0.3)

fig.tight_layout()
fig.savefig(joinpath(results_dir, "config4_homotopy_timeseries.png"), dpi=150, bbox_inches="tight")
fig.savefig(joinpath(obsidian_dir, "config4_homotopy_timeseries.png"), dpi=150, bbox_inches="tight")
PythonPlot.close("all")
println("  Config 4 time series plot saved.")

# --- C3: Smoothness check ---
println("\n--- Smoothness check on final Config 4 ---")
for (name, data) in ts
    if name in ["Emissions", "MIU", "T_AD_FLOW", "I_AD"] && ndims(data) == 2 && size(data, 2) >= 2
        for r in 1:2
            diffs = abs.(diff(data[2:n_plot, r]))
            max_jump = maximum(diffs)
            mean_diff = mean(diffs)
            ratio = mean_diff > 1e-8 ? max_jump / mean_diff : 0.0
            status = ratio > 5.0 ? "NOISY" : "ok"
            println("  $(rpad(name, 14)) $(region_labels[r]): max_jump=$(round(max_jump, sigdigits=3)), mean=$(round(mean_diff, sigdigits=3)), ratio=$(round(ratio, sigdigits=2)) [$status]")
        end
    end
end

# --- C4: Save time series CSVs ---
for (name, data) in ts
    CSV.write(joinpath(results_dir, "config4_$(name).csv"), DataFrame(data isa Matrix ? data : reshape(data, :, 1), :auto))
end
println("  Time series CSVs saved.")

# --- C5: Cumulative emissions analysis ---
println("\n--- Cumulative emissions ---")
em = ts["Emissions"]
cum_rich = sum(em[:, 1])
cum_poor = sum(em[:, 2])
cum_total = cum_rich + cum_poor
share_rich = cum_rich / cum_total * 100
share_poor = cum_poor / cum_total * 100
println("  Cumulative: rich=$(round(cum_rich, digits=1)) poor=$(round(cum_poor, digits=1)) total=$(round(cum_total, digits=1)) GtCO2")
println("  Shares:     rich=$(round(share_rich, digits=1))% poor=$(round(share_poor, digits=1))%")

# Final report
println("\n" * "="^80)
println("HOMOTOPY COMPLETE")
println("="^80)
final_r = homotopy_results[end]
println("  Final utility:  $(round(final_r.utility, sigdigits=8))")
println("  Final CCA:      $(round(final_r.cca, sigdigits=6)) GtC (budget=1022)")
println("  Final max cost: $(round(final_r.max_cost, sigdigits=4)) (cap=0.008)")
println("  Feasible:       $(final_r.feasible)")
println("  Smoothness:     $(round(final_r.smoothness, sigdigits=3)) (max MIU jump)")
println("\nResults dir: $results_dir")
println("Done!")
