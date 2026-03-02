########################################################################################################
# Systematic convergence testing across model configurations.
# Runs key scenarios, generates diagnostic plots, and saves results for the implementation diary.
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
results_dir = joinpath(@__DIR__, "../results/convergence_test")
obsidian_dir = "/Users/fsch/Documents/Obsidian_Vault/PhD/Inefficient mitigation/Figures"
mkpath(results_dir)

# Get backstop prices
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

# Time axis for plots (first 20 periods = 2015-2205)
time_axis = collect(2015:10:2605)
n_plot = 20  # first 20 periods

region_labels = ["rich", "poor"]
region_colors = ["#2166AC", "#D6604D"]  # blue for rich, red for poor

#------------------------------------------------------------------------------------------------------
# Test configurations
#------------------------------------------------------------------------------------------------------

struct TestConfig
    name::String
    opt_ad::Bool
    stock_ad::Bool
    cost_cap::Union{Nothing, Float64}
    cbudget::Union{Nothing, Float64}
    use_multistart::Bool
    n_starts::Int
    stop_time::Int
    tolerance::Float64
end

configs = [
    TestConfig("1_mitigation_only",        false, false, nothing, nothing, false, 1, 120,  1e-10),
    TestConfig("2_stock_flow_ad_no_constr", true,  true,  nothing, nothing, false, 1, 600,  1e-12),
    TestConfig("3_stock_flow_ad_cap008",   true,  true,  0.008,  nothing, true,  5, 3000, 1e-15),
    TestConfig("4_stock_flow_ad_cap008_cb",true,  true,  0.008,  1022.0,  true,  5, 3000, 1e-15),
]

#------------------------------------------------------------------------------------------------------
# Helper: extract key metrics from a solved model
#------------------------------------------------------------------------------------------------------

function extract_metrics(model, config)
    utility = model[:welfare, :UTILITY]
    cca = model[:emissions, :CCA][end]

    metrics = Dict{String, Any}(
        "utility" => utility,
        "cca_gtc" => cca,
    )

    if config.opt_ad
        max_cost = maximum(model[:neteconomy, :TOTAL_COST])
        metrics["max_total_cost"] = max_cost
        metrics["cost_feasible"] = config.cost_cap === nothing || max_cost <= config.cost_cap + 1e-4
    end

    if config.cbudget !== nothing
        metrics["cca_feasible"] = cca <= config.cbudget + 0.1
    end

    return metrics
end

#------------------------------------------------------------------------------------------------------
# Helper: extract time series from a solved model
#------------------------------------------------------------------------------------------------------

function extract_timeseries(model, config)
    ts = Dict{String, Matrix{Float64}}()

    ts["Emissions"] = model[:emissions, :EIND] .* 10  # Convert to GtCO2/yr (×10 for decade)
    ts["MIU"] = model[:emissions, :MIU]
    ts["Temperature"] = reshape(model[:climatedynamics, :TATM], :, 1)  # Global, not regional
    ts["CPC"] = model[:neteconomy, :CPC]

    if config.opt_ad
        ts["TOTAL_COST"] = model[:neteconomy, :TOTAL_COST]
        ts["ABATECOST_FRAC"] = model[:neteconomy, :ABATECOST_FRAC]
        ts["ADAPTCOST_FRAC"] = model[:neteconomy, :ADAPTCOST_FRAC]
        ts["T_AD_FLOW"] = model[:damages, :T_AD_FLOW]
        ts["I_AD"] = model[:damages, :I_AD]
        ts["T_AD_COMBINED"] = model[:damages, :T_AD_COMBINED]
        ts["T_AD_STOCK"] = model[:damages, :T_AD_STOCK]
    end

    return ts
end

#------------------------------------------------------------------------------------------------------
# Helper: sanity checks on time series
#------------------------------------------------------------------------------------------------------

function run_sanity_checks(ts, config, label)
    warnings = String[]

    # Check TOTAL_COST vs cost_cap
    if config.cost_cap !== nothing && haskey(ts, "TOTAL_COST")
        max_cost = maximum(ts["TOTAL_COST"])
        if max_cost > config.cost_cap + 1e-4
            push!(warnings, "COST CAP VIOLATED: max TOTAL_COST = $(round(max_cost, sigdigits=4)) > cap $(config.cost_cap)")
        end
    end

    # Check adaptation is non-negative and bounded
    if haskey(ts, "T_AD_FLOW")
        if any(ts["T_AD_FLOW"] .< -1e-6)
            push!(warnings, "NEGATIVE T_AD_FLOW detected")
        end
        if any(ts["T_AD_FLOW"] .> 5.0)
            push!(warnings, "T_AD_FLOW > 5.0 (unreasonably high)")
        end
    end
    if haskey(ts, "I_AD")
        if any(ts["I_AD"] .< -1e-6)
            push!(warnings, "NEGATIVE I_AD detected")
        end
        if any(ts["I_AD"] .> 5.0)
            push!(warnings, "I_AD > 5.0 (unreasonably high)")
        end
    end

    # Check smoothness (large jumps between adjacent periods)
    if config.cost_cap === nothing  # Only for unconstrained runs
        for (name, data) in ts
            if name in ["Emissions", "MIU", "T_AD_FLOW", "I_AD"] && size(data, 2) >= 2
                for r in 1:size(data, 2)
                    diffs = abs.(diff(data[2:n_plot, r]))
                    max_jump = maximum(diffs)
                    mean_diff = mean(diffs)
                    if max_jump > 5 * mean_diff && mean_diff > 1e-6
                        push!(warnings, "NOISY: $(name) region $(r) has jump $(round(max_jump, sigdigits=3)) vs mean diff $(round(mean_diff, sigdigits=3))")
                    end
                end
            end
        end
    end

    if isempty(warnings)
        println("  ✓ [$label] All sanity checks passed")
    else
        println("  ⚠ [$label] Sanity check warnings:")
        for w in warnings
            println("    - $w")
        end
    end

    return warnings
end

#------------------------------------------------------------------------------------------------------
# Helper: generate diagnostic plots
#------------------------------------------------------------------------------------------------------

function generate_plots(ts, config, label; save_dir=results_dir, obsidian_save=true)
    fig, axes = PythonPlot.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("$(config.name): $label", fontsize=14, fontweight="bold")

    t = time_axis[1:n_plot]

    # 1. Emissions (EIND)
    ax = axes[0, 0]
    for r in 1:min(size(ts["Emissions"], 2), 2)
        ax.plot(t, ts["Emissions"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
    end
    ax.set_ylabel("GtCO₂/yr")
    ax.set_title("EIND — Industrial Emissions")
    ax.legend(fontsize=8)
    ax.grid(true, alpha=0.3)

    # 2. Mitigation rate (MIU)
    ax = axes[0, 1]
    for r in 1:min(size(ts["MIU"], 2), 2)
        ax.plot(t, ts["MIU"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
    end
    ax.set_ylabel("Fraction")
    ax.set_title("MIU — Mitigation Rate")
    ax.legend(fontsize=8)
    ax.grid(true, alpha=0.3)

    # 3. Temperature (TATM)
    ax = axes[0, 2]
    ax.plot(t, ts["Temperature"][1:n_plot, 1], color="black")
    ax.set_ylabel("°C above 1900")
    ax.set_title("TATM — Atmospheric Temperature")
    ax.grid(true, alpha=0.3)

    # 4. Total climate cost (TOTAL_COST)
    ax = axes[1, 0]
    if haskey(ts, "TOTAL_COST")
        for r in 1:min(size(ts["TOTAL_COST"], 2), 2)
            ax.plot(t, ts["TOTAL_COST"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
        end
        if config.cost_cap !== nothing
            ax.axhline(config.cost_cap, color="red", linestyle="--", alpha=0.7, label="cost cap")
        end
        ax.set_ylabel("Fraction of GDP")
        ax.set_title("TOTAL_COST — (ABATECOST + ADAPTCOST) / YGROSS")
        ax.legend(fontsize=8)
    else
        ax.text(0.5, 0.5, "N/A (no adaptation)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("TOTAL_COST")
    end
    ax.grid(true, alpha=0.3)

    # 5. Adaptation levels (T_AD_FLOW, T_AD_COMBINED)
    ax = axes[1, 1]
    if haskey(ts, "T_AD_FLOW")
        for r in 1:min(size(ts["T_AD_FLOW"], 2), 2)
            ax.plot(t, ts["T_AD_FLOW"][1:n_plot, r], label="T_AD_FLOW $(region_labels[r])", color=region_colors[r])
            ax.plot(t, ts["T_AD_COMBINED"][1:n_plot, r], label="T_AD_COMBINED $(region_labels[r])", color=region_colors[r], linestyle="--")
        end
        ax.set_ylabel("°C adapted")
        ax.set_title("T_AD_FLOW / T_AD_COMBINED")
        ax.legend(fontsize=7)
    else
        ax.text(0.5, 0.5, "N/A (no adaptation)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("T_AD_FLOW / T_AD_COMBINED")
    end
    ax.grid(true, alpha=0.3)

    # 6. Stock adaptation investment (I_AD)
    ax = axes[1, 2]
    if haskey(ts, "I_AD")
        for r in 1:min(size(ts["I_AD"], 2), 2)
            ax.plot(t, ts["I_AD"][1:n_plot, r], label=region_labels[r], color=region_colors[r])
        end
        ax.set_ylabel("Investment level")
        ax.set_title("I_AD — Stock Adaptation Investment")
        ax.legend(fontsize=8)
    else
        ax.text(0.5, 0.5, "N/A (no stock adaptation)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("I_AD — Stock Adaptation Investment")
    end
    ax.grid(true, alpha=0.3)

    fig.tight_layout()

    # Save to results directory
    png_name = "$(config.name)_$(label).png"
    fig.savefig(joinpath(save_dir, png_name), dpi=150, bbox_inches="tight")

    # Also save to Obsidian directory for embedding
    if obsidian_save
        fig.savefig(joinpath(obsidian_dir, png_name), dpi=150, bbox_inches="tight")
    end

    PythonPlot.close("all")
    println("  Plot saved: $png_name")

    return png_name
end

#------------------------------------------------------------------------------------------------------
# Helper: save time series to CSV
#------------------------------------------------------------------------------------------------------

function save_timeseries_csv(ts, config, label)
    out_dir = joinpath(results_dir, config.name)
    mkpath(out_dir)

    for (name, data) in ts
        df = DataFrame(data, :auto)
        CSV.write(joinpath(out_dir, "$(name)_$(label).csv"), df)
    end
end

#------------------------------------------------------------------------------------------------------
# Run all configurations
#------------------------------------------------------------------------------------------------------

all_results = Dict{String, Any}()

println("\n" * "="^80)
println("SYSTEMATIC CONVERGENCE TESTING")
println("="^80)

for (i, config) in enumerate(configs)
    println("\n" * "#"^80)
    println("Config $(i)/$(length(configs)): $(config.name)")
    println("  opt_ad=$(config.opt_ad), stock_ad=$(config.stock_ad), cost_cap=$(config.cost_cap), cbudget=$(config.cbudget)")
    println("  multistart=$(config.use_multistart), n_starts=$(config.n_starts), stop_time=$(config.stop_time)s")
    println("#"^80)

    config_results = Dict{String, Any}()
    start_time = time()

    # Choose algorithm: SBPLX for unconstrained, AUGLAG for constrained
    algo = (config.cost_cap !== nothing || config.cbudget !== nothing) ? :LN_AUGLAG : :LN_SBPLX

    if config.use_multistart
        result = multistart_optimize_rice(
            config.n_starts, :LN_AUGLAG, n_opt_periods, config.stop_time, config.tolerance,
            backstop_prices;
            run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
            opt_ad=config.opt_ad, stock_ad=config.stock_ad,
            cbudget=config.cbudget, cost_cap=config.cost_cap
        )

        model = result[7]
        metrics = extract_metrics(model, config)
        ts = extract_timeseries(model, config)

        config_results["metrics"] = metrics
        config_results["timeseries"] = ts
        config_results["result"] = result

        # Generate plots and run sanity checks
        png = generate_plots(ts, config, "refined")
        config_results["plot"] = png
        config_results["warnings"] = run_sanity_checks(ts, config, "refined")
        save_timeseries_csv(ts, config, "refined")
    else
        # Single-start
        result = optimize_rice(
            algo, n_opt_periods, config.stop_time, config.tolerance,
            backstop_prices;
            run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
            opt_ad=config.opt_ad, stock_ad=config.stock_ad,
            cbudget=config.cbudget, cost_cap=config.cost_cap
        )

        model = result[7]
        metrics = extract_metrics(model, config)
        ts = extract_timeseries(model, config)

        config_results["metrics"] = metrics
        config_results["timeseries"] = ts
        config_results["result"] = result

        png = generate_plots(ts, config, "single")
        config_results["plot"] = png
        config_results["warnings"] = run_sanity_checks(ts, config, "single")
        save_timeseries_csv(ts, config, "single")
    end

    elapsed = round(time() - start_time, digits=1)
    config_results["elapsed_s"] = elapsed

    all_results[config.name] = config_results

    # Save raw policy vector for warm-starting homotopy (especially Config 3)
    if haskey(config_results, "result")
        pv_dir = joinpath(results_dir, config.name)
        mkpath(pv_dir)
        pv_path = joinpath(pv_dir, "policy_vector.csv")
        policy_vec = config_results["result"][1]
        CSV.write(pv_path, DataFrame(x=policy_vec))
        println("  Policy vector saved to: $pv_path ($(length(policy_vec)) elements)")
    end

    # Print summary for this config
    m = config_results["metrics"]
    println("\n  --- SUMMARY ---")
    println("  Utility:         $(round(m["utility"], sigdigits=8))")
    println("  CCA (GtC):       $(round(m["cca_gtc"], sigdigits=6))")
    if haskey(m, "max_total_cost")
        println("  Max TOTAL_COST:  $(round(m["max_total_cost"], sigdigits=4))")
        println("  Cost feasible:   $(m["cost_feasible"])")
    end
    if haskey(m, "cca_feasible")
        println("  CCA feasible:    $(m["cca_feasible"])")
    end
    println("  Time:            $(elapsed)s")
end

#------------------------------------------------------------------------------------------------------
# Print summary table
#------------------------------------------------------------------------------------------------------

println("\n\n" * "="^80)
println("SUMMARY TABLE")
println("="^80)
println()

# Header
header = rpad("Config", 35) * rpad("Utility", 14) * rpad("CCA(GtC)", 12) * rpad("MaxCost", 10) * rpad("Feasible", 10) * rpad("Time(s)", 10)
println(header)
println("-"^length(header))

for config in configs
    r = all_results[config.name]
    m = r["metrics"]

    cost_str = haskey(m, "max_total_cost") ? string(round(m["max_total_cost"], sigdigits=4)) : "—"
    feas_str = ""
    if haskey(m, "cost_feasible")
        feas_str *= m["cost_feasible"] ? "cost✓" : "cost✗"
    end
    if haskey(m, "cca_feasible")
        feas_str *= (feas_str == "" ? "" : " ") * (m["cca_feasible"] ? "cca✓" : "cca✗")
    end
    if feas_str == ""
        feas_str = "—"
    end

    row = rpad(config.name, 35) *
          rpad(string(round(m["utility"], sigdigits=8)), 14) *
          rpad(string(round(m["cca_gtc"], sigdigits=6)), 12) *
          rpad(cost_str, 10) *
          rpad(feas_str, 10) *
          rpad(string(r["elapsed_s"]), 10)
    println(row)
end

#------------------------------------------------------------------------------------------------------
# Compare cumulative emissions shares
#------------------------------------------------------------------------------------------------------

println("\n\n" * "="^80)
println("CUMULATIVE EMISSIONS ANALYSIS")
println("="^80)

for config in configs
    r = all_results[config.name]
    ts = r["timeseries"]
    em = ts["Emissions"]  # Already in GtCO2/yr (×10)

    cum_rich = sum(em[:, 1])
    cum_poor = sum(em[:, 2])
    cum_total = cum_rich + cum_poor
    share_rich = cum_rich / cum_total * 100
    share_poor = cum_poor / cum_total * 100

    println("\n$(config.name):")
    println("  Cumulative: rich=$(round(cum_rich, digits=1)) poor=$(round(cum_poor, digits=1)) total=$(round(cum_total, digits=1)) GtCO₂")
    println("  Shares:     rich=$(round(share_rich, digits=1))% poor=$(round(share_poor, digits=1))%")
end

#------------------------------------------------------------------------------------------------------
# Save summary to file
#------------------------------------------------------------------------------------------------------

summary_path = joinpath(results_dir, "convergence_summary.txt")
open(summary_path, "w") do f
    println(f, "Systematic Convergence Test — $(Dates.now())")
    println(f, "="^60)
    for config in configs
        r = all_results[config.name]
        m = r["metrics"]
        println(f, "\n$(config.name):")
        println(f, "  Utility: $(m["utility"])")
        println(f, "  CCA: $(m["cca_gtc"]) GtC")
        if haskey(m, "max_total_cost")
            println(f, "  Max TOTAL_COST: $(m["max_total_cost"])")
        end
        println(f, "  Time: $(r["elapsed_s"])s")
        if !isempty(r["warnings"])
            println(f, "  Warnings:")
            for w in r["warnings"]
                println(f, "    - $w")
            end
        end
    end
end

println("\n\nSummary saved to: $summary_path")
println("All plots saved to: $results_dir")
println("Obsidian plots saved to: $obsidian_dir (Figures subfolder)")
println("\nDone!")
