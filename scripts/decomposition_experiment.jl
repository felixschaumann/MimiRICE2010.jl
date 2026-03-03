########################################################################################################
# "Full Minus One" Decomposition Experiment
#
# Start from the full asymmetric model, then symmetrize one parameter group at a time.
# The contribution of each group = full_model_share - full_minus_X_share.
#
# This avoids the degeneracy problem of the symmetric baseline: each experiment
# retains most asymmetries, giving the optimizer a well-defined unique optimum.
########################################################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

using NLopt
using CSVFiles
using DataFrames
using Statistics
using Dates

# Load required files
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
stop_time = 1200       # seconds per optimization run
tolerance = 1e-15
carbon_budget = 1022.5950757761052
cost_cap = 0.008

results_folder = "Decomposition_v2"
results_base = joinpath(@__DIR__, "../results", results_folder)
mkpath(results_base)

#------------------------------------------------------------------------------------------------------
# Define experiments: name → list of params to SYMMETRIZE (everything else stays real)
#------------------------------------------------------------------------------------------------------
experiments = [
    "full_model"        => Symbol[],   # baseline: nothing symmetrized = standard model
    "minus_damages"     => [:a1, :a2, :a3],
    "minus_mac"         => [:sigma, :pbacktime, :cost1, :expcost2],
    "minus_pop_gdp"     => [:l, :k0, :al],
    "minus_negishi"     => [:alpha, :scale1, :scale2, :rr],
    "minus_slr"         => [:slrmultiplier, :slrelasticity, :slrdamlinear, :slrdamquadratic],
]

#------------------------------------------------------------------------------------------------------
# Define model configurations
#------------------------------------------------------------------------------------------------------
# Config 1:  mitigation-only, no constraints
# Config 2f: + flow adaptation, no constraints
# Config 3f: + flow adaptation + cost cap (no CB)
# Config 4f: + flow adaptation + cost cap + carbon budget

configs = [
    (name="config1",  opt_ad=false, stock_ad=false, cbudget=nothing,       cost_cap=nothing),
    (name="config2f", opt_ad=true,  stock_ad=false, cbudget=nothing,       cost_cap=nothing),
    (name="config3f", opt_ad=true,  stock_ad=false, cbudget=nothing,       cost_cap=cost_cap),
    (name="config4f", opt_ad=true,  stock_ad=false, cbudget=carbon_budget, cost_cap=cost_cap),
]

#------------------------------------------------------------------------------------------------------
# Helper: compute rich share of total emissions (sum over all periods)
#------------------------------------------------------------------------------------------------------
function compute_share_rich(model)
    eind = model[:emissions, :EIND]  # 60×2 matrix
    total_rich = sum(eind[:, 1])
    total_poor = sum(eind[:, 2])
    return total_rich / (total_rich + total_poor) * 100.0
end

#------------------------------------------------------------------------------------------------------
# Helper: create model constructor closure for a given set of params to symmetrize
#------------------------------------------------------------------------------------------------------
function make_partial_symmetric_constructor(symmetrize::Vector{Symbol})
    return (ρ, η, remove_negishi, opt_ad=false, stock_ad=false, cbudget=nothing) ->
        create_rice_partial_symmetric(ρ, η, remove_negishi, opt_ad, stock_ad, cbudget; symmetrize=symmetrize)
end

#------------------------------------------------------------------------------------------------------
# Run all experiments
#------------------------------------------------------------------------------------------------------

# Results table: experiment × config → (share_rich, utility, cca, max_cost, convergence)
results_table = Dict{String, Dict{String, NamedTuple}}()

# Get backstop prices from the standard model (needed for optimization)
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

println("\n" * "="^70)
println("FULL-MINUS-ONE DECOMPOSITION EXPERIMENT")
println("="^70)
println("Experiments: ", join([e[1] for e in experiments], ", "))
println("Configs:     ", join([c.name for c in configs], ", "))
println("="^70)

for (exp_name, symmetrize_params) in experiments
    println("\n" * "#"^70)
    println("EXPERIMENT: $exp_name  (symmetrize: $(isempty(symmetrize_params) ? "nothing" : join(symmetrize_params, ", ")))")
    println("#"^70)

    results_table[exp_name] = Dict{String, NamedTuple}()

    # For full_model (no symmetrization), use the standard create_rice
    if isempty(symmetrize_params)
        mc = create_rice
    else
        mc = make_partial_symmetric_constructor(symmetrize_params)
    end

    # Track the previous config's solution for warm-starting Config 4f from Config 3f
    prev_solution = nothing
    prev_config_name = ""

    for cfg in configs
        println("\n--- $exp_name / $(cfg.name) ---")

        exp_dir = joinpath(results_base, exp_name, cfg.name)
        mkpath(exp_dir)

        # For Config 4f, warm-start from Config 3f solution
        ext_sp = nothing
        if cfg.name == "config4f" && prev_config_name == "config3f" && prev_solution !== nothing
            println("  Warm-starting from config3f solution")
            ext_sp = prev_solution
        end

        local result
        t_start = time()
        try
            result = optimize_rice(
                optimization_algorithm, n_opt_periods, stop_time, tolerance, backstop_prices;
                run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                opt_ad=cfg.opt_ad, stock_ad=cfg.stock_ad,
                cbudget=cfg.cbudget, cost_cap=cfg.cost_cap,
                ext_starting_points=ext_sp,
                model_constructor=mc
            )
        catch e
            println("  ERROR: $e")
            results_table[exp_name][cfg.name] = (share_rich=NaN, utility=NaN, cca=NaN, max_cost=NaN, convergence="ERROR", elapsed=0.0)
            continue
        end
        elapsed = round(time() - t_start, digits=1)

        opt_model = result[7]
        raw_utility = result[9]
        convergence = result[8]

        # Save solution vector for warm-starting next config
        prev_solution = result[1]
        prev_config_name = cfg.name

        # Compute share and diagnostics
        share_rich = compute_share_rich(opt_model)
        cca = opt_model[:emissions, :CCA][end]
        max_cost = cfg.opt_ad ? maximum(opt_model[:neteconomy, :TOTAL_COST]) : NaN

        results_table[exp_name][cfg.name] = (
            share_rich=share_rich, utility=raw_utility, cca=cca,
            max_cost=max_cost, convergence=string(convergence), elapsed=elapsed
        )

        println("  Utility:    $(round(raw_utility, sigdigits=8))")
        println("  Share rich: $(round(share_rich, digits=3))%")
        println("  CCA:        $(round(cca, digits=1)) GtC")
        if cfg.opt_ad
            println("  Max cost:   $(round(max_cost, sigdigits=4))")
        end
        println("  Convergence: $convergence  ($(elapsed)s)")

        # Save key outputs
        save(joinpath(exp_dir, "Emissions.csv"), DataFrame(opt_model[:emissions, :EIND], :auto))
        save(joinpath(exp_dir, "MitigationRate.csv"), DataFrame(result[3], :auto))
        if cfg.opt_ad
            save(joinpath(exp_dir, "FlowAdaptation.csv"), DataFrame(result[4], :auto))
        end
        save(joinpath(exp_dir, "Temperature.csv"), DataFrame(temp=opt_model[:climatedynamics, :TATM]))
        save(joinpath(exp_dir, "PerCapitaConsumption.csv"), DataFrame(opt_model[:neteconomy, :CPC], :auto))
        save(joinpath(exp_dir, "GDP.csv"), DataFrame(opt_model[:neteconomy, :YNET], :auto))

        # Save starting point for reproducibility
        sp_dir = joinpath(exp_dir, "StartingPoints")
        mkpath(sp_dir)
        save(joinpath(sp_dir, "policy_vector.csv"), DataFrame(x=result[1]))
    end
end

#------------------------------------------------------------------------------------------------------
# Print results table
#------------------------------------------------------------------------------------------------------
println("\n\n" * "="^100)
println("RESULTS: Rich Region Emissions Share (%)")
println("="^100)

config_names = [c.name for c in configs]
header = rpad("Experiment", 18) * join([rpad(c, 14) for c in config_names])
println(header)
println("-"^length(header))

for (exp_name, _) in experiments
    shares = results_table[exp_name]
    row = rpad(exp_name, 18)
    for c in config_names
        r = get(shares, c, nothing)
        s = r !== nothing ? r.share_rich : NaN
        row *= rpad(isnan(s) ? "ERROR" : string(round(s, digits=3)), 14)
    end
    println(row)
end

# Contribution of each parameter group = full_model - minus_X
full = results_table["full_model"]
println("\n" * "-"^length(header))
println("Contribution of each parameter group (full - minus_X):")
println("-"^length(header))

for (exp_name, _) in experiments
    exp_name == "full_model" && continue
    shares = results_table[exp_name]
    row = rpad(exp_name, 18)
    for c in config_names
        r_full = get(full, c, nothing)
        r_exp = get(shares, c, nothing)
        s_full = r_full !== nothing ? r_full.share_rich : NaN
        s_exp = r_exp !== nothing ? r_exp.share_rich : NaN
        delta = s_full - s_exp
        row *= rpad(isnan(delta) ? "ERROR" : string(round(delta, digits=3), " pp"), 14)
    end
    println(row)
end
println("="^100)

# Utility comparison
println("\n" * "="^100)
println("UTILITY VALUES")
println("="^100)
println(rpad("Experiment", 18) * join([rpad(c, 14) for c in config_names]))
println("-"^(18 + 14*length(config_names)))
for (exp_name, _) in experiments
    shares = results_table[exp_name]
    row = rpad(exp_name, 18)
    for c in config_names
        r = get(shares, c, nothing)
        u = r !== nothing ? r.utility : NaN
        row *= rpad(isnan(u) ? "ERROR" : string(round(u, sigdigits=8)), 14)
    end
    println(row)
end
println("="^100)

# Save results to CSV
results_df = DataFrame(
    experiment = String[],
    config = String[],
    share_rich = Float64[],
    contribution_pp = Float64[],
    utility = Float64[],
    cca = Float64[],
    max_cost = Float64[],
    convergence = String[],
    elapsed_s = Float64[]
)

for (exp_name, _) in experiments
    for c in config_names
        r = get(results_table[exp_name], c, nothing)
        if r !== nothing
            r_full = get(full, c, nothing)
            s_full = r_full !== nothing ? r_full.share_rich : NaN
            contribution = s_full - r.share_rich
            push!(results_df, (exp_name, c, r.share_rich, contribution,
                               r.utility, r.cca, r.max_cost, r.convergence, r.elapsed))
        end
    end
end

csv_path = joinpath(results_base, "decomposition_results.csv")
save(csv_path, results_df)
println("\nResults saved to: $csv_path")
println("Finished at: $(now())")
