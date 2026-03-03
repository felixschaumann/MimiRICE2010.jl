########################################################################################################
# Complete the remaining v2 decomposition runs: minus_negishi (config3f, config4f), minus_slr (all 4)
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
# Settings (same as main decomposition experiment)
#------------------------------------------------------------------------------------------------------
ρ = 0.008
η = 1.5
remove_negishi = false
n_opt_periods = 30
optimization_algorithm = :LN_AUGLAG
stop_time = 1200
tolerance = 1e-15
carbon_budget = 1022.5950757761052
cost_cap = 0.008

results_base = joinpath(@__DIR__, "../results", "Decomposition_v2")

configs = [
    (name="config1",  opt_ad=false, stock_ad=false, cbudget=nothing,       cost_cap=nothing),
    (name="config2f", opt_ad=true,  stock_ad=false, cbudget=nothing,       cost_cap=nothing),
    (name="config3f", opt_ad=true,  stock_ad=false, cbudget=nothing,       cost_cap=cost_cap),
    (name="config4f", opt_ad=true,  stock_ad=false, cbudget=carbon_budget, cost_cap=cost_cap),
]

function compute_share_rich(model)
    eind = model[:emissions, :EIND]
    total_rich = sum(eind[:, 1])
    total_poor = sum(eind[:, 2])
    return total_rich / (total_rich + total_poor) * 100.0
end

function make_partial_symmetric_constructor(symmetrize::Vector{Symbol})
    return (ρ, η, remove_negishi, opt_ad=false, stock_ad=false, cbudget=nothing) ->
        create_rice_partial_symmetric(ρ, η, remove_negishi, opt_ad, stock_ad, cbudget; symmetrize=symmetrize)
end

# Get backstop prices
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000

#------------------------------------------------------------------------------------------------------
# Define remaining runs
#------------------------------------------------------------------------------------------------------
remaining = [
    ("minus_negishi", [:alpha, :scale1, :scale2, :rr], ["config3f", "config4f"]),
    ("minus_slr",     [:slrmultiplier, :slrelasticity, :slrdamlinear, :slrdamquadratic], ["config1", "config2f", "config3f", "config4f"]),
]

println("\n" * "="^70)
println("COMPLETING DECOMPOSITION v2 — REMAINING RUNS")
println("="^70)

for (exp_name, symmetrize_params, config_names) in remaining
    println("\n" * "#"^70)
    println("EXPERIMENT: $exp_name  (symmetrize: $(join(symmetrize_params, ", ")))")
    println("#"^70)

    mc = make_partial_symmetric_constructor(symmetrize_params)

    # If we need warm-starting for config4f from config3f, track prev solution
    prev_solution = nothing
    prev_config_name = ""

    # If config3f is in the list and we need warm-start for config4f,
    # load config2f solution as starting point if available
    ext_sp_for_first = nothing
    first_cfg_name = config_names[1]

    for cfg_name in config_names
        cfg = configs[findfirst(c -> c.name == cfg_name, configs)]
        println("\n--- $exp_name / $(cfg.name) ---")

        exp_dir = joinpath(results_base, exp_name, cfg.name)
        mkpath(exp_dir)

        # Warm-start config4f from config3f solution
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
            continue
        end
        elapsed = round(time() - t_start, digits=1)

        opt_model = result[7]
        raw_utility = result[9]
        convergence = result[8]

        prev_solution = result[1]
        prev_config_name = cfg.name

        share_rich = compute_share_rich(opt_model)
        cca = opt_model[:emissions, :CCA][end]
        max_cost = cfg.opt_ad ? maximum(opt_model[:neteconomy, :TOTAL_COST]) : NaN

        println("  Utility:    $(round(raw_utility, sigdigits=8))")
        println("  Share rich: $(round(share_rich, digits=3))%")
        println("  CCA:        $(round(cca, digits=1)) GtC")
        if cfg.opt_ad
            println("  Max cost:   $(round(max_cost, sigdigits=4))")
        end
        println("  Convergence: $convergence  ($(elapsed)s)")

        # Save outputs
        save(joinpath(exp_dir, "Emissions.csv"), DataFrame(opt_model[:emissions, :EIND], :auto))
        save(joinpath(exp_dir, "MitigationRate.csv"), DataFrame(result[3], :auto))
        if cfg.opt_ad
            save(joinpath(exp_dir, "FlowAdaptation.csv"), DataFrame(result[4], :auto))
        end
        save(joinpath(exp_dir, "Temperature.csv"), DataFrame(temp=opt_model[:climatedynamics, :TATM]))
        save(joinpath(exp_dir, "PerCapitaConsumption.csv"), DataFrame(opt_model[:neteconomy, :CPC], :auto))
        save(joinpath(exp_dir, "GDP.csv"), DataFrame(opt_model[:neteconomy, :YNET], :auto))

        sp_dir = joinpath(exp_dir, "StartingPoints")
        mkpath(sp_dir)
        save(joinpath(sp_dir, "policy_vector.csv"), DataFrame(x=result[1]))
    end
end

println("\n" * "="^70)
println("Completion runs finished at: $(now())")
println("="^70)
