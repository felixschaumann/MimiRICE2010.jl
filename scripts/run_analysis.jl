########################################################################################################
# This file runs a number of model analyses from Budolfson et al. (2019) given user-specified settings.
########################################################################################################

# Activate the project for the paper and make sure all packages we need
# are installed.
using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()

# Load required Julia packages.
using NLopt
using CSVFiles

# Load required files.
include("create_rice.jl")
include(joinpath(@__DIR__, "../src/", "optimisation_functions.jl"))

#%%
#------------------------------------------------------------------------------------------------------
# Model Parameters To Modify.
#------------------------------------------------------------------------------------------------------

# Pure rate of time preference.
ρ = 0.008

# Elasticity of marginal utility of consumption.
η = 1.5

# Should the RICE model be run with a social welfare function that uses Negish weights (true = remove Negishi weights).
# *Note: if using Negishi weights, the model uses the default RICE settings of η=1.5 and ρ=1.5%.
remove_negishi = false

#------------------------------------------------------------------------------------------------------
# Choices For Model Versions To Run.
#------------------------------------------------------------------------------------------------------

# Run an optiization with RICE using global carbon prices?
rice_cost_minimization = false

# Run an optization with RICE using regional carbon prices?
rice_utilitarian = true

# Run an optiization with RICE using global carbon prices?
fund_cost_minimization = false

# Run an optization with RICE using regional carbon prices?
fund_utilitarian = false

#------------------------------------------------------------------------------------------------------
# Optimization Settings to Modify.
#------------------------------------------------------------------------------------------------------

# Name of folder to save this set of model runs in (a folder will be created with this name).
results_folder = "MyResults"

# Number of model periods to optimize over for RICE (after which model assumes full decarbonization).
# NOTE: FUND does not have a backstop price and optimizes from 2010-2200 by default, and then assumes a constant carbon tax.
n_opt_periods = 30

# Optimization algorithm (the type should be a Symbol, e.g. :LN_SBPLX). See options at http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
optimization_algorithm = :LN_COBYLA # :LN_SBPLX

# Maximum time in seconds to run each model (NOTE: FUND takes much longer to optimize than RICE).
stop_time_rice = 7200
# stop_time_fund = 7500

# Relative tolerance criteria for convergence (will stop if |Δf| / |f| < tolerance from one iteration to the next.)
tolerance_rice = 1e-10
# tolerance_fund = 1e-10

#%%
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Run Everything and Save Key Results (*no need to modify code below this line).
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
println("Starting Analysis.")
println()

# Load and run an instance of RICE just to extract the backstop prices (needed for multiple analyses).
backstop_rice = create_rice(ρ, η, remove_negishi)
run(backstop_rice)
backstop_prices = backstop_rice[:emissions, :pbacktime] .* 1000
output_directory = joinpath(@__DIR__, "../", "results", results_folder, "rice_BAU")
mkpath(output_directory)
save(joinpath(output_directory, "Emissions.csv"), DataFrame(backstop_rice[:emissions, :EIND], :auto))
save(joinpath(output_directory, "PerCapitaConsumption.csv"), DataFrame(backstop_rice[:neteconomy, :CPC], :auto))
save(joinpath(output_directory, "Population.csv"), DataFrame(backstop_rice[:welfare, :l], :auto))
# LandUse = load(joinpath(@__DIR__, "../", "data", "RICELandUse.csv")) |> DataFrame
# save(joinpath(output_directory, "LandUse.csv"), LandUse)

#%%
#------------------------------------------------------------------------------------------------------
# Run RICE Cost-minimization Optimization.
#------------------------------------------------------------------------------------------------------
if rice_cost_minimization == true

    println("Starting RICE cost-minimization optimization...")

    # Optimize model.
    opt_output_rice_cost_minimization, opt_emissions_rice_cost_minimization, opt_mitigation_rice_cost_minimization, opt_tax_rice_cost_minimization, opt_model_rice_cost_minimization, convergence_rice_cost_minimization = optimize_rice(optimization_algorithm, n_opt_periods, stop_time_rice, tolerance_rice, backstop_prices, run_utilitarian=false, ρ=ρ, η=η, remove_negishi=remove_negishi)

    # Create folder to store some key results.
    output_directory = joinpath(@__DIR__, "../", "results", results_folder, "rice_costmin")
    mkpath(output_directory)

    # Save optimal CO₂ mitigation, carbon tax, per capita consumption, and global temperature anomaly.
    save(joinpath(output_directory, "Emissions.csv"), DataFrame(opt_emissions_rice_cost_minimization, :auto))
    save(joinpath(output_directory, "MitigationRate.csv"), DataFrame(opt_mitigation_rice_cost_minimization, :auto))
    save(joinpath(output_directory, "Carbon Tax.csv"), DataFrame(global_carbon_tax=opt_tax_rice_cost_minimization))
    save(joinpath(output_directory, "Temperature.csv"), DataFrame(temp=opt_model_rice_cost_minimization[:climatedynamics, :TATM]))
    save(joinpath(output_directory, "PerCapitaConsumption.csv"), DataFrame(opt_model_rice_cost_minimization[:neteconomy, :CPC], :auto))
    save(joinpath(output_directory, "GDP.csv"), DataFrame(opt_model_rice_cost_minimization[:neteconomy, :YNET], :auto))

    println("Optimization convergence result: ", convergence_rice_cost_minimization)
    println("RICE cost-minimization optimization complete.")
    println()
end

#%%
#------------------------------------------------------------------------------------------------------
# Run RICE Utilitarian Optimization.
#------------------------------------------------------------------------------------------------------

opt_ad = true # whether to optimise (endogenise) adaptation or have it exogenous/fixed
stock_ad = true # whether to have stock adaptation (true) or flow adaptation (false)
ad_string = opt_ad ? "ad_" : ""
if opt_ad == true
    ad_string = "var_ad_"
    if stock_ad == true
        ad_string = "stock_ad_"
    end
end

carbon_budget = 1022.5950757761052 # 921.0
cost_cap = 0.1 # 0.005 # 0.003

# Load external starting points if available (set to nothing to use defaults)
use_ext_starting_points = true
ext_starting_points = nothing

if use_ext_starting_points
    starting_points_dir = joinpath(@__DIR__, "../", "results", results_folder, "$(ad_string)rice_utilitarian"*(remove_negishi ? "_no_negishi" : ""), "StartingPoints")

    if isdir(starting_points_dir)
        println("Loading starting points from: $starting_points_dir")

        # Load mitigation starting points (rows 2:n_opt_periods+1, all columns)
        mit_df = DataFrame(load(joinpath(starting_points_dir, "opt_mitigation_rice_utilitarian.csv")))
        mit_matrix = Matrix(mit_df)[2:(n_opt_periods+1), :]
        mit_vec = vec(mit_matrix)  # Flatten column-major

        if opt_ad && stock_ad
            # Load flow and stock adaptation
            flow_df = DataFrame(load(joinpath(starting_points_dir, "opt_flow_adaptation_rice_utilitarian.csv")))
            flow_matrix = Matrix(flow_df)[2:(n_opt_periods+1), :]
            flow_vec = vec(flow_matrix)

            stock_df = DataFrame(load(joinpath(starting_points_dir, "opt_stock_adaptation_rice_utilitarian.csv")))
            stock_matrix = Matrix(stock_df)[2:(n_opt_periods+1), :]
            stock_vec = vec(stock_matrix)

            ext_starting_points = vcat(mit_vec, flow_vec, stock_vec)
        elseif opt_ad
            # Load flow adaptation only
            flow_df = DataFrame(load(joinpath(starting_points_dir, "opt_flow_adaptation_rice_utilitarian.csv")))
            flow_matrix = Matrix(flow_df)[2:(n_opt_periods+1), :]
            flow_vec = vec(flow_matrix)

            ext_starting_points = vcat(mit_vec, flow_vec)
        else
            ext_starting_points = mit_vec
        end

        println("Loaded starting points vector of length $(length(ext_starting_points))")
    else
        println("Starting points directory not found, using defaults")
    end
end

if rice_utilitarian == true

    println("Starting $(ad_string)RICE utilitarian optimization...")

    # Optimize model.
    @time opt_output_rice_utilitarian, opt_emissions_rice_utilitarian, opt_mitigation_rice_utilitarian, opt_flow_adaptation_rice_utilitarian, opt_stock_adaptation_rice_utilitarian, opt_tax_rice_utilitarian, opt_model_rice_utilitarian, convergence_rice_utilitarian = optimize_rice(optimization_algorithm, n_opt_periods, stop_time_rice, tolerance_rice, backstop_prices, run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi, opt_ad=opt_ad, stock_ad=stock_ad, cbudget=carbon_budget, cost_cap=cost_cap, ext_starting_points=ext_starting_points)

    # Create folder to store some key results.
    output_directory = joinpath(@__DIR__, "../", "results", results_folder, "$(ad_string)rice_utilitarian"*(remove_negishi ? "_no_negishi" : ""))
    mkpath(output_directory)

    # Save optimal CO₂ mitigation, adaptation, carbon tax, per capita consumption, and global temperature anomaly.
    save(joinpath(output_directory, "MitigationRate.csv"), DataFrame(opt_mitigation_rice_utilitarian, :auto))
    if opt_ad == true
        if stock_ad == true
            save(joinpath(output_directory, "FlowAdaptation.csv"), DataFrame(opt_flow_adaptation_rice_utilitarian, :auto))
            save(joinpath(output_directory, "StockAdaptation.csv"), DataFrame(opt_stock_adaptation_rice_utilitarian, :auto))
        end
        
    end
    save(joinpath(output_directory, "Emissions.csv"), DataFrame(opt_emissions_rice_utilitarian, :auto))
    save(joinpath(output_directory, "Carbon Tax.csv"), DataFrame(opt_tax_rice_utilitarian, :auto))
    save(joinpath(output_directory, "Temperature.csv"), DataFrame(temp=opt_model_rice_utilitarian[:climatedynamics, :TATM]))
    save(joinpath(output_directory, "GDP.csv"), DataFrame(opt_model_rice_utilitarian[:neteconomy, :YNET], :auto))
    save(joinpath(output_directory, "PerCapitaConsumption.csv"), DataFrame(opt_model_rice_utilitarian[:neteconomy, :CPC], :auto))

    println("Optimization convergence result: ", convergence_rice_utilitarian)
    println("RICE utilitarian optimization complete.")
    println()
end

#------------------------------------------------------------------------------------------------------
# End of Analysis.
#------------------------------------------------------------------------------------------------------
println("Analysis complete.")  

#%%

# write optimised chioce variables to file as starting points for future runs
CSV.write(joinpath(@__DIR__, "../", "results", results_folder, "$(ad_string)rice_utilitarian"*(remove_negishi ? "_no_negishi" : ""), "StartingPoints", "opt_mitigation_rice_utilitarian.csv"), DataFrame(opt_mitigation_rice_utilitarian, :auto))
if opt_ad == true
    CSV.write(joinpath(@__DIR__, "../", "results", results_folder, "$(ad_string)rice_utilitarian"*(remove_negishi ? "_no_negishi" : ""), "StartingPoints", "opt_flow_adaptation_rice_utilitarian.csv"), DataFrame(opt_flow_adaptation_rice_utilitarian, :auto))
    if stock_ad == true
        CSV.write(joinpath(@__DIR__, "../", "results", results_folder, "$(ad_string)rice_utilitarian"*(remove_negishi ? "_no_negishi" : ""), "StartingPoints", "opt_stock_adaptation_rice_utilitarian.csv"), DataFrame(opt_stock_adaptation_rice_utilitarian, :auto))
    end
end