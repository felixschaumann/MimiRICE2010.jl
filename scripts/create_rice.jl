#-------------------------------------------------------------------------------
# This function creates an instance of MimiRICE2010, given user specifications.
#-------------------------------------------------------------------------------

# Load packages.
using Mimi
using DataFrames
using CSVFiles
using Main.MimiRICE2010

include(joinpath(@__DIR__, "..", "src", "new_components", "updated_welfare_rice.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_neteconomy_component.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_damages_component.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_welfare_component.jl"))

# Load necessary model and data files.
# un_population = DataFrame(load(joinpath(@__DIR__, "..", "data", "UN_population_rice_regions.csv"), skiplines_begin=3))
# t_opt_ad = DataFrame(load(joinpath(@__DIR__, "..", "results/MyResults/orig_var_ad_rice_utilitarian/Adaptation.csv"), skiplines_begin=0))[5, :]

# Create a function to construct an updated version of RICE2010.
function create_rice(ρ::Float64, η::Float64, remove_negishi::Bool, add_ad::Bool=false, stock_ad::Bool=false)

    # ---------------------------------------------
    # Create MimiRICE2010 model and set parameters.
    # ---------------------------------------------

    # Load base version of MimiRICE2010.
    m = MimiRICE2010.get_model()
    #include(joinpath(@__DIR__, "..", "src", "new_components\\Emissions_SlowDown_rice.jl"))
    #include(joinpath(@__DIR__, "..", "src", "new_components\\CatastrophicDamages.jl"))
    #replace_comp!(m, emissions, :emissions, reconnect=true) #unblock this for political feasibility run; need to lower initial guess in cost-min run
    #replace_comp!(m, damages, :damages, reconnect=true)  #unblock this for catastrophic damages output

    # Set savings rate to 25.8% for all regions and time periods.
    update_param!(m, :S, ones(60, 2) .* 0.258)

    # Set population to updated UN projections.
    # update_param!(m, :l, Matrix(un_population))

    if add_ad == true
        replace!(m, :damages => ad_damages)
        replace!(m, :neteconomy => ad_neteconomy)
        connect_param!(m, :neteconomy, :ADAPTCOST, :damages, :ADAPTCOST)
        
        replace!(m, :welfare=>ad_welfare, reconnect=true)
        # Connect cost constraint penalty (commented out for now - uncomment to activate)
        connect_param!(m, :welfare, :COST_CONSTRAINT_PENALTY, :neteconomy, :COST_CONSTRAINT_PENALTY)

        # Initialize cost constraint parameters (inactive by default)
        set_param!(m, :neteconomy, :COST_CAP_FRACTION, 0.003) # 0.3% of GDP - set to large value to effectively disable
        set_param!(m, :neteconomy, :CONSTRAINT_PENALTY_STRENGTH, 0) # Set to 0 to disable constraint (10000 works well)
        
        # Set adapted temperature to a linear increase from 0.5 degrees in 2010 to 0.5 degrees in 2100. And 0.5 degrees from 2100 to 2200.
        T_AD = zeros(60, 2)
        for t in 1:20
            T_AD[t, :] .= 0.5 + (t - 1) * (0.5 - 0.5) / 20
        end
        for t in 21:60
            T_AD[t, :] .= 0.5
        end
        set_param!(m, :damages, :T_AD, T_AD)
        set_param!(m, :damages, :ADAPTFRAC, ones(60, 2) .* 0.0) # initially set to zero adaptation
        set_param!(m, :damages, :OPT_T_SHIFT, 0.0) # set optimal temperature shift parameter
        
        # Set abatement and adaptation synergy factor (0 means no synergy)
        update_param!(m, :neteconomy, :AB_AD_SYN, 0.0)
        
        # Parameters required for stock adaptaptation
        set_param!(m, :damages, :T_AD_FLOW, T_AD)
        set_param!(m, :damages, :I_AD, ones(60, 2) .* 0.0)
        connect_param!(m, :damages, :dk,  :dk)

        if stock_ad == true
            set_param!(m, :damages, :HAS_STOCK_AD, true)
        end

    end

    # Replace welfare component if switching off Negishi weights and then set necessary parameters.
    if remove_negishi == true
        # We cannot use `reconnect` here because the new welfare component does not have all the same
        # parameters as the old one
        replace!(m, :welfare=>welfare, reconnect=false)

        # Set the welfare function parameters
        set_param!(m, :welfare, :ρ, ρ)
        set_param!(m, :welfare, :η, η)

        # Connect the population parameter in the new component to the existing
        # population model level parameter
        connect_param!(m, :welfare, :pop, :l)

        # Connect the welfare component with the net economy component
        connect_param!(m, :welfare, :cpc, :neteconomy, :CPC)

    end

    # Return user-specified model.
    return m
end
