#-------------------------------------------------------------------------------
# This function creates an instance of MimiRICE2010, given user specifications.
#-------------------------------------------------------------------------------

# Load packages.
using Mimi
using DataFrames
using CSVFiles
using Statistics
include(joinpath(@__DIR__, "..", "src", "MimiRICE2010.jl"))
using Main.MimiRICE2010

include(joinpath(@__DIR__, "..", "src", "new_components", "updated_welfare_rice.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_neteconomy_component.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_damages_component.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_welfare_component.jl"))
include(joinpath(@__DIR__, "..", "src", "new_components", "ad_emissions_component.jl"))

# Load necessary model and data files.
# un_population = DataFrame(load(joinpath(@__DIR__, "..", "data", "UN_population_rice_regions.csv"), skiplines_begin=3))
# t_opt_ad = DataFrame(load(joinpath(@__DIR__, "..", "results/MyResults/orig_var_ad_rice_utilitarian/Adaptation.csv"), skiplines_begin=0))[5, :]

"""
    compute_negishi_weights(m)

Run the model at BAU and recompute Negishi welfare weights (α) so that
weighted marginal utilities are equalized across regions in every period.

The Negishi condition requires:
    scale1[r] * α[t,r] * l[t,r] * rr[t,r] * CPC[t,r]^(-η) = const  ∀ r

Solving for α:
    α[t,r] ∝ 1 / (scale1[r] * l[t,r] * rr[t,r] * CPC[t,r]^(-η))

Normalized so that the geometric mean across regions equals 1 each period.
"""
function compute_negishi_weights(m)
    run(m)
    CPC    = m[:neteconomy, :CPC]      # T×R
    l      = m[:welfare, :l]            # T×R
    rr     = m[:welfare, :rr]           # T×R
    scale1 = m[:welfare, :scale1]       # R-vector
    elasmu = m[:welfare, :elasmu]       # R-vector
    η = elasmu[1]

    T, R = size(CPC)
    new_alpha = zeros(T, R)
    for t in 1:T
        raw = [1.0 / (scale1[r] * l[t,r] * rr[t,r] * CPC[t,r]^(-η)) for r in 1:R]
        geo_mean = prod(raw)^(1/R)
        new_alpha[t,:] = raw ./ geo_mean   # geometric mean = 1 each period
    end
    return new_alpha
end


# Create a function to construct an updated version of RICE2010.
function create_rice(ρ::Float64, η::Float64, remove_negishi::Bool, opt_ad::Bool=false, stock_ad::Bool=false, cbudget=nothing)

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

    if isa(cbudget, Float64)
        replace!(m, :emissions => ad_emissions)
        replace!(m, :welfare=>ad_welfare, reconnect=true)
        connect_param!(m, :welfare, :CARBON_CONSTRAINT_PENALTY, :emissions, :CARBON_CONSTRAINT_PENALTY)

        # Set carbon budget
        set_param!(m, :emissions, :CBUDGET, cbudget) # Set cumulative global carbon budget (GtC)
        set_param!(m, :emissions, :CONSTRAINT_PENALTY_STRENGTH, 0) # Set to 0 to disable constraint (10000 works well)
    end    

    if opt_ad == true
        replace!(m, :damages => ad_damages)
        replace!(m, :neteconomy => ad_neteconomy)
        connect_param!(m, :neteconomy, :ADAPTCOST, :damages, :ADAPTCOST)
        
        # replace!(m, :welfare=>ad_welfare, reconnect=true)
        # Connect constraint penalties
        # connect_param!(m, :welfare, :COST_CONSTRAINT_PENALTY, :neteconomy, :COST_CONSTRAINT_PENALTY)
        # connect_param!(m, :welfare, :CARBON_CONSTRAINT_PENALTY, :emissions, :CARBON_CONSTRAINT_PENALTY)

        # Initialize cost constraint parameters (inactive by default)
        set_param!(m, :neteconomy, :COST_CAP_FRACTION, 0.003) # 0.3% of GDP - set to large value to effectively disable
        if !isa(cbudget, Float64)
            # Only set if not already created by the emissions block (shared model-level parameter)
            set_param!(m, :neteconomy, :CONSTRAINT_PENALTY_STRENGTH, 0) # Set to 0 to disable constraint (10000 works well)
        end
        
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
        else
            set_param!(m, :damages, :HAS_STOCK_AD, false)
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

    # Recalibrate Negishi weights from BAU per-capita consumption.
    if remove_negishi == false
        new_alpha = compute_negishi_weights(m)
        update_param!(m, :alpha, new_alpha)
    end

    # Return user-specified model.
    return m
end


#-------------------------------------------------------------------------------
# Create a symmetric version of RICE where both regions have identical parameters.
# `overrides` is a Dict mapping parameter symbols to their original (asymmetric)
# values. Parameters listed in overrides are NOT symmetrized.
#-------------------------------------------------------------------------------

function create_rice_symmetric(ρ::Float64, η::Float64, remove_negishi::Bool, opt_ad::Bool=false, stock_ad::Bool=false, cbudget=nothing;
                                overrides=Dict{Symbol,Any}())

    # Read raw 2-region parameters from Excel
    datafile = joinpath(@__DIR__, "..", "data", "RICE_2010_base_000.xlsm")
    params = MimiRICE2010.getrice2010parameters(datafile)

    # --- Symmetrize region-varying parameters ---

    # Extensive params (originally summed across regions): each region gets half the total
    extensive = Set([:l, :k0])

    # Single-value per region (1D vectors of length 2)
    region_params_single = [:k0, :dk, :elasmu, :expcost2, :a1, :a2, :a3,
                            :slrmultiplier, :slrelasticity, :slrdamlinear, :slrdamquadratic,
                            :scale1, :scale2]

    # Time series per region (60×2 matrices)
    region_params_ts = [:l, :al, :sigma, :pbacktime, :cost1, :rr, :alpha, :S, :MIU]

    for param in region_params_single
        haskey(params, param) || continue
        haskey(overrides, param) && continue
        val = params[param]
        if param in extensive
            params[param] = fill(sum(val) / length(val), length(val))
        else
            params[param] = fill(mean(val), length(val))
        end
    end

    for param in region_params_ts
        haskey(params, param) || continue
        haskey(overrides, param) && continue
        val = params[param]
        n = size(val, 2)
        if param in extensive
            params[param] = repeat(sum(val, dims=2) ./ n, 1, n)
        else
            params[param] = repeat(mean(val, dims=2), 1, n)
        end
    end

    # Apply overrides (restore original asymmetric values for selected params)
    for (param, val) in overrides
        params[param] = val
    end

    # Build model from (symmetrized + overridden) parameters
    m = MimiRICE2010.constructrice(params)

    # --- Continue with same setup as create_rice() ---

    # Set savings rate to 25.8%
    update_param!(m, :S, ones(60, 2) .* 0.258)

    if isa(cbudget, Float64)
        replace!(m, :emissions => ad_emissions)
        replace!(m, :welfare=>ad_welfare, reconnect=true)
        connect_param!(m, :welfare, :CARBON_CONSTRAINT_PENALTY, :emissions, :CARBON_CONSTRAINT_PENALTY)
        set_param!(m, :emissions, :CBUDGET, cbudget)
        set_param!(m, :emissions, :CONSTRAINT_PENALTY_STRENGTH, 0)
    end

    if opt_ad == true
        replace!(m, :damages => ad_damages)
        replace!(m, :neteconomy => ad_neteconomy)
        connect_param!(m, :neteconomy, :ADAPTCOST, :damages, :ADAPTCOST)

        set_param!(m, :neteconomy, :COST_CAP_FRACTION, 0.003)
        if !isa(cbudget, Float64)
            set_param!(m, :neteconomy, :CONSTRAINT_PENALTY_STRENGTH, 0)
        end

        T_AD = zeros(60, 2)
        for t in 1:20
            T_AD[t, :] .= 0.5 + (t - 1) * (0.5 - 0.5) / 20
        end
        for t in 21:60
            T_AD[t, :] .= 0.5
        end
        set_param!(m, :damages, :T_AD, T_AD)
        set_param!(m, :damages, :ADAPTFRAC, ones(60, 2) .* 0.0)
        set_param!(m, :damages, :OPT_T_SHIFT, 0.0)

        update_param!(m, :neteconomy, :AB_AD_SYN, 0.0)

        set_param!(m, :damages, :T_AD_FLOW, T_AD)
        set_param!(m, :damages, :I_AD, ones(60, 2) .* 0.0)
        connect_param!(m, :damages, :dk, :dk)

        if stock_ad == true
            set_param!(m, :damages, :HAS_STOCK_AD, true)
        else
            set_param!(m, :damages, :HAS_STOCK_AD, false)
        end
    end

    if remove_negishi == true
        replace!(m, :welfare=>welfare, reconnect=false)
        set_param!(m, :welfare, :ρ, ρ)
        set_param!(m, :welfare, :η, η)
        connect_param!(m, :welfare, :pop, :l)
        connect_param!(m, :welfare, :cpc, :neteconomy, :CPC)
    end

    # Recalibrate Negishi weights from BAU per-capita consumption.
    if remove_negishi == false
        new_alpha = compute_negishi_weights(m)
        update_param!(m, :alpha, new_alpha)
    end

    return m
end


#-------------------------------------------------------------------------------
# Create a RICE model where only specified parameters are symmetrized.
# Starts from the full asymmetric model and averages only the listed params.
# Use for "full minus one" decomposition: symmetrize one group to measure
# its contribution to the allocation result.
#-------------------------------------------------------------------------------

function create_rice_partial_symmetric(ρ::Float64, η::Float64, remove_negishi::Bool, opt_ad::Bool=false, stock_ad::Bool=false, cbudget=nothing;
                                        symmetrize=Symbol[])

    # Read raw 2-region parameters from Excel
    datafile = joinpath(@__DIR__, "..", "data", "RICE_2010_base_000.xlsm")
    params = MimiRICE2010.getrice2010parameters(datafile)

    # Extensive params (originally summed): each region gets half the total
    extensive = Set([:l, :k0])

    # All known region-varying params and their shapes
    region_params_single = Set([:k0, :dk, :elasmu, :expcost2, :a1, :a2, :a3,
                                :slrmultiplier, :slrelasticity, :slrdamlinear, :slrdamquadratic,
                                :scale1, :scale2])
    region_params_ts = Set([:l, :al, :sigma, :pbacktime, :cost1, :rr, :alpha, :S, :MIU])

    for param in symmetrize
        haskey(params, param) || continue
        val = params[param]

        if param in region_params_single
            if param in extensive
                params[param] = fill(sum(val) / length(val), length(val))
            else
                params[param] = fill(mean(val), length(val))
            end
        elseif param in region_params_ts
            n = size(val, 2)
            if param in extensive
                params[param] = repeat(sum(val, dims=2) ./ n, 1, n)
            else
                params[param] = repeat(mean(val, dims=2), 1, n)
            end
        else
            @warn "Parameter $param not recognized as region-varying, skipping"
        end
    end

    # Build model from modified parameters
    m = MimiRICE2010.constructrice(params)

    # --- Same setup as create_rice() ---

    update_param!(m, :S, ones(60, 2) .* 0.258)

    if isa(cbudget, Float64)
        replace!(m, :emissions => ad_emissions)
        replace!(m, :welfare=>ad_welfare, reconnect=true)
        connect_param!(m, :welfare, :CARBON_CONSTRAINT_PENALTY, :emissions, :CARBON_CONSTRAINT_PENALTY)
        set_param!(m, :emissions, :CBUDGET, cbudget)
        set_param!(m, :emissions, :CONSTRAINT_PENALTY_STRENGTH, 0)
    end

    if opt_ad == true
        replace!(m, :damages => ad_damages)
        replace!(m, :neteconomy => ad_neteconomy)
        connect_param!(m, :neteconomy, :ADAPTCOST, :damages, :ADAPTCOST)

        set_param!(m, :neteconomy, :COST_CAP_FRACTION, 0.003)
        if !isa(cbudget, Float64)
            set_param!(m, :neteconomy, :CONSTRAINT_PENALTY_STRENGTH, 0)
        end

        T_AD = zeros(60, 2)
        for t in 1:20
            T_AD[t, :] .= 0.5 + (t - 1) * (0.5 - 0.5) / 20
        end
        for t in 21:60
            T_AD[t, :] .= 0.5
        end
        set_param!(m, :damages, :T_AD, T_AD)
        set_param!(m, :damages, :ADAPTFRAC, ones(60, 2) .* 0.0)
        set_param!(m, :damages, :OPT_T_SHIFT, 0.0)

        update_param!(m, :neteconomy, :AB_AD_SYN, 0.0)

        set_param!(m, :damages, :T_AD_FLOW, T_AD)
        set_param!(m, :damages, :I_AD, ones(60, 2) .* 0.0)
        connect_param!(m, :damages, :dk, :dk)

        set_param!(m, :damages, :HAS_STOCK_AD, stock_ad == true)
    end

    if remove_negishi == true
        replace!(m, :welfare=>welfare, reconnect=false)
        set_param!(m, :welfare, :ρ, ρ)
        set_param!(m, :welfare, :η, η)
        connect_param!(m, :welfare, :pop, :l)
        connect_param!(m, :welfare, :cpc, :neteconomy, :CPC)
    end

    # Recalibrate Negishi weights from BAU per-capita consumption.
    if remove_negishi == false
        new_alpha = compute_negishi_weights(m)
        update_param!(m, :alpha, new_alpha)
    end

    return m
end


"""
    get_original_params()

Read the original (asymmetric) 2-region parameters from the RICE Excel file.
"""
function get_original_params()
    datafile = joinpath(@__DIR__, "..", "data", "RICE_2010_base_000.xlsm")
    return MimiRICE2010.getrice2010parameters(datafile)
end
