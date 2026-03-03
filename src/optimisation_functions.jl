# #----------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------------
# This file contains functions and other snippets of code that are used in various calculations.
# #----------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------------

# From https://github.com/Environment-Research/Utilitarianism/blob/master/src/helper_functions.jl


#######################################################################################################################
# PARSE SOLUTION VECTOR INTO FULL PARAMETER MATRICES.
########################################################################################################################
# Description: Reshapes a flat optimization vector into full 60×n_regions parameter matrices
#              for mitigation, flow adaptation, and stock adaptation. Eliminates duplicated
#              reshape logic between evaluate_model! and post-optimization parsing.
#----------------------------------------------------------------------------------------------------------------------

function parse_solution_vector(x::Vector{Float64}, n_opt_periods::Int, n_regions::Int,
                               opt_ad::Bool, stock_ad::Bool;
                               stock_ad_default::Float64=0.5)
    if opt_ad
        if stock_ad
            mitigation_vec = x[1:(n_opt_periods*n_regions)]
            flow_adaptation_vec = x[(n_opt_periods*n_regions+1):(n_opt_periods*n_regions*2)]
            stock_adaptation_vec = x[(n_opt_periods*n_regions*2+1):end]

            optimal_mitigation = vcat(zeros(1, n_regions), ones(59, n_regions))
            optimal_mitigation[2:(n_opt_periods+1), :] = reshape(mitigation_vec, (n_opt_periods, n_regions))

            optimal_flow_adaptation = vcat(zeros(1, n_regions), ones(59, n_regions) .* 0.5)
            optimal_flow_adaptation[2:(n_opt_periods+1), :] = reshape(flow_adaptation_vec, (n_opt_periods, n_regions))

            optimal_stock_adaptation = vcat(zeros(1, n_regions), ones(59, n_regions) .* stock_ad_default)
            optimal_stock_adaptation[2:(n_opt_periods+1), :] = reshape(stock_adaptation_vec, (n_opt_periods, n_regions))

            return optimal_mitigation, optimal_flow_adaptation, optimal_stock_adaptation
        else
            mitigation_vec = x[1:(n_opt_periods*n_regions)]
            adaptation_vec = x[(n_opt_periods*n_regions+1):end]

            optimal_mitigation = vcat(zeros(1, n_regions), ones(59, n_regions))
            optimal_mitigation[2:(n_opt_periods+1), :] = reshape(mitigation_vec, (n_opt_periods, n_regions))

            optimal_flow_adaptation = vcat(zeros(1, n_regions), ones(59, n_regions) .* 0.5)
            optimal_flow_adaptation[2:(n_opt_periods+1), :] = reshape(adaptation_vec, (n_opt_periods, n_regions))

            return optimal_mitigation, optimal_flow_adaptation, nothing
        end
    else
        optimal_mitigation = vcat(zeros(1, n_regions), ones(59, n_regions))
        optimal_mitigation[2:(n_opt_periods+1), :] = reshape(x, (n_opt_periods, n_regions))

        return optimal_mitigation, nothing, nothing
    end
end

#######################################################################################################################
# CALCULATE REGIONAL CO₂ MITIGATION.
########################################################################################################################
# Description: This function calculates regional CO₂ mitigation levels as a function of a global carbon tax. It
#              uses the RICE2010 backstop price values and assumes a carbon tax of $0 in period 1.  If the number of
#              tax values is less than the total number of model time periods, the function assumes full decarbonization
#              (e.g. the tax = the backstop price) for all future periods without a specified tax.
#
# Function Arguments:
#
#       optimal_tax:    A vector of global carbon tax values to be optimized.
#       backstop_price: The regional backstop prices from RICE2010 (units must be in $1000s)
#       theta2:         The exponent on the abatement cost function (defaults to RICE2010 value).
#
# Function Output:
#
#       mitigation:     Regional mitigation rates resulting from the global carbon tax.
#----------------------------------------------------------------------------------------------------------------------

function mitigation_from_tax(optimal_tax::Array{Float64,1}, backstop_prices::Array{Float64,2}, theta2::Float64)

    # Initialize full tax vector with $0 tax in period 1 and maximum of the backstop price across all regions for remaining periods.
    full_tax = [0.0; maximum(backstop_prices, dims=2)[2:end]]

    # Set the periods being optimized to the optimized tax value (assuming full decarbonization for periods after optimization time frame).
    full_tax[2:(length(optimal_tax)+1)] = optimal_tax

    # Calculate regional mitigation rates from the full tax vector.
    mitigation = min.((max.(((full_tax ./ backstop_prices) .^ (1 / (theta2 - 1.0))), 0.0)), 1.0)

    return mitigation
end



#######################################################################################################################
# CREATE RICE OBJECTIVE FUNCTION.
########################################################################################################################
# Description: This function creates an objective function and an instance of RICE with user-specified parameter settings.
#              The objective function will take in a vector of global carbon tax values (cost-minimization) or regional
#              CO₂ mitigation rates (utilitarian) and returns the total economic welfare generated by that specifc
#              climate policy.
#
# Function Arguments:
#
#       run_utilitarian: A true/false indicator for whether or not to run the utilitarian optimization (true = run utilitarian).
#       ρ:                  Pure rate of time preference.
#       η:                  Elasticity of marginal utility of consumption.
#       backstop_prices:    The regional backstop prices from RICE2010 (units must be in dollars).
#       remove_negishi:     A true/false indicator for whether RICE should use a social welfare function with Negishi weights (true = remove Negishi weights).
#                           *Note: if using Negishi weights, RICE's discounting parameters default to η=1.5 and ρ=1.5%.
#
# Function Output:
#
#       rice_objective:     The objective function specific to user model settings.
#       m:                  An instance of RICE2010 consistent with user model settings.
#----------------------------------------------------------------------------------------------------------------------

function construct_rice_objective(run_utilitarian::Bool, ρ::Float64, η::Float64, backstop_prices::Array{Float64,2}, remove_negishi::Bool, opt_ad::Bool=false, stock_ad::Bool=false, cbudget=nothing, cost_cap=nothing; model_constructor=create_rice, alpha_override=nothing)

    # Get an instance of RICE given user settings.
    m = model_constructor(ρ, η, remove_negishi, opt_ad, stock_ad, cbudget; alpha_override=alpha_override)

    n_regions = length(m.md.dim_dict[:regions])

    #--------------------------------------------------------------------------------------------------------
    # Create either a (i) cost-minimization or (ii) utilitarian objective function for this instance of RICE.
    #--------------------------------------------------------------------------------------------------------

    rice_objective = nothing
    rice_constraint = nothing
    rice_cost_constraint = nothing

    if run_utilitarian == false

        #---------------------------------------
        # Cost-minimzation price objective function.
        #---------------------------------------
        rice_objective = function(optimal_global_tax::Array{Float64,1})
            # Set the regional mitigation rates to the value implied by the global optimal carbon tax and return total welfare.
            update_param!(m, :MIU, mitigation_from_tax(optimal_global_tax, backstop_prices, 2.8))
            run(m)
            return m[:welfare, :UTILITY]
        end

    else
        # Create shared cache for objective and constraint evaluation
        # This ensures the model runs only once per unique parameter vector
        cache_x = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        cache_utility = Ref(0.0)
        cache_cca = Ref(0.0)
        cache_max_cost = Ref(0.0)  # Maximum TOTAL_COST across all (time, region)

        # Shared evaluation function that updates cache if needed
        function evaluate_model!(x::Array{Float64,1})
            # Check if we need to run the model (x has changed)
            if cache_x[] === nothing || cache_x[] != x
                # Parse solution vector into full parameter matrices
                n_controls = opt_ad ? (stock_ad ? 3 : 2) : 1
                n_opt_periods = Int(length(x) / (n_regions * n_controls))
                optimal_regional_mitigation, optimal_regional_flow_adaptation, optimal_regional_stock_adaptation = parse_solution_vector(x, n_opt_periods, n_regions, opt_ad, stock_ad; stock_ad_default=0.25)

                update_param!(m, :MIU, optimal_regional_mitigation)
                if opt_ad && stock_ad
                    update_param!(m, :T_AD_FLOW, optimal_regional_flow_adaptation)
                    update_param!(m, :I_AD, optimal_regional_stock_adaptation)
                elseif opt_ad
                    update_param!(m, :T_AD, 2 .* optimal_regional_flow_adaptation)
                end

                # Run the model
                run(m)

                # Update cache
                cache_x[] = copy(x)
                cache_utility[] = m[:welfare, :UTILITY]
                cache_cca[] = m[:emissions, :CCA][end]

                # Cache max cost across all (time, region)
                # Note: log-sum-exp smooth max was tried here but introduces massive bias
                # when most costs are near zero (bias ≈ log(n_elements)/α ≈ 0.01, larger than typical max cost).
                # Plain maximum() works fine with SBPLX (designed for non-smooth functions).
                cache_max_cost[] = maximum(m[:neteconomy, :TOTAL_COST])
            end
        end

        # Create objective function that uses cached evaluation.
        # Include a fixed quadratic penalty for constraint violations directly in the objective.
        # This supplements AUGLAG's constraint handling which struggles with derivative-free solvers.
        # Penalty is calibrated to the utility scale: for a 20% cost cap violation, the penalty
        # should be comparable to the utility gap between constrained and unconstrained optima (~15 units).
        # With penalty_weight=500 and relative_violation=0.2: penalty = 500*0.04 = 20 ✓
        eval_counter = Ref(0)
        rice_objective = function(x::Array{Float64,1})
            evaluate_model!(x)
            eval_counter[] += 1

            obj_penalty = 0.0
            if cost_cap !== nothing
                cost_violation = max(0.0, cache_max_cost[] - cost_cap)
                if cost_violation > 1e-6
                    rel_violation = cost_violation / cost_cap
                    obj_penalty += 500.0 * rel_violation^2
                end
            end

            if cbudget !== nothing
                cca_violation = max(0.0, cache_cca[] - cbudget)
                if cca_violation > 0.1
                    rel_cca_violation = cca_violation / cbudget
                    obj_penalty += 500.0 * rel_cca_violation^2
                end
            end

            if eval_counter[] % 500 == 1
                println("Eval $(eval_counter[]): Utility = $(cache_utility[]), CCA = $(cache_cca[]), Max Cost = $(cache_max_cost[]), Penalty = $(round(obj_penalty, sigdigits=4))")
                flush(stdout)
            end
            return cache_utility[] - obj_penalty
        end

        # Create carbon budget constraint function if cbudget is provided
        if cbudget !== nothing
            rice_constraint = function(x::Array{Float64,1}, grad::Vector{Float64})
                evaluate_model!(x)
                # NLopt expects constraint of form f(x) <= 0
                constraint_value = cache_cca[] - cbudget
                return constraint_value  # Should be <= 0 (satisfied if CCA <= budget)
            end
        end

        # Create cost cap constraint function if cost_cap is provided
        if cost_cap !== nothing #&& opt_ad
            rice_cost_constraint = function(x::Array{Float64,1}, grad::Vector{Float64})
                evaluate_model!(x)
                # NLopt expects constraint of form f(x) <= 0
                # max(TOTAL_COST) - cost_cap <= 0 means no (t,r) exceeds cost_cap
                constraint_value = cache_max_cost[] - cost_cap
                return constraint_value  # Should be <= 0 (satisfied if max cost <= cap)
            end
        end
    end

    # Return the objective function, constraint functions (or nothing), and the specific instance of RICE.
    return rice_objective, rice_constraint, rice_cost_constraint, m, n_regions
end



#######################################################################################################################
# OPTIMIZE RICE.
########################################################################################################################
# Description: This function takes an objective function (given user-supplied model settings), and optimizes it for the
#              cost-minimization (global carbon taxes) or utilitarian (regional carbon taxes) approach to find the
#              policy that maximizes global economic welfare. Note that the utilitarian objective function optimizes
#              on the decarbonization fraction (for more efficient code) and then calculates the corresponding regional
#              carbon tax values.
#
# Function Arguments:
#
#       optimization_algorithm:  The optimization algorithm to use from the NLopt package.
#       n_opt_periods:           The number of model time periods to optimize over.
#       stop_time:               The length of time (in seconds) for the optimization to run in case things do not converge.
#       tolerance:               Relative tolerance criteria for convergence (will stop if |Δf| / |f| < tolerance from one iteration to the next.)
#       backstop_price:          The regional backstop prices from RICE2010 (units must be in dollars).
#       run_utilitarian:      A true/false indicator for whether or not to run the utilitarian optimization (true = run utilitarian).
#       ρ:                       Pure rate of time preference.
#       η:                       Elasticity of marginal utility of consumption.
#       remove_negishi:          A true/false indicator for whether RICE should use a social welfare function with Negishi weights (true = remove Negishi weights).
#
# Function Output
#
#       optimized_policy_vector: The vector of optimized policy values returned by the optimization algorithm.
#       optimal_emissions:       Optimal emissions for all time periods.
#       optimal_mitigation:      Optimal mitigation rates for all time periods (2005-2595) resulting form the optimization.
#       optimal_tax:             The optimal tax for all periods and regions used to run the model.
#       opt_model:               An instance of RICE (with user-defined settings) set with the optimal CO₂ mitigation policy.
#       convergence_result:      Indicator for whether or not the optimization converged.
#----------------------------------------------------------------------------------------------------------------------


function optimize_rice(optimization_algorithm::Symbol, n_opt_periods::Int, stop_time::Int, tolerance::Float64, backstop_prices::Array{Float64,2}; run_utilitarian::Bool=true, ρ::Float64=0.008, η::Float64=1.5, remove_negishi::Bool=true, opt_ad::Bool=false, stock_ad::Bool=false, cbudget=nothing, cost_cap=nothing, ext_starting_points=nothing, model_constructor=create_rice, alpha_override=nothing)

    # -------------------------------------------------------------
    # Create objective function and values needed for optimization.
    #--------------------------------------------------------------

    # Create objective function, constraint functions, and instance of RICE, given user settings.
    objective_function, constraint_function, cost_constraint_function, optimal_model, n_regions = construct_rice_objective(run_utilitarian, ρ, η, backstop_prices, remove_negishi, opt_ad, stock_ad, cbudget, cost_cap; model_constructor=model_constructor, alpha_override=alpha_override)

    # Set number of optimzation objectives (will differ between cost-minimization and utilitarian approaches).
    if run_utilitarian == false
        # Number of objectives is equal to time periods being optimized.
        n_objectives = n_opt_periods
        # Upper bound is maximum of backstop price, across all regions.
        upper_bound = maximum(backstop_prices, dims=2)[2:(n_objectives+1)]
        lower_bound = zeros(n_objectives)
        starting_point = upper_bound/2
    else
        if opt_ad == true
            if stock_ad == true
                # Number of objectives is equal to time periods being optimizer × n_regions regions × 3 (mitigation + flow adaptation + stock adaptation).
                n_objectives = n_opt_periods * n_regions * 3
                # Upper bound is 1.0 for mitigation, 2.0 for flow adaptation, 2.0 for stock adaptation.
                upper_bound = vcat(ones(n_opt_periods*n_regions), ones(n_opt_periods*n_regions) .* 2.0, ones(n_opt_periods*n_regions) .* 2.0)

                if ext_starting_points !== nothing
                    # Use external starting points (should be a vector of length n_objectives)
                    starting_point = ext_starting_points
                    println("Using external starting points (length=$(length(starting_point)))")
                else
                    # Default starting points
                    starting_point = vcat(ones(n_opt_periods*n_regions) .* 0.9,
                                         ones(n_opt_periods*n_regions) .* 0.15,
                                         ones(n_opt_periods*n_regions) .* 0.15)
                end
            else
                # Number of objectives is equal to time periods being optimizer × n_regions regions × 2 (mitigation + adaptation).
                n_objectives = n_opt_periods * n_regions * 2
                # Upper bound is 1.0 for mitigation, 2.0 for adaptation (rescaled in constraint).
                upper_bound = vcat(ones(n_opt_periods*n_regions), ones(n_opt_periods*n_regions) .* 1.0)
                if ext_starting_points !== nothing
                    starting_point = ext_starting_points
                    println("Using external starting points (length=$(length(starting_point)))")
                else
                    starting_point = vcat(ones(n_opt_periods*n_regions) .* 0.9,
                                         ones(n_opt_periods*n_regions) .* 0.1)
                end
            end
        else
            n_objectives = n_opt_periods * n_regions
            upper_bound = ones(n_objectives)
            if ext_starting_points !== nothing
                starting_point = ext_starting_points
                println("Using external starting points (length=$(length(starting_point)))")
            else
                starting_point = ones(n_objectives) .* 0.9
            end
        end
        lower_bound = zeros(n_objectives)
    end

    # -------------------------------------------------------
    # Create an NLopt optimization object and optimize model.
    #--------------------------------------------------------
    opt = Opt(optimization_algorithm, n_objectives)

    # Set the bounds.
    lower_bounds!(opt, lower_bound)
    upper_bounds!(opt, upper_bound)

    # Special handling for AUGLAG (meta-algorithm that wraps another optimizer)
    if optimization_algorithm == :LN_AUGLAG || optimization_algorithm == :LD_AUGLAG
        # Create subsidiary optimizer (SBPLX for derivative-free)
        local_opt = Opt(:LN_SBPLX, n_objectives)
        lower_bounds!(local_opt, lower_bound)
        upper_bounds!(local_opt, upper_bound)
        # Key insight: AUGLAG needs MANY outer iterations to enforce constraints.
        # Short, achievable sub-problems → more outer iterations → better constraint handling.
        ftol_rel!(local_opt, 1e-8)                       # Achievable tolerance (was 1e-14, never converged)
        maxtime!(local_opt, max(30, stop_time ÷ 100))    # Short sub-iterations (was stop_time/20)
        maxeval!(local_opt, 3000)                         # Prevent runaway sub-problems

        # Set initial step sizes appropriate for each variable type
        # Larger steps for far-future periods (less sensitive), smaller for near-term
        if run_utilitarian && opt_ad
            time_weight = [1.0 + (t - 1) / (n_opt_periods - 1) for t in 1:n_opt_periods]
            region_time_weight = repeat(time_weight, n_regions)
            if stock_ad
                istep = vcat(
                    region_time_weight .* 0.05,   # Mitigation: 0.05-0.10 (range [0,1])
                    region_time_weight .* 0.10,   # Flow adaptation: 0.10-0.20 (range [0,2])
                    region_time_weight .* 0.10    # Stock adaptation: 0.10-0.20 (range [0,2])
                )
            else
                istep = vcat(
                    region_time_weight .* 0.05,
                    region_time_weight .* 0.10
                )
            end
            initial_step!(local_opt, istep)
        end

        local_optimizer!(opt, local_opt)
        sub_maxtime = max(30, stop_time ÷ 100)
        println("Using AUGLAG with SBPLX subsidiary (sub_ftol=1e-8, sub_time=$(sub_maxtime)s, sub_eval=3000)")
    end

    # Assign the objective function to maximize.
    max_objective!(opt, (x, grad) -> objective_function(x))

    # Add NLopt inequality constraints only for algorithms that support them (AUGLAG, COBYLA)
    if optimization_algorithm == :LN_AUGLAG || optimization_algorithm == :LD_AUGLAG || optimization_algorithm == :LN_COBYLA
        if constraint_function !== nothing
            inequality_constraint!(opt, constraint_function, 1e-1)
        end
        if cost_constraint_function !== nothing
            inequality_constraint!(opt, cost_constraint_function, 1e-4)
        end
    end

    # Set termination time.
    maxtime!(opt, stop_time)

    # Set optimizatoin tolerance (will stop if |Δf| / |f| < tolerance from one iteration to the next).
    ftol_rel!(opt, tolerance)

    # Optimize model.
    maximum_objective_value, optimized_policy_vector, convergence_result = optimize(opt, starting_point)

    # Create optimal decarbonization rates and adaptation for all time periods.
    if run_utilitarian == false
        optimal_mitigation = mitigation_from_tax(optimized_policy_vector, backstop_prices, 2.8)
        optimal_flow_adaptation = nothing
        optimal_stock_adaptation = nothing
    else
        optimal_mitigation, optimal_flow_adaptation, optimal_stock_adaptation = parse_solution_vector(optimized_policy_vector, n_opt_periods, n_regions, opt_ad, stock_ad; stock_ad_default=0.25)
    end

    # Run user-specified version of RICE with optimal mitigation and adaptation policy.
    update_param!(optimal_model, :MIU, optimal_mitigation)
    if run_utilitarian == true && opt_ad == true
        if stock_ad == true
            update_param!(optimal_model, :T_AD_FLOW, optimal_flow_adaptation)
            update_param!(optimal_model, :I_AD, optimal_stock_adaptation)
        else
            update_param!(optimal_model, :T_AD, 2 .* optimal_flow_adaptation)
        end
    end
    run(optimal_model)

    # Create optimal tax rates for all time periods (approach to do so will differ between cost-minimization and utilitarian).
    if run_utilitarian == false
        optimal_tax = [0.0; maximum(backstop_prices, dims=2)[2:end]]
        optimal_tax[2:(length(optimized_policy_vector)+1)] = optimized_policy_vector
    else
        optimal_tax = optimal_model[:emissions, :CPRICE]
    end

    # Create optimal industiral emissions for all time periods.
    optimal_emissions = optimal_model[:emissions, :EIND]

    # Return raw utility (not penalized) for multi-start comparison.
    raw_utility = optimal_model[:welfare, :UTILITY]
    return optimized_policy_vector, optimal_emissions, optimal_mitigation, optimal_flow_adaptation, optimal_stock_adaptation, optimal_tax, optimal_model, convergence_result, raw_utility
end
