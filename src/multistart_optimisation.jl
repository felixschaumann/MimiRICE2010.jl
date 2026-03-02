#######################################################################################################################
# MULTI-START OPTIMIZATION FOR RICE
#######################################################################################################################
# Hierarchical warm-up + structured multi-start + refinement for finding global optima
# in the multimodal AUGLAG+SBPLX optimization landscape.
#
# Key insight: Instead of random sampling in 180 dimensions, parameterize each control as a
# monotonic ramp defined by ~3 shape parameters, then sample those meta-parameters via LHS.
#######################################################################################################################

using Random


"""
    simple_lhs(n, dims)

Latin Hypercube Sampling: generates `n` points in `dims` dimensions, each in [0,1].
Each dimension is divided into `n` equal strata, with one sample per stratum.
"""
function simple_lhs(n::Int, dims::Int)
    samples = zeros(n, dims)
    for d in 1:dims
        perm = randperm(n)
        for i in 1:n
            samples[i, d] = (perm[i] - rand()) / n
        end
    end
    return samples
end


"""
    generate_monotonic_ramp(n_periods, start_val, end_val, steepness)

Generate a smooth monotonic profile using a logistic function.
`steepness` controls the S-curve shape: low (~3) = gradual, high (~15) = sharp transition.
"""
function generate_monotonic_ramp(n_periods::Int, start_val::Float64, end_val::Float64, steepness::Float64)
    t = range(0, 1, length=n_periods)
    midpoint = 0.5
    raw = @. 1.0 / (1.0 + exp(-steepness * (t - midpoint)))
    # Normalize to [0,1], then scale to [start_val, end_val]
    raw_min, raw_max = extrema(raw)
    if raw_max ≈ raw_min
        return fill((start_val + end_val) / 2, n_periods)
    end
    normalized = @. (raw - raw_min) / (raw_max - raw_min)
    return @. start_val + (end_val - start_val) * normalized
end


"""
    perturb_solution(x, magnitude; lower=0.0, upper_mit=1.0, upper_ad=2.0, n_opt_periods, n_regions, stock_ad)

Create a perturbed copy of a solution vector. Perturbation is proportional to
the variable value to avoid pushing near-zero values negative.
"""
function perturb_solution(x::Vector{Float64}, magnitude::Float64,
                          n_opt_periods::Int, n_regions::Int, stock_ad::Bool)
    perturbed = copy(x)
    n_mit = n_opt_periods * n_regions

    for i in eachindex(perturbed)
        # Determine bounds for this variable
        if i <= n_mit
            ub = 1.0   # mitigation
        else
            ub = 2.0   # adaptation
        end
        # Perturbation proportional to distance from bounds
        room_up = ub - perturbed[i]
        room_down = perturbed[i]
        noise = magnitude * (2 * rand() - 1)
        if noise > 0
            perturbed[i] += noise * room_up
        else
            perturbed[i] += noise * room_down
        end
        perturbed[i] = clamp(perturbed[i], 0.0, ub)
    end
    return perturbed
end


"""
    generate_structured_starting_points(n_starts, n_opt_periods, n_regions; kwargs...)

Generate diverse starting points for multi-start optimization.
Combines known good solutions with LHS-sampled structured ramps.

Returns a vector of `(starting_point_vector, label)` tuples.
"""
function generate_structured_starting_points(n_starts::Int, n_opt_periods::Int, n_regions::Int;
                                              base_solution::Union{Nothing, Vector{Float64}}=nothing,
                                              ext_solution::Union{Nothing, Vector{Float64}}=nothing,
                                              stock_ad::Bool=true)
    starts = Tuple{Vector{Float64}, String}[]

    # 1. Base solution (warm-up result from Phase 1)
    if base_solution !== nothing
        push!(starts, (copy(base_solution), "warm-up"))
        # Perturbed variant
        push!(starts, (perturb_solution(base_solution, 0.08, n_opt_periods, n_regions, stock_ad), "warm-up-p"))
    end

    # 2. External starting points (loaded from previous run)
    if ext_solution !== nothing
        push!(starts, (copy(ext_solution), "external"))
        # Perturbed variant
        push!(starts, (perturb_solution(ext_solution, 0.08, n_opt_periods, n_regions, stock_ad), "external-p"))
    end

    # 3. Default starting point (same as original optimize_rice default)
    if stock_ad
        default_sp = vcat(ones(n_opt_periods * n_regions) .* 0.9,
                         ones(n_opt_periods * n_regions) .* 0.15,
                         ones(n_opt_periods * n_regions) .* 0.15)
    else
        default_sp = vcat(ones(n_opt_periods * n_regions) .* 0.9,
                         ones(n_opt_periods * n_regions) .* 0.15)
    end
    push!(starts, (default_sp, "default"))

    # 4+. Fill remaining slots with LHS-sampled structured ramps
    n_ramps = n_starts - length(starts)
    if n_ramps > 0
        ramp_starts = generate_ramp_starting_points(n_ramps, n_opt_periods, n_regions, stock_ad)
        for (i, sp) in enumerate(ramp_starts)
            push!(starts, (sp, "ramp-$i"))
        end
    end

    return starts[1:min(n_starts, length(starts))]
end


"""
    generate_ramp_starting_points(n_starts, n_opt_periods, n_regions, stock_ad)

Generate starting points by parameterizing controls as smooth ramps and sampling
the shape parameters via Latin Hypercube Sampling.

Meta-parameters per region:
- Mitigation (logistic): start_level ∈ [0,0.5], end_level ∈ [0.5,1], steepness ∈ [3,15]
- Flow adaptation (power law): scale ∈ [0.05,0.5], curvature ∈ [0.3,2]
- Stock adaptation (power law): scale ∈ [0.05,0.5], curvature ∈ [0.3,2]
"""
function generate_ramp_starting_points(n_starts::Int, n_opt_periods::Int, n_regions::Int, stock_ad::Bool)
    meta_per_region = stock_ad ? 7 : 5  # 3 (mit) + 2 (flow) [+ 2 (stock)]
    total_meta = meta_per_region * n_regions

    lhs = simple_lhs(n_starts, total_meta)

    # Bounds
    mit_ub = 1.0
    ad_ub = 2.0

    starting_points = Vector{Float64}[]

    for i in 1:n_starts
        n_controls = stock_ad ? 3 : 2
        sp = zeros(n_opt_periods * n_regions * n_controls)

        for r in 1:n_regions
            offset = (r - 1) * meta_per_region

            # Mitigation: logistic ramp
            mit_start = 0.0 + 0.5 * lhs[i, offset + 1]       # [0.0, 0.5]
            mit_end   = 0.5 + 0.5 * lhs[i, offset + 2]       # [0.5, 1.0]
            mit_steep = 3.0 + 12.0 * lhs[i, offset + 3]       # [3.0, 15.0]

            mit_ramp = generate_monotonic_ramp(n_opt_periods, mit_start, mit_end, mit_steep)
            mit_ramp = clamp.(mit_ramp, 0.0, mit_ub)

            # Column-major layout: all times for region r together
            mit_idx = (r - 1) * n_opt_periods
            sp[(mit_idx + 1):(mit_idx + n_opt_periods)] = mit_ramp

            # Flow adaptation: power-law ramp A(t) = scale * (t/T)^curvature
            flow_scale = 0.05 + 0.45 * lhs[i, offset + 4]    # [0.05, 0.50]
            flow_curv  = 0.3 + 1.7 * lhs[i, offset + 5]      # [0.3, 2.0]

            flow_ramp = [flow_scale * (t / n_opt_periods)^flow_curv for t in 1:n_opt_periods]
            flow_ramp = clamp.(flow_ramp, 0.0, ad_ub)

            flow_idx = n_opt_periods * n_regions + (r - 1) * n_opt_periods
            sp[(flow_idx + 1):(flow_idx + n_opt_periods)] = flow_ramp

            # Stock adaptation: power-law ramp (if enabled)
            if stock_ad
                stock_scale = 0.05 + 0.45 * lhs[i, offset + 6]
                stock_curv  = 0.3 + 1.7 * lhs[i, offset + 7]

                stock_ramp = [stock_scale * (t / n_opt_periods)^stock_curv for t in 1:n_opt_periods]
                stock_ramp = clamp.(stock_ramp, 0.0, ad_ub)

                stock_idx = 2 * n_opt_periods * n_regions + (r - 1) * n_opt_periods
                sp[(stock_idx + 1):(stock_idx + n_opt_periods)] = stock_ramp
            end
        end

        push!(starting_points, sp)
    end

    return starting_points
end


"""
    check_feasibility(model, cbudget, cost_cap)

Check whether a solved model satisfies the carbon budget and cost cap constraints.
Returns true if all constraints are satisfied (within tolerance).
"""
function check_feasibility(model, cbudget, cost_cap)
    if cbudget !== nothing
        cca = model[:emissions, :CCA][end]
        if cca > cbudget + 0.1  # 0.1 GtC tolerance
            return false
        end
    end
    if cost_cap !== nothing
        max_cost = maximum(model[:neteconomy, :TOTAL_COST])
        if max_cost > cost_cap + 1e-4  # 0.01% GDP tolerance
            return false
        end
    end
    return true
end


"""
    multistart_optimize_rice(n_starts, optimization_algorithm, n_opt_periods, total_time,
                              tolerance, backstop_prices; kwargs...)

Multi-start optimization with hierarchical warm-up and refinement.
Returns the same 9-element tuple as `optimize_rice()` for drop-in compatibility.

## Time budget allocation (for total_time):
- Phase 1 (~1/6): Hierarchical warm-up (mitigation-only → full solve)
- Phase 2 (~2/3): Multi-start from diverse structured starting points
- Phase 3 (~1/6): Refinement of best feasible solution
"""
function multistart_optimize_rice(n_starts::Int, optimization_algorithm::Symbol,
                                   n_opt_periods::Int, total_time::Int, tolerance::Float64,
                                   backstop_prices::Array{Float64,2};
                                   run_utilitarian::Bool=true, ρ::Float64=0.008, η::Float64=1.5,
                                   remove_negishi::Bool=true, opt_ad::Bool=false, stock_ad::Bool=false,
                                   cbudget=nothing, cost_cap=nothing, ext_starting_points=nothing)

    n_regions = 2  # 2-region aggregation

    # Time allocation — generous refinement for tight convergence
    warmup_time = total_time ÷ 8
    per_start_time = max(60, (total_time * 5 ÷ 8) ÷ max(1, n_starts))
    refinement_time = total_time ÷ 4

    # Collect all candidate solutions
    all_results = @NamedTuple{utility::Float64, feasible::Bool, result::Any, label::String}[]

    # ==========================================
    # Phase 1: Hierarchical warm-up
    # ==========================================
    println("\n" * "="^60)
    println("Phase 1: Hierarchical warm-up ($(warmup_time)s total)")
    println("="^60)

    # Stage 1: Mitigation-only solve (unconstrained, SBPLX — faster for unconstrained)
    warmup_stage1_time = warmup_time ÷ 2
    println("Stage 1: Mitigation-only SBPLX solve ($(warmup_stage1_time)s)...")

    mit_result = optimize_rice(:LN_SBPLX, n_opt_periods, warmup_stage1_time,
                               1e-8, backstop_prices;
                               run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                               opt_ad=false, stock_ad=false, cbudget=nothing, cost_cap=nothing)

    mit_policy = mit_result[1]  # optimized_policy_vector (n_opt_periods × n_regions)
    println("  Mitigation-only utility: $(mit_result[9])")

    # Stage 2: Extend mitigation solution with default adaptation, solve full problem
    if opt_ad && stock_ad
        warmup_start = vcat(mit_policy,
                           ones(n_opt_periods * n_regions) .* 0.15,
                           ones(n_opt_periods * n_regions) .* 0.15)
    elseif opt_ad
        warmup_start = vcat(mit_policy, ones(n_opt_periods * n_regions) .* 0.15)
    else
        warmup_start = mit_policy
    end

    warmup_stage2_time = warmup_time - warmup_stage1_time
    println("Stage 2: Full AUGLAG solve from warm-up ($(warmup_stage2_time)s)...")

    warmup_result = optimize_rice(optimization_algorithm, n_opt_periods, warmup_stage2_time,
                                   tolerance, backstop_prices;
                                   run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                                   opt_ad=opt_ad, stock_ad=stock_ad, cbudget=cbudget,
                                   cost_cap=cost_cap, ext_starting_points=warmup_start)

    warmup_utility = warmup_result[9]
    warmup_feasible = check_feasibility(warmup_result[7], cbudget, cost_cap)
    push!(all_results, (utility=warmup_utility, feasible=warmup_feasible,
                        result=warmup_result, label="warm-up"))
    println("  Warm-up result: utility = $(warmup_utility), feasible = $(warmup_feasible)")

    # ==========================================
    # Phase 2: Multi-start from diverse points
    # ==========================================
    println("\n" * "="^60)
    println("Phase 2: Multi-start ($(n_starts) starts × $(per_start_time)s)")
    println("="^60)

    starting_points = generate_structured_starting_points(n_starts, n_opt_periods, n_regions;
                                                          base_solution=warmup_result[1],
                                                          ext_solution=ext_starting_points,
                                                          stock_ad=stock_ad)

    for (i, (sp, label)) in enumerate(starting_points)
        println("\nStart $i/$(length(starting_points)) ($label)...")
        try
            result = optimize_rice(optimization_algorithm, n_opt_periods, per_start_time,
                                    1e-6, backstop_prices;   # achievable exploration tolerance
                                    run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                                    opt_ad=opt_ad, stock_ad=stock_ad, cbudget=cbudget,
                                    cost_cap=cost_cap, ext_starting_points=sp)
            utility = result[9]
            feasible = check_feasibility(result[7], cbudget, cost_cap)
            push!(all_results, (utility=utility, feasible=feasible, result=result, label=label))
            println("  -> utility = $utility, feasible = $feasible")
        catch e
            println("  -> FAILED: $e")
        end
    end

    # ==========================================
    # Phase 3: Refinement of best feasible
    # ==========================================
    feasible_results = filter(r -> r.feasible, all_results)
    if isempty(feasible_results)
        println("\nWARNING: No feasible solutions found! Using best infeasible.")
        candidates = all_results
    else
        candidates = feasible_results
    end

    # Find best candidate (manual argmax for Julia version compatibility)
    best_idx = 1
    for i in 2:length(candidates)
        if candidates[i].utility > candidates[best_idx].utility
            best_idx = i
        end
    end
    best = candidates[best_idx]

    println("\n" * "="^60)
    println("Phase 3: Refinement from best ($(best.label), utility=$(best.utility)) — $(refinement_time)s")
    println("="^60)

    refined = optimize_rice(optimization_algorithm, n_opt_periods, refinement_time, tolerance,
                            backstop_prices; run_utilitarian=true, ρ=ρ, η=η,
                            remove_negishi=remove_negishi, opt_ad=opt_ad, stock_ad=stock_ad,
                            cbudget=cbudget, cost_cap=cost_cap,
                            ext_starting_points=best.result[1])

    refined_utility = refined[9]

    # ==========================================
    # Summary
    # ==========================================
    println("\n" * "="^60)
    println("Multi-start optimization summary")
    println("="^60)
    for (i, r) in enumerate(all_results)
        marker = (r === best) ? " <-- BEST" : ""
        println("  $(lpad(i, 2)). $(rpad(r.label, 12)) utility = $(round(r.utility, sigdigits=8))  feasible = $(r.feasible)$marker")
    end
    improvement = refined_utility - best.utility
    println("  Refined: utility = $(round(refined_utility, sigdigits=8)) (delta = $(round(improvement, sigdigits=4)))")
    println("="^60)

    # Save multi-start log
    save_multistart_log(all_results, refined_utility, best.label)

    return refined
end


"""
    save_multistart_log(results, refined_utility, best_label)

Save multi-start optimization log to CSV for cross-run comparison.
"""
function save_multistart_log(results, refined_utility, best_label)
    log_dir = joinpath(@__DIR__, "../results")
    mkpath(log_dir)
    log_path = joinpath(log_dir, "multistart_log.csv")

    open(log_path, "w") do f
        println(f, "start_num,label,utility,feasible,is_best")
        for (i, r) in enumerate(results)
            is_best = r.label == best_label
            println(f, "$i,$(r.label),$(r.utility),$(r.feasible),$is_best")
        end
        println(f, "refined,refined,$refined_utility,true,true")
    end
    println("Multi-start log saved to: $log_path")
end


"""
    sweep_cost_caps(caps, n_starts, optimization_algorithm, n_opt_periods, total_time,
                     tolerance, backstop_prices; kwargs...)

Continuation/homotopy across cost cap values. Solves from loosest to tightest constraint,
using each solution as starting point for the next (the solution path is continuous as
the feasible region shrinks).

Returns a Dict mapping each cost cap value to its optimization result tuple.
"""
function sweep_cost_caps(caps::Vector{Float64}, n_starts::Int, optimization_algorithm::Symbol,
                          n_opt_periods::Int, total_time::Int, tolerance::Float64,
                          backstop_prices::Array{Float64,2};
                          run_utilitarian::Bool=true, ρ::Float64=0.008, η::Float64=1.5,
                          remove_negishi::Bool=true, opt_ad::Bool=false, stock_ad::Bool=false,
                          cbudget=nothing, results_folder::String="MyResults",
                          ad_string::String="")

    # Sort from loosest to tightest
    sorted_caps = sort(caps, rev=true)

    results = Dict{Float64, Any}()
    prev_solution = nothing

    for (i, cap) in enumerate(sorted_caps)
        println("\n" * "#"^60)
        println("Cost cap sweep: $cap ($(i)/$(length(sorted_caps)))")
        println("#"^60)

        if i == 1
            # First (loosest): full multi-start exploration
            result = multistart_optimize_rice(n_starts, optimization_algorithm, n_opt_periods,
                                              total_time, tolerance, backstop_prices;
                                              run_utilitarian=true, ρ=ρ, η=η,
                                              remove_negishi=remove_negishi, opt_ad=opt_ad,
                                              stock_ad=stock_ad, cbudget=cbudget, cost_cap=cap,
                                              ext_starting_points=prev_solution)
        else
            # Subsequent: use previous solution + fewer starts (solution path is continuous)
            result = multistart_optimize_rice(max(3, n_starts ÷ 2), optimization_algorithm,
                                              n_opt_periods, total_time ÷ 2, tolerance,
                                              backstop_prices; run_utilitarian=true, ρ=ρ, η=η,
                                              remove_negishi=remove_negishi, opt_ad=opt_ad,
                                              stock_ad=stock_ad, cbudget=cbudget, cost_cap=cap,
                                              ext_starting_points=prev_solution)
        end

        results[cap] = result
        prev_solution = result[1]  # optimized_policy_vector for warm-starting next cap

        # Save results for this cap value
        cap_string = replace(string(cap), "." => "p")
        output_directory = joinpath(@__DIR__, "../results", results_folder,
                                    "$(ad_string)rice_utilitarian_cap$(cap_string)")
        mkpath(output_directory)

        opt_mitigation = result[3]
        opt_flow_adaptation = result[4]
        opt_stock_adaptation = result[5]
        opt_tax = result[6]
        opt_model = result[7]

        save(joinpath(output_directory, "MitigationRate.csv"), DataFrame(opt_mitigation, :auto))
        if opt_ad && stock_ad
            save(joinpath(output_directory, "FlowAdaptation.csv"), DataFrame(opt_flow_adaptation, :auto))
            save(joinpath(output_directory, "StockAdaptation.csv"), DataFrame(opt_stock_adaptation, :auto))
        end
        save(joinpath(output_directory, "Emissions.csv"), DataFrame(opt_model[:emissions, :EIND], :auto))
        save(joinpath(output_directory, "Carbon Tax.csv"), DataFrame(opt_tax, :auto))
        save(joinpath(output_directory, "Temperature.csv"), DataFrame(temp=opt_model[:climatedynamics, :TATM]))
        save(joinpath(output_directory, "GDP.csv"), DataFrame(opt_model[:neteconomy, :YNET], :auto))
        save(joinpath(output_directory, "PerCapitaConsumption.csv"), DataFrame(opt_model[:neteconomy, :CPC], :auto))

        println("Results saved to: $output_directory")
    end

    return results
end


"""
    homotopy_solve(steps, initial_solution, optimization_algorithm, n_opt_periods,
                    backstop_prices; kwargs...)

Continuation/homotopy across multiple constraint parameters (cost_cap + cbudget).
Traces the solution path from a loose constraint to the target, using each step's
solution as warm-start for the next. This is more robust than cold-starting a
tightly-constrained problem because AUGLAG can calibrate its Lagrange multipliers
gradually.

Each element of `steps` is a NamedTuple with fields `cost_cap` and `cbudget`.
Steps should be ordered from loosest to tightest constraints.

Returns a vector of NamedTuples with fields:
  step, cost_cap, cbudget, utility, cca, max_cost, feasible, policy_vector, result
"""
function homotopy_solve(
    steps::Vector{<:NamedTuple},
    initial_solution::Vector{Float64},
    optimization_algorithm::Symbol,
    n_opt_periods::Int,
    backstop_prices::Array{Float64,2};
    time_per_step::Int=600,
    first_step_time::Int=1200,
    final_step_time::Int=1500,
    first_step_starts::Int=4,
    final_step_starts::Int=4,
    tolerance::Float64=1e-15,
    ρ::Float64=0.008, η::Float64=1.5,
    remove_negishi::Bool=true, opt_ad::Bool=true, stock_ad::Bool=true
)
    n_steps = length(steps)
    @assert n_steps >= 2 "Need at least 2 homotopy steps"

    results = NamedTuple[]
    current_solution = copy(initial_solution)

    println("\n" * "="^70)
    println("HOMOTOPY SOLVE — $(n_steps) steps")
    println("="^70)
    for (i, s) in enumerate(steps)
        println("  Step $i: cost_cap=$(s.cost_cap), cbudget=$(s.cbudget)")
    end
    println("="^70)

    for (i, step) in enumerate(steps)
        println("\n" * "#"^70)
        println("Homotopy step $i/$(n_steps): cost_cap=$(step.cost_cap), cbudget=$(step.cbudget)")
        println("#"^70)

        step_start = time()
        local result

        if i == 1
            # First step (loosest): multi-start from initial solution
            println("  Strategy: multistart ($(first_step_starts) starts, $(first_step_time)s)")
            result = multistart_optimize_rice(
                first_step_starts, optimization_algorithm, n_opt_periods,
                first_step_time, tolerance, backstop_prices;
                run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                opt_ad=opt_ad, stock_ad=stock_ad,
                cbudget=step.cbudget, cost_cap=step.cost_cap,
                ext_starting_points=current_solution
            )
        elseif i == n_steps
            # Final step (tightest): multi-start for robustness
            println("  Strategy: multistart ($(final_step_starts) starts, $(final_step_time)s)")
            result = multistart_optimize_rice(
                final_step_starts, optimization_algorithm, n_opt_periods,
                final_step_time, tolerance, backstop_prices;
                run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                opt_ad=opt_ad, stock_ad=stock_ad,
                cbudget=step.cbudget, cost_cap=step.cost_cap,
                ext_starting_points=current_solution
            )
        else
            # Middle steps: single-start from previous + perturbed alternative
            println("  Strategy: single-start + perturbed ($(time_per_step)s)")
            n_regions = 2

            # Primary: warm-start from previous solution
            result_a = optimize_rice(
                optimization_algorithm, n_opt_periods, time_per_step,
                tolerance, backstop_prices;
                run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                opt_ad=opt_ad, stock_ad=stock_ad,
                cbudget=step.cbudget, cost_cap=step.cost_cap,
                ext_starting_points=current_solution
            )

            # Alternative: perturbed start
            perturbed = perturb_solution(current_solution, 0.05, n_opt_periods, n_regions, stock_ad)
            result_b = optimize_rice(
                optimization_algorithm, n_opt_periods, time_per_step,
                tolerance, backstop_prices;
                run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                opt_ad=opt_ad, stock_ad=stock_ad,
                cbudget=step.cbudget, cost_cap=step.cost_cap,
                ext_starting_points=perturbed
            )

            # Pick the better feasible result
            feas_a = check_feasibility(result_a[7], step.cbudget, step.cost_cap)
            feas_b = check_feasibility(result_b[7], step.cbudget, step.cost_cap)
            util_a = result_a[9]
            util_b = result_b[9]

            if feas_a && feas_b
                result = util_a >= util_b ? result_a : result_b
                println("  Both feasible: picked $(util_a >= util_b ? "primary" : "perturbed") ($(max(util_a, util_b)))")
            elseif feas_a
                result = result_a
                println("  Only primary feasible ($util_a)")
            elseif feas_b
                result = result_b
                println("  Only perturbed feasible ($util_b)")
            else
                # Neither feasible — retry with 2× time
                println("  ⚠ Neither feasible! Retrying with 2× time ($(2*time_per_step)s)...")
                result = optimize_rice(
                    optimization_algorithm, n_opt_periods, 2 * time_per_step,
                    tolerance, backstop_prices;
                    run_utilitarian=true, ρ=ρ, η=η, remove_negishi=remove_negishi,
                    opt_ad=opt_ad, stock_ad=stock_ad,
                    cbudget=step.cbudget, cost_cap=step.cost_cap,
                    ext_starting_points=util_a >= util_b ? result_a[1] : result_b[1]
                )
            end
        end

        # Extract diagnostics
        model = result[7]
        utility = result[9]
        cca = model[:emissions, :CCA][end]
        max_cost = maximum(model[:neteconomy, :TOTAL_COST])
        feasible = check_feasibility(model, step.cbudget, step.cost_cap)

        # Smoothness metric: max absolute period-to-period jump in MIU (first 20 periods)
        miu = model[:emissions, :MIU]
        n_plot = min(20, size(miu, 1))
        smoothness = maximum(abs.(diff(miu[1:n_plot, :], dims=1)))

        elapsed = round(time() - step_start, digits=1)

        push!(results, (
            step=i,
            cost_cap=step.cost_cap,
            cbudget=step.cbudget,
            utility=utility,
            cca=cca,
            max_cost=max_cost,
            feasible=feasible,
            policy_vector=result[1],
            result=result,
            smoothness=smoothness,
            elapsed=elapsed
        ))

        # Log
        println("\n  --- Step $i summary ---")
        println("  Utility:     $(round(utility, sigdigits=8))")
        println("  CCA:         $(round(cca, sigdigits=6)) GtC $(step.cbudget !== nothing ? "(budget=$(step.cbudget))" : "")")
        println("  Max cost:    $(round(max_cost, sigdigits=4)) $(step.cost_cap !== nothing ? "(cap=$(step.cost_cap))" : "")")
        println("  Feasible:    $feasible")
        println("  Smoothness:  $(round(smoothness, sigdigits=3)) (max MIU jump)")
        println("  Time:        $(elapsed)s")

        # Update warm-start for next step
        current_solution = result[1]
    end

    # Final summary table
    println("\n" * "="^70)
    println("HOMOTOPY SUMMARY")
    println("="^70)
    header = rpad("Step", 6) * rpad("cbudget", 10) * rpad("Utility", 14) * rpad("CCA", 10) * rpad("MaxCost", 10) * rpad("Feasible", 10) * rpad("Smooth", 8) * rpad("Time", 8)
    println(header)
    println("-"^length(header))
    for r in results
        cb_str = r.cbudget !== nothing ? string(round(r.cbudget, digits=0)) : "—"
        row = rpad(string(r.step), 6) *
              rpad(cb_str, 10) *
              rpad(string(round(r.utility, sigdigits=8)), 14) *
              rpad(string(round(r.cca, sigdigits=6)), 10) *
              rpad(string(round(r.max_cost, sigdigits=4)), 10) *
              rpad(string(r.feasible), 10) *
              rpad(string(round(r.smoothness, sigdigits=3)), 8) *
              rpad(string(r.elapsed), 8)
        println(row)
    end
    println("="^70)

    return results
end
