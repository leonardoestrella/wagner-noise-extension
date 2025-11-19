module CustomStats

using Statistics
using StatsBase 
using Distributions
using LinearAlgebra
using Random
using Base.Threads

# Include core simulation functionality
include("wagner_algorithm.jl")
using .BooleanNetwork

export
    # Simulation summary functions
    summarize_simulation_run,
    compute_fitness_stats,
    compute_path_stats,
    compute_alignment_stats,
    # Population analysis functions
    compute_mut_robustness,
    compute_population_mut_robustness
    # Future features
    # compute_network_motifs  # TODO: Implement motif analysis

# Type aliases for clarity
const SimulationMatrix = Matrix{Float64}
const GenerationSummary = NamedTuple{
    (:mean, :p5, :p25, :p50, :p75, :p95, :validity_pct),
    Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}
}
const MutationalRobustnessSummary = NamedTuple{
    (:mean_expression_shift, :mean_unstable_shift),
    Tuple{Float64, Float64}
}

    """
        compute_percentile_stats(data::AbstractVector{<:Real}) -> GenerationSummary

    Compute mean, percentiles (5,25,50,75,95), and percentage of valid datapoints 
    for a generation's data skipping missing data. 

    Returns a NamedTuple with fields: mean, p5, p25, p50, p75, p95, validity
    """
    function compute_percentile_stats(data::AbstractVector{<:Union{Real,Nothing}})::GenerationSummary
        valid_data = filter(!isnothing, data)
        validity = if isempty(data)
            # If the original input was empty, it's 100% "valid" (empty)
            100.0
        else
            length(valid_data) / length(data) * 100.0
        end

        if isempty(valid_data)
            @warn "No valid data points found ($(round(validity, digits=1))% valid)"
            # Return default values, now with `validity_pct`
            return (mean=0.0, p5=0.0, p25=0.0, p50=0.0, p75=0.0, p95=0.0, validity_pct=validity)
        end

        if validity < 100
            @info "Computing stats on $(round(validity, digits=1))% of data points"
        end

        return (
            mean=mean(valid_data),
            p5=percentile(valid_data, 5),
            p25=percentile(valid_data, 25),
            p50=percentile(valid_data, 50),
            p75=percentile(valid_data, 75),
            p95=percentile(valid_data, 95),
            validity_pct=validity 
        )
    end
    """
        compute_fitness_stats(fitness_matrix::Matrix{Float64}) -> Vector{GenerationSummary}

    Compute per-generation fitness statistics including mean and percentiles.
    """
    function compute_fitness_stats(fitness_matrix::Matrix{Float64})::Vector{GenerationSummary}
        return [compute_percentile_stats(fitness_matrix[gen, :]) 
                for gen in axes(fitness_matrix, 1)]
    end

    """
        compute_path_stats(path_matrix::Matrix) -> Vector{GenerationSummary}

    Compute per-generation path length statistics, handling nothing values.
    """
    function compute_path_stats(path_matrix::Matrix)::Vector{GenerationSummary}
        return [compute_percentile_stats(path_matrix[gen, :])
                for gen in axes(path_matrix, 1)]
    end

    """
        compute_alignment_score(matrix::AbstractMatrix, vector::AbstractVector) -> Float64

    Compute alignment between a matrix's rows and a target vector using normalized dot products.
    """
    function compute_alignment_score(matrix::AbstractMatrix, vector::AbstractVector)::Float64
        n_rows = size(matrix, 1)
        scores = zeros(n_rows)
        
        for i in 1:n_rows
            row = view(matrix, i, :)
            row_norm = norm(row, 1)  # L1 norm
            scores[i] = iszero(row_norm) ? 0.0 : dot(row, vector) / row_norm
        end
        
        return dot(scores, vector) / length(vector)
    end

    """
        compute_alignment_stats(matrices::AbstractArray{Matrix{Float64}}, 
                            target::Vector) -> Vector{GenerationSummary}

    Compute per-generation alignment statistics for a population of matrices.
    """
    function compute_alignment_stats(
        matrices::AbstractArray{Matrix{Float64}}, 
        target::Vector
    )::Vector{GenerationSummary}
        n_gens = size(matrices, 1)
        alignments = [Float64[] for _ in 1:n_gens]
        
        for gen in 1:n_gens
            gen_alignments = [compute_alignment_score(matrices[gen, i], target) 
                            for i in axes(matrices, 2)]
            alignments[gen] = gen_alignments
        end
        
        return [compute_percentile_stats(gen_align) for gen_align in alignments]
    end
    """
        compute_alignment_stats(matrices::AbstractArray{Matrix{Float64}}, 
                            target::Vector) -> Vector{GenerationSummary}

    Compute per-generation alignment score for a population of matrices.
    """
    function compute_all_alignments(
        matrices::AbstractArray{Matrix{Float64}},
        target::Vector
    )::Matrix{Float64}
        return map(M -> compute_alignment_score(M, target), matrices)
    end


    """
        summarize_simulation_run(result::Dict) -> Dict

    Compute comprehensive statistics for a simulation run.

    Returns a dictionary with keys:
    - fitness_stats: Vector of per-generation fitness statistics
    - path_stats: Vector of per-generation path length statistics
    - completion: Vector of per-generation completion rates
    - alignment_stats: Vector of per-generation alignment statistics
    """
    function summarize_simulation_run(result::Dict)::Dict
        return Dict(
            "fitness_stats" => compute_fitness_stats(result["fitness"]),
            "path_stats" => compute_path_stats(result["path_length"]),
            "completion" => result["completion"],
            "alignment_stats" => compute_alignment_stats(
                result["matrices"], 
                result["phenotypic_optima"]
            )
        )
    end

    """
        generate_expression_distribution(matrix::Matrix{Float64}, 
                            initial_state::Vector{Int},
                            n_noise_masks::Int,
                            noise_dist::Distribution,
                            noise_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS["noise_prob"],
                            max_steps::Int=BooleanNetwork.STANDARD_PARAMETERS["max_steps"],
                            activation=BooleanNetwork.activation)

    Generates a sample from the expression distribution for a given matrix
    using the noise distribution. 

    Returns a tuple
    - avg_stable_expression: A vector with the average expression of each gene
        in the expression distribution
    - unstable_proportion: The proportion of times the expressed phenotype was unstable
    """

    function generate_expression_distribution(matrix::Matrix{Float64}, 
                            initial_state::Vector{Int},
                            n_noise_masks::Int,
                            noise_dist::Distribution,
                            noise_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS["noise_prob"],
                            max_steps::Int=BooleanNetwork.STANDARD_PARAMETERS["max_steps"],
                            activation=BooleanNetwork.activation)
        N_genes = size(matrix, 1)
        stable_states = Matrix{Float64}(undef, n_noise_masks, N_genes)
        stable_count = 0
        unstable_count = 0
        buffer_matrix = Matrix{Float64}(undef, N_genes, N_genes)

        for idx in 1:n_noise_masks
            copyto!(buffer_matrix, matrix)
            BooleanNetwork.apply_noise!(buffer_matrix, noise_prob, noise_dist)

            final_state, _ = BooleanNetwork.develop(buffer_matrix, initial_state, max_steps, activation)
            
            if final_state !== nothing
                stable_count += 1
                stable_states[stable_count, :] = final_state
            else
                unstable_count += 1
            end
        end

        if stable_count == 0
            avg_stable_expression = zeros(Float64, N_genes)
        else
            avg_stable_expression = vec(mean(view(stable_states, 1:stable_count, :); dims=1))
        end

        return avg_stable_expression, unstable_count / n_noise_masks
    end

    """
        compute_mut_robustness(matrix::Matrix{Float64}, 
                            initial_state::Vector{Int},
                            n_mutations::Int,
                            n_noise_masks::Int,
                            noise_dist::Distribution;
                            mut_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS.pr,
                            mr::Float64=BooleanNetwork.STANDARD_PARAMETERS.mr,
                            ؟r::Float64=BooleanNetwork.STANDARD_PARAMETERS.؟r,
                            noise_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS.noise_prob,
                            max_steps::Int=BooleanNetwork.STANDARD_PARAMETERS.max_steps,
                            activation=BooleanNetwork.activation)::MutationalRobustnessSummary

    Estimate the mutational robustness of a regulatory matrix by repeatedly
    mutating its non-zero entries, simulating noisy dynamics, and aggregating
    how much the stable expression profile and instability probability shift
    relative to the unmutated baseline.

    Returns a `NamedTuple` with:
    - `mean_expression_shift`: average absolute difference between the baseline
      and mutated average stable expression vectors (averaged over mutations).
    - `mean_unstable_shift`: average absolute difference in the probability of converging to
      an unstable phenotype after mutation (mutated minus baseline).
    """
    function compute_mut_robustness(matrix::Matrix{Float64}, 
                            initial_state::Vector{Int},
                            n_mutations::Int,
                            n_noise_masks::Int,
                            noise_dist::Distribution;
                            mut_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS["pr"],
                            mr::Float64=BooleanNetwork.STANDARD_PARAMETERS["mr"],
                            sigma_r::Float64=BooleanNetwork.STANDARD_PARAMETERS["σr"],
                            noise_prob::Float64=BooleanNetwork.STANDARD_PARAMETERS["noise_prob"],
                            max_steps::Int=BooleanNetwork.STANDARD_PARAMETERS["max_steps"],
                            activation=BooleanNetwork.activation)::MutationalRobustnessSummary
        if n_mutations <= 0
            return (mean_expression_shift=0.0, mean_unstable_shift=0.0)
        end

        mutation_dist = Normal(mr,sigma_r)
        n_genes = size(matrix, 1)
        baseline_mean, baseline_unstable_prob = generate_expression_distribution(
            matrix,
            initial_state,
            n_noise_masks,
            noise_dist,
            noise_prob,
            max_steps,
            activation
        )

        expression_shifts = Vector{Float64}(undef, n_mutations)
        unstable_probability_shifts = Vector{Float64}(undef, n_mutations)
        mutated_matrix = Matrix{Float64}(undef, n_genes, n_genes)

        for mutation_idx in 1:n_mutations
            copyto!(mutated_matrix, matrix)
            BooleanNetwork.reg_mutation!(mutated_matrix, mut_prob, mutation_dist)

            mutated_mean, mutated_unstable_prob = generate_expression_distribution(
                mutated_matrix,
                initial_state,
                n_noise_masks,
                noise_dist,
                noise_prob,
                max_steps,
                activation
            )

            expression_shifts[mutation_idx] = mean(abs.(mutated_mean .- baseline_mean))
            unstable_probability_shifts[mutation_idx] = abs(mutated_unstable_prob - baseline_unstable_prob)
        end

        return (
            mean_expression_shift=mean(expression_shifts),
            mean_unstable_shift=mean(unstable_probability_shifts)
        )
    end   

    """
        compute_population_mut_robustness(matrices::AbstractVector{Matrix{Float64}},
                                        initial_state::Vector{Int},
                                        n_mutations::Int,
                                        n_noise_masks::Int,
                                        noise_dist::Distribution; kwargs...) 
            -> Vector{MutationalRobustnessSummary}

    Compute mutational robustness summaries for every matrix within a population.
    Returns a vector with one `MutationalRobustnessSummary` per network in the
    order provided.
    """
    function compute_population_mut_robustness(
        matrices::AbstractVector{Matrix{Float64}},
        initial_state::Vector{Int},
        n_mutations::Int,
        n_noise_masks::Int,
        noise_dist::Distribution;
        kwargs...
    )::Vector{MutationalRobustnessSummary}
        results = Vector{MutationalRobustnessSummary}(undef, length(matrices))
        for (idx, matrix) in enumerate(matrices)
            results[idx] = compute_mut_robustness(
                matrix,
                initial_state,
                n_mutations,
                n_noise_masks,
                noise_dist;
                kwargs...
            )
        end
        return results
    end

    """
        compute_population_mut_robustness(matrices::AbstractArray{Matrix{Float64}}, 
                                        initial_state::Vector{Int},
                                        n_mutations::Int,
                                        n_noise_masks::Int,
                                        noise_dist::Distribution; kwargs...) 
            -> Array{MutationalRobustnessSummary}

    Array-based overload returning results with the same shape as the input
    matrix collection.
    """
    function compute_population_mut_robustness(
        matrices::AbstractArray{Matrix{Float64}},
        initial_state::Vector{Int},
        n_mutations::Int,
        n_noise_masks::Int,
        noise_dist::Distribution;
        kwargs...
    )::Array{MutationalRobustnessSummary}
        result = similar(matrices, MutationalRobustnessSummary)
        for idx in eachindex(matrices)
            result[idx] = compute_mut_robustness(
                matrices[idx],
                initial_state,
                n_mutations,
                n_noise_masks,
                noise_dist;
                kwargs...
            )
        end
        return result
    end 

end # module
