"""
    Custom Stats

Provides the data analysis tools used in these simulations. It achieves three things:
1. Summarize data into average and standard deviations per generation across simulations
2. Compute alignment score per matrix and population
3. Compute mutational robustness per matrix and population

# Exported functions
- `summarize_history`: Computes the average and sample standard deviation of data with shape
    (generations, pop_size) into two vectors of size generations
- `alignment_score`: Computes the alignment score of a marix, returning a Float64. To use in
    a grid of matrices, use broadcasting as alignment_score.(grid_matrices, Ref(vector)).
- `mutational_robustness`: Computes the mutational robustness metrics of a matrix and returns
    a Tuple of four Float64. To use in a grid of matrices, use broadcasting as
    mutational_robustness.(grid_matrices,Ref(...))

# Notes
- mutational_robustness_pop takes samples across time and population sizes because it is very
    expensive to run. 
- The module does not use parallel processing because threads are dedicated to running
    multiple experiments. 
"""
module CustomStats

using Distributions
using LinearAlgebra
using Random

include("../src/wagner_algorithm.jl")
using .BooleanNetwork  # SimulationData, SimulationParameters
#  TODO - Corroborate if this is appropriate. I want to specify the structure I am using, and
#   importing the entire module at the same time?

using StatsBase: mean, std, var

export summarize_history, alignment_score, mutational_robustness_pop

    """
        summarize_history(data::AbstractArray) -> Tuple {Vector{Float64}, Vector{Float64}}

    Summarizes the data of an abstract matrix of size (generations, pop_size) in their mean
    and sample standard deviation. It excludes "nothing" values. 

    # Arguments
    - `data::AbstractMatrix`: Abstract matrix holding the data, that can be fitness, path 
        length, or alignment score.
    
    # Returns
    A tuple containing:
    - `Vector{Float64}`: Averages with size generations
    - `Vector{Float64}`: Sample standard deviation with size generations. 

    # Notes
    - It ignores nothing values, so [1.0, 2.0, nothing] will be taken as [1.0, 2.0]
    - Output vectors might contain NaN if there are not enough data points! 
    - Standard error is SEM = σ/√N, where σ is the sample std and N the sample size
    """
    function summarize_history(data::AbstractMatrix)::Tuple{Vector{Float64}, Vector{Float64}}
        n_rows = size(data)[1]
        means = Vector{Float64}(undef, n_rows)
        stds = Vector{Float64}(undef, n_rows)

        for row_idx in 1:n_rows
            row_view = @view data[row_idx,:]
            filtered_data = filter(!isnothing, row_view)
            n = length(filtered_data)

            if n == 0
                means[i] = NaN
                stds[i]  = NaN
            elseif n == 1
                means[i] = filtered_data[1]
                stds[i]  = NaN  
            else
                means[i] = mean(filtered_data)
                stds[i]  = std(filtered_data; corrected=true)
            end
        end
        return (means, stds)
    end

    """
        alignment_score(matrix::Matrix{Float64}, vector::Vector{Float64}) -> Float64

    Compute alignment between a matrix's rows and a target vector using normalized dot products.

    # Arguments
    - `matrix::Matrix{Float64}`: A matrix whose weights represent gene-gene interactions
    - `vector::Vector{Float64}`: A vector from where alignment score will be computed 

    # Returns
    - `Float64`: The alignment score of the given matrix

    # Notes
    - It is possible to broadcast to a grid of matrices by setting Ref(vec)
    """
    function alignment_score(matrix::Matrix{Float64}, vector::Vector{Float64})::Float64
        normalized_mat = mapslices(matrix; dims=2) do row  # Normalize rows by their L1 norms
            row_norm = norm(row, 1)
            iszero(row_norm) ? row : row / row_norm  # Leave as empty if it was empty
        end

        return dot(vector, normalized_mat, vector)  # vector^T * normalized_mat * vector
    end

    """
        summarize_simulation_run(result::Dict) -> Dict

    Wrapper for the main statistics of a run. 

    Returns a dictionary with keys:
    - fitness_stats: Vector of per-generation fitness statistics
    - path_stats: Vector of per-generation path length statistics
    - completion_stats: Vector of per-generation completion rates
    - alignment_stats: Vector of per-generation alignment statistics
    """  
    function summarize_simulation_run(result::SimulationData)::Dict
        alignment_scores_population = alignment_score.(result.matrices_history, 
            Ref(result.optimal_phenotype))
        return Dict(
            "fitness_stats" => summarize_history(result.fitness_history),
            "path_stats" => summarize_history(result.path_length_history),
            "completion_stats" => (mean(result.completion_history), 
                std(result.completion_history; corrected=true)),
            "alignment_stats" => summarize_history(alignment_scores_population)
        )
    end

    """
        generate_expression_distribution(matrix::Matrix{Float64}, 
                            initial_state::Vector{Float64},
                            number_noise_samples::Int,
                            noise_dist::Distribution,
                            max_steps::Int,  # Default: simulation parameter
        )::Tuple{Vector{Float64}, Float64}

    Generates a sample from the expression distribution for a given matrix using a specified
    noise distribution

    # Returns
    A tuple containing: 
    - avg_stable_expression: The sampled average expression vector.
    - unstable_proportion: The fraction of times the expressed phenotype was unstable.

    # Notes
    - stable_count helps keep track of the index at which the last stable state was stored

    # TODO
    - Ensure compatibility with asynchornous development. 
    - Ensure compatibility with different activation functions. 
    """

    function generate_expression_distribution(matrix::Matrix{Float64}, 
                            initial_state::Vector{Float64},
                            number_noise_samples::Int,
                            noise_dist::Distribution,
                            max_steps::Int=SimulationParameters.max_steps,
    )::Tuple{Vector{Float64}, Float64}
        number_genes = size(matrix, 1)
        unstable_count = 0
        stable_count = 0
        stable_states = Matrix{Float64}(undef, number_noise_samples, number_genes)
        buffer_matrix = Matrix{Float64}(undef, number_genes, number_genes)
        buffer_vec_1 = Vector{Float64}(undef, number_genes)
        buffer_vec_2 = Vector{Float64}(undef, number_genes)  # Memory allocation

        for idx in 1:number_noise_samples
            copyto!(buffer_matrix, matrix)
            BooleanNetwork.apply_noise!(buffer_matrix, noise_prob, noise_dist)
            final_state, _ = BooleanNetwork.develop(buffer_matrix, initial_state, max_steps;
                buffer1=buffer_vec_1, buffer2=buffer_vec_2)

            if !isnothing(final_state)
                stable_count += 1
                stable_states[stable_count, :] = final_state
            else
                unstable_count += 1
            end
        end

        if stable_count == 0
            avg_stable_expression = zeros(Float64, number_genes)
            # No stable expression was found in this sample, so we assume every state is
            # equally likely to be a stable state
        else
            avg_stable_expression = vec(mean(view(stable_states, 1:stable_count, :); dims=1))
            # Average across the number of stable states that were sampled
        end

        return avg_stable_expression, unstable_count / number_noise_samples
    end

    """
        compute_mut_robustness(
            matrix::Matrix{Float64},  # Matrix to test mutational robustness from 
            initial_state::Vector{Float64},  # Development begins using this state
            number_mutation_samples::Int,  # Number of mutations
            number_noise_samples::Int,  # Number of samples in the expression the distribuition
            noise_dist::Distribution;  # Factor multiplying each weight in the matrix
            mutation_prob::Float64,  # Probability that a weight is resampled (independently)
            weights_dist::Distribution,  # Matrix weights will be resampled from here
                Default: simulation parameter
            max_steps::Int,  # Max steps in development. Default: simulation parameter
            )::NamedTuple{stable_expression_shift::Float64, stable_expression_var::Float64,
                unstable_prob_shift::Float64, unstable_prob_var::Float64}

    Estimate the mutational robustness of a regulatory matrix by repeatedly mutating its 
    non-zero entries, simulating noisy dynamics, and aggregating how much the stable 
    expression distribution sample summary statistics differ relative to the non-mutated
    baseline.

    Returns a `NamedTuple` with:
    - `stable_expression_shift`: average absolute difference between the baseline and 
    mutated average stable expression vectors (averaged over mutations).
    - `stable_expression_var`: total sample variance in the expressed phenotypes average 
        expression from mutated phenotypes
    - `unstable_prob_shift`: average difference in the probability of converging to
      an unstable phenotype after mutation relative to baseline.
    - `unstable_prob_var`: variance in the probaiblity of converging to an unstable
    phenotype from mutated matrices.

    # Throws
    - `ArgumentError` if number_mutation_samples or number_noise_samples are less than or
        equal to 1.

    # Notes
    - Can broadcast to a grid of matrices by using compute_mut_robustness.(grid_matrices, 
    Ref(initial_state), Ref(number_mutation_samples), ...)

    # TODO:
    - Ensure compatibility with asynchronous development and different activation functions
    """
    function compute_mut_robustness(
        matrix::Matrix{Float64}, 
        initial_state::Vector{Float64},
        number_mutation_samples::Int,
        number_noise_samples::Int,
        noise_dist::Distribution;
        mutation_prob::Float64, 
        weights_dist::Distribution=SimulationParameters.weights_dist,
        max_steps::Int=SimulationParameters.max_steps,
    )::NamedTuple{stable_expression_shift::Float64, stable_expression_var::Float64,
        unstable_prob_shift::Float64, unstable_prob_var::Float64}
        number_mutation_samples > 1 ||
            throw(ArgumentError("number_mutations_samples must be larger than 1! Got " *
                "$number_mutation_samples"))
        number_noise_samples > 1 ||
            throw(ArgumentError("number_noise_samples must be larger than 1! Got " *
                                "$number_mutation_samples"))

        number_genes = size(matrix, 1)
        baseline_mean, baseline_unstable_prob = generate_expression_distribution(matrix,
            initial_state, number_noise_samples, noise_dist, max_steps)

        unstable_probs = Vector{Float64}(undef, number_mutation_samples)
        stable_expressions = Matrix{Float64}(undef, number_mutation_samples, number_genes)
        mutated_matrix = Matrix{Float64}(undef, number_genes, number_genes)

        for mutation_idx in 1:number_mutation_samples
            copyto!(mutated_matrix, matrix)
            BooleanNetwork.mutation!(mutated_matrix, mutation_prob, mutation_dist)
            
            mutated_mean, mutated_unstable_prob = generate_expression_distribution(
                mutated_matrix, initial_state, number_noise_samples, noise_dist, max_steps)

            stable_expressions[mutation_idx,:] .= mutated_mean
            unstable_probs[mutation_idx] .= mutated_unstable_prob
        end

        mean_stable_expression_mutations = vec(mean(stable_expressions, dims=1))
        stable_expression_var = sum(var(stable_expressions, dims=1, corrected=true))
        mean_unstable_prob_mutations = mean(unstable_probs)
        unstable_prob_var = var(unstable_probs, corrected=true)

        stable_expression_shift = norm(baseline_mean - mean_stable_expression_mutations,1)
        unstable_prob_shift = baseline_unstable_prob - mean_unstable_prob_mutations

        return (
            stable_expression_shift = stable_expression_shift,
            stable_expression_var = stable_expression_var,
            unstable_prob_shift = unstable_prob_shift,
            unstable_prob_var = unstable_prob_var
        )
    end  # PROGRESS MARK: AUGUST 17, 2026
end # module
