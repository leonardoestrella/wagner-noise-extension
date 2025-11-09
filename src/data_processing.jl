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
    compute_mut_robustness
    # Future features
    # compute_network_motifs  # TODO: Implement motif analysis

# Type aliases for clarity
const SimulationMatrix = Matrix{Float64}
const GenerationSummary = NamedTuple{
    (:mean, :p5, :p25, :p50, :p75, :p95, :validity_pct),
    Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}
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
        compute_mut_robustness(matrix::Matrix{Float64}, 
                            initial_state::Vector{Int},
                            parameters::Dict,
                            trials::Int=100) -> Float64

    Compute mutational robustness score for a single matrix by applying
    random mutations and checking phenotype preservation.

    Returns fraction of mutations that preserve the original phenotype.
    """
    function compute_mut_robustness(
        matrix::Matrix{Float64},
        initial_state::Vector{Int},
        parameters::Dict,
        trials::Int=100
    )::Float64
        # Get original phenotype
        orig_phen, _ = develop(matrix, initial_state, parameters["max_steps"])
        if orig_phen === nothing
            return 0.0
        end
        
        # Cache distribution for mutations
        d = Normal(parameters["mr"], parameters["σr"])
        
        # Find mutable elements
        nz_inds = findall(!iszero, matrix)
        if isempty(nz_inds)
            return 1.0  # Empty matrix is trivially robust
        end
        
        # Try random mutations
        invariant_count = 0
        W_test = copy(matrix)
        
        for _ in 1:trials
            idx = rand(nz_inds)
            old_val = W_test[idx]
            W_test[idx] = rand(d)
            
            phen, _ = develop(W_test, initial_state, parameters["max_steps"])
            if phen !== nothing && phen == orig_phen
                invariant_count += 1
            end
            
            W_test[idx] = old_val  # Restore
        end
        
        return invariant_count / trials
    end

    # TODO: Network motif analysis
    # Placeholder for future implementation of network motif detection and analysis
    """
        compute_network_motifs(matrix::Matrix{Float64}) -> Dict

    [Not yet implemented]
    Will analyze network structure for common motifs and regulatory patterns.
    """
    function compute_network_motifs(matrix::Matrix{Float64})
        error("Network motif analysis not yet implemented")
    end

end # module