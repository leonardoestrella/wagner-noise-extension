"""
    BooleanNetwork

Provides tools and structures to run an evolutionary algorithm of gene-regulatory networks

# Exported structures
- `SimulationParameters`: Holds the parameters of the simulation. See # Fields
- `SimulationData`: The results of a simulation. 

# Exported functions
- `run_simulation`: Executes a single simulation run with a given set of parameters

# Notes
- The module is based on Wagner (1996). Modifications and extensions include more types of 
    initial populations, noisy gene-gene interactions, and different types of selection.
- The module does not use parallel processing because threads are dedicated to running 
    multiple experiments simultaneously.
"""
module BooleanNetwork

using Distributions
using LinearAlgebra
using Parameters
using Random
using StatsBase
using UnPack

using Base: @kwdef

export SimulationParameters, SimulationData
export run_simulation

const valid_initial_pop_types = [:random, :stable, :unstable, :optimal_clones,
    :nonoptimal_clones, :ensemble_sample]
const valid_selection_types = [:wagner, :roulette]

    """
    --------------------
    UTILITIES
    --------------------
    """

    """
        SimulationParameters
        (Mutable)

    Holds configuration settings and initial conditions for a simulation run.

    # Fields
    - `generations::Int`: Total number of evolutionary generations to simulate. 
    - `initial_density::Float64`: Initial connectivity density of networks in the population.
    - `initial_pop_type::String`: Initial population stability.
    - `max_steps::Int`: Maximum steps in phenotype expression.
    - `mutation_prob::Float64`: Probability that a weight is mutated in each generation
    - `noise_dist::X`: Probability distribution for interaction strength noise. 
    - `number_genes::Int`: Number of genes per regulatory network. 
    - `pop_size::Int`: Number of individuals in the population. 
    - `selection_type::String`: Type of selection used to generate new generations of organisms.
    - `selection_pressure::Float64`: Scaling factor determining selection intensity.
    - `unstable_fitness::Float64`: Baseline fitness assigned to unstable network states.
    - `weights_dist::W`: Probability distribution used to sample edge weights. 

    # TODO
    - add regulatory genes, which are not directly under selective pressure
    - add development mode (a string containing synchronous or asynchronous)
    """
    @kwdef mutable struct SimulationParameters{W<:Distribution,X<:Distribution}
        generations::Int = 500
        initial_density::Float64 = 1.0
        initial_pop_type::Symbol = :stable
        max_steps::Int = 100
        mutation_prob::Float64 = 0.01
        noise_dist::X = Bernoulli(1.0)  # standard is no noise
        number_genes::Int = 10
        pop_size::Int = 300
        selection_pressure::Float64 = 10.0
        selection_type::Symbol = :wagner
        unstable_fitness::Float64 = exp(-10.0)
        weights_dist::W = Normal(0.0, 1.0)
    end

    """
        ArtificialPop
        (Mutable)

    A population of gene regulatory networks

    # Fields
    - `pop_size`: Number of networks in the population
    - `number_genes`: Number of genes in each network
    - `matrices`: Interaction weights in each matrix
    - `initial_state`: Initial gene expression state
    - `optimal_phenotype`: Target gene expression

    # TODO
    - Add regulator genes (not directly selected)
    """
    @kwdef mutable struct ArtificialPop
        pop_size::Int
        number_genes::Int
        matrices::Vector{Matrix{Float64}}
        initial_state::Vector{Float64}
        optimal_phenotype::Vector{Float64}
    end

    """
        SimulationData
        (Mutable)

    The data from the simulation.

    # Fields
    - `completion_history`: Number of GRNs that reached stability in that generation.
        Size generations
    - `fitness_history`: Fitness of each GRN per generation. Shape (generations, pop_size)
    - `matrices_history`: All matrices of each generation. Size generations
    - `path_length_history`: Path length of each GRN per generation. Shape (generations, pop_size)
    - `initial_state`: Initial gene expression vector.
    - `optimal_phenotype`: Target gene expression vector.

    # TODO
    - Add regulator genes (not directly selected)
    """
    @kwdef mutable struct SimulationData
        completion_history::Vector{Float64}
        fitness_history::Matrix{Float64}
        matrices_history::Array{Matrix{Float64}}
        path_length_history::Matrix{Union{Float64, Nothing}}
        initial_state::Vector{Float64}
        optimal_phenotype::Vector{Float64}
    end

    """
        hamming_distance(v1::AbstractVector, v2::AbstractVector) -> Float64

    Compute normalized Hamming distance between two gene state vectors.

    # Arguments
    - `v1::AbstractVector`: First vector containing gene states (typically values in {-1, 1}).
    - `v2::AbstractVector`: Second vector containing gene states.

    # Returns
    - `Float64`: Normalized distance in [0,1] where 0 means identical and 1 means opposite

    # Throws
    - `DimensionMismatch`: If vectors are of different size
    """
    function hamming_distance(v1::AbstractVector, v2::AbstractVector)::Float64
        length(v1) == length(v2) || 
            throw(DimensionMismatch("v1 and v2 must be the same size!"))
        size_vectors = length(v1)
        matching_genes = dot(v1, v2)
        return (size_vectors - matching_genes) / (2 * size_vectors)
    end

    """
        develop(W::Matrix{<:Real}, initial_state::Vector{<:Real}, max_steps::Int;
            buffer1::Vector{Float64}=Vector{Float64}(undef, size(W,1)),
            buffer2::Vector{Float64}=Vector{Float64}(undef, size(W,1))
        ) -> Tuple{Union{Vector{Float64},Nothing}, Union{Int,Nothing}}
    
    Develop a network phenotype by iterating `state -> sign.(W * state)` from
    `initial_state`. Halts as soon as a fixed point is reached, a previously visited state 
    recurs (indicating a limit cycle), or `max_steps` iterations have elapsed.
    
    # Arguments
    - `W`: gene interaction matrix of the network; assumed square.
    - `initial_state`: initial gene expression state, of length `size(W, 1)`.
    - `max_steps`: maximum number of iterations attempted before the state is
        declared unstable.
    
    # Keywords
    - `buffer1`, `buffer2`: pre-allocated, length-`size(W,1)` buffers used to hold
        intermediate states. Repeated calls avoid reallocating the state vector each time.
    
    # Returns
    - `(final_state, steps_taken)` if a fixed point was reached. 
    - `(nothing, nothing)` if a cycle was detected or `max_steps` was exhausted
        without stabilizing.

    # Throws
    - `DimensionMismatch` if W is not square, has a different number of columns than
        the size of initial_state, or the buffers do not have the same size as initial_state.
    
    # TODO
    - Add asynchronous development possibility
    - Add the possibility of other activation functions
    - Optimize cycle finding algorithm. It currently holds the entire history
    """
    function develop(
        W::Matrix{<:Real},
        initial_state::Vector{<:Real},
        max_steps::Int;
        buffer1::Vector{Float64}=Vector{Float64}(undef, size(W, 1)),
        buffer2::Vector{Float64}=Vector{Float64}(undef, size(W, 1))
    )::Tuple{Union{Vector{Float64},Nothing},Union{Int,Nothing}} 
        n = size(W, 1)
        size(W, 2) == n ||
            throw(DimensionMismatch("W must be square"))
        length(initial_state) == n ||
            throw(DimensionMismatch("initial_state must have length size(W, 1)"))
        (length(buffer1) == n && length(buffer2) == n) ||
            throw(DimensionMismatch("buffer1 and buffer2 must have length size(W, 1)"))

        buffer1 .= initial_state

        visited_states = Dict{Vector{Float64},Int}()
        sizehint!(visited_states, max_steps + 1)
        visited_states[copy(buffer1)] = 0 

        for step in 0:max_steps 
            mul!(buffer2, W, buffer1)  # in-place matrix multiplication
            buffer2 .= sign.(buffer2)  # in-place activation

            if buffer2 == buffer1
                return (copy(buffer2), step)  # fixed point reached in `step` iterations
            end
            if haskey(visited_states, buffer2)
                return (nothing, nothing)  # limit cycle detected
            end
            visited_states[copy(buffer2)] = step  # copy: buffer2 will be overwritten later

            buffer1, buffer2 = buffer2, buffer1  # swap roles for the next iteration
        end

        return (nothing, nothing)  # did not stabilize within max_steps
    end
    
    """
        apply_noise!(W::Matrix{<:Real}, noise_dist::Distribution) -> Nothing

    Apply noise to a weight matrix. It alters the input to recycle objects instead of 
    creating new ones.

    # Arguments
    - `W`: Weight matrix to modify
    - `noise_dist`: Distribution to sample noise values
    """
    function apply_noise!(W::Matrix{<:Real},noise_dist::Distribution)::Nothing
        if noise_dist isa Bernoulli && params(noise_dist)[1] == 1.0 # early exit if no noise
            return nothing
        end

        noise_matrix = rand(noise_dist, size(W))
        W .*= noise_matrix # apply noise directly
        return nothing
    end

    """
        generate_random_matrix(number_genes::Int, weights_dist::Distribution, density::Float64
        ) -> Matrix{Float64}

    Generate a random gene regulatory matrix with a specified connection density.
    Matrix entries are sampled from `noise_dist`, and each entry is independently
    retained with probability `density` or set to zero otherwise.

    # Arguments
    - `number_genes`: Number of genes
    - `weights_dist`: Distribution used to sample nonzero interaction strengths
    - `density`: Probability that a given interaction is present (must be between 0 and 1)

    # Returns
    - A `number_genes × number_genes` matrix whose entries are sampled from
    `noise_dist` and masked by a Bernoulli adjacency matrix with parameter
    `density`.

    # Throws
    - `DomainError` if the density of the matrix is not between 0.0 and 1.0

    # TODO
    - Add ability to specify a topology (or create new functions to choose topologies)
    """
    function generate_random_matrix(
        number_genes::Int, weights_dist::Distribution, density::Float64
        )::Matrix{Float64}
        0.0 <= density <= 1.0 ||
            throw(DomainError("density must be between 0.0 and 1.0, got $density"))
        adjacency = Bernoulli(density) 
        mask_weights = rand(adjacency,(number_genes,number_genes))
        sample_weights = rand(weights_dist, (number_genes,number_genes))
        return sample_weights .* mask_weights
    end

    """
        initialize_population(params::Dict
        ) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Matrix{Float64}}}

    Initialize a population of gene regulatory networks according to specified mode.
    Uses pre-allocated buffers and optimized matrix generation.

    # Arguments
    - `params`: Dictionary of simulation parameters

    # Returns
    - Tuple of:
        1. Initial state vector
        2. Optimal phenotype vector
        3. Vector of weight matrices
    
    # Throws
    - `ArgumentError`: If the type of initial population is not valid

    #TODO
    - Add asynchornous development mode
    - Add possibility to switch domains [1.0, -1.0] <-> [1.0, 0.0]
    """
    function initialize_population(
        params::SimulationParameters
        )::Tuple{Vector{Float64}, Vector{Float64}, Vector{Matrix{Float64}}}
        params.initial_pop_type  in valid_initial_pop_types ||
            throw(ArgumentError("$(params.initial_pop_type) is not a valid type of initial" * 
                "population. Try with $valid_initial_pop_types"))
             # TODO - check if the type of error is correct
        # ---Constants---
        initial_state = rand([1.0,-1.0], params.number_genes) 
        optimal_phenotype = rand([1.0, -1.0], params.number_genes)
        # TODO - Add possibility of switching [1.0,-1.0] domain to [1.0,0.0]
        buffer1 = Vector{Float64}(undef, params.number_genes)
        buffer2 = Vector{Float64}(undef, params.number_genes)
        
        # ---Useful methods---
        is_stable(matrix) = !isnothing(develop(matrix, initial_state, params.max_steps; 
            buffer1=buffer1, buffer2=buffer2)[1])
        stable_matrix() = begin
            while true
                candidate = generate_random_matrix(params.number_genes,
                    params.weights_dist, params.initial_density)
                stability_test = is_stable(candidate)
                stability_test && return candidate
            end
        end
        unstable_matrix() = begin
            while true
                candidate = generate_random_matrix(params.number_genes,
                    params.weights_dist, params.initial_density)
                stability_test = !is_stable(candidate)
                stability_test && return candidate
            end
        end

        # ---Generating populations---
        if params.initial_pop_type == :random  
        # No stability checks
            matrices = [
                generate_random_matrix(params.number_genes, 
                    params.weights_dist, params.initial_density) 
                for _ in 1:pop_size
            ]
            return (initial_state, optimal_phenotype, matrices)

        elseif params.initial_pop_type == :stable
        # Deterministic development reaches a stable state
            matrices = [stable_matrix() for _ in 1:params.pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif params.initial_pop_type == :unstable
        # Deterministic development does not reach a stable state
            matrices = [unstable_matrix() for _ in 1:params.pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif params.initial_pop_type == :optimal_clones
        # Find a stable matrix and clone it with its expressed phenotype set as optimal
            candidate = stable_matrix()
            expressed_phenotype, _ = develop(candidate, initial_state, params.max_steps;
                buffer1=buffer1, buffer2=buffer2)
            matrices = [copy(candidate) for _ in 1:params.pop_size]
            return (initial_state, expressed_phenotype, matrices)

        elseif params.initial_pop_type == :nonoptimal_clones
        # Find a stable matrix and clone it. The expressed phenotype is not necessarily optimal
            candidate = stable_matrix()
            matrices = [copy(candidate) for _ in 1:params.pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif params.initial_pop_type == :ensemble_sample
        # Find many matrices in which the expressed phenotype is optimal
            matrices = Vector{Matrix{Float64}}(undef, params.pop_size)
            population_count = 0
            attempts = 0
            attempt_limit = 5 * 2^(2*params.number_genes)  # Max attempts before restart (Wagner, 1996)
            
            while population_count < params.pop_size
                candidate = generate_random_matrix(params.number_genes, 
                    params.weights_dist, params.initial_density)
                phenotype, _ = develop(candidate, initial_state, params.max_steps;
                buffer1=buffer1, buffer2=buffer2)

                if !isnothing(phenotype) && phenotype == optimal_phenotype
                # Found a matrix with the desired phenotype
                    population_count += 1
                    matrices[population_count] = candidate
                    attempts = 0
                else
                    attempts += 1
                end
                
                if attempts > attempt_limit  # Reset if we've tried too many times
                    population_count = 0
                    attempts = 0
                    initial_state = rand([1.0, -1.0], params.number_genes)
                    optimal_phenotype = rand([1.0, -1.0], params.number_genes)
                end
            end
            
            return (initial_state, optimal_phenotype, matrices)
        else
            error("Unknown initial population type: $(params.initial_pop_type)")
        end 
    end

    """
    --------------------------
    EVOLUTIONARY DYNAMICS
    --------------------------
    """

    """
        mutation!(W::NetworkMatrix, mutation_prob::Float64, weights_dist::D) where 
        D<:Distribution -> Nothing

    Mutate network weights by resampling each nonzero element independently with a probability

    # Arguments
    - `W`: Weight matrix to mutate
    - `mutation_prob`: Probability of a weight being resampled
    - `weights_dist`: Weights distribution
    """
    function mutation!(W::Matrix{Float64}, mutation_prob::Float64, weights_dist::D
        )::Nothing where D<:Distribution
        if mutation_prob == 0.0  # Early exit if no mutation
            return nothing
        end

        nz_inds = findall(!iszero, W)

        if isempty(nz_inds)  # Early exit if no nonzero elements
            return nothing
        end

        n_elements = length(nz_inds)  # Decide which non-zero elements to mutate
        apply_mutation = rand(n_elements) .< mutation_prob

        if any(apply_mutation)  # Apply mutation
            n_mutations = count(apply_mutation)
            W[nz_inds[apply_mutation]] .= rand(weights_dist, n_mutations)
        end

        return nothing
    end

    """
        indiv_fitness(expressed_phenotype::Union{Vector{<:Real}, Nothing},
            optimal_phenotype::Vector{<:Real}, 
            selection_pressure::Float64,
            unstable_fitness::Float64;
            distance::Function=hamming_distance,
        ) -> Float64

    Calculate individual fitness of a member in the population

    # Arguments
    - `expressed_phenotype`: Phenotype vector or nothing if unstable
    - `optimal_phenotype`: Target phenotype vector
    - `selection_pressure`: Selection strength parameter
    - `unstable_fitness`: Fitness value for unstable phenotypes

    # Keywords
    - `distance`: Distance function. As default, it uses hamming_distance

    # Returns
    - `fitness` evaluated for a given phenotype
    """
    function indiv_fitness(
        expressed_phenotype::Union{Vector{<:Real}, Nothing},
        optimal_phenotype::Vector{<:Real},         selection_pressure::Float64,
        unstable_fitness::Float64;
        distance::Function=hamming_distance,
        )::Float64
        if !isnothing(expressed_phenotype)
            @fastmath begin
                dist = distance(expressed_phenotype, optimal_phenotype)
                return exp(-selection_pressure * dist)
            end
        else 
            return unstable_fitness
        end
    end

    """
        recombine_rows(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:Real -> Matrix{T}

    Recombine rows from two matrices.

    # Arguments
    - `A`, `B`: Source matrices of same size

    # Returns
    - New matrix with rows selected from A or B

    # Throws
    - `DimensionMismatch` if A and B are not the same size
    """
    function recombine_rows(A::AbstractMatrix{T}, B::AbstractMatrix{T}
        )::AbstractMatrix{T} where T<:Real
        size(A) == size(B) ||  # TODO - check if this is an error
            throw(DimensionMismatch("A and B must have the same size. A is $(size(A))" *
                "and B is $(size(B))"))

        rows, cols = size(A)
        C = similar(A)

        @inbounds for i in 1:rows
            if rand() > 0.5  # Non-biased recombination
                C[i, :] .= A[i, :]
            else
                C[i, :] .= B[i, :]
            end
        end

        return C
    end

    """
        create_offspring(pop::ArtificialPop, params::SimulationParameters
        ) -> Tuple{
            Vector{Matrix{Float64}},  # offspring
            Vector{Float64},          # fitness
            Vector{Any},              # steps
            Float64,                  # completion_gen
        }

    Generates a new generation of offspring matrices from an existing `ArtificialPop`
    using recombination, mutation, and fitness-based selection.

    # Arguments
    - `pop::ArtificialPop`: Population containing individuals and their properties.
    - `params::SimulationParameters`: The configuration of the simulation

    # Returns
    A tuple containing:
    1. `Vector{Matrix{Float64}}`: Offspring weight matrices.
    2. `Vector{Float64}`: Fitness of each offspring (Wagner-type selection) 
        or parent (Roulette-type selection)
    3. `Vector{Union{Int,Nothing}}`: Number of steps each offspring took to reach a stable 
        state (`nothing` if unstable).
    4. `Float64`: Fraction of offspring that reached stability (`completion_gen`).

    # Notes
    - Offspring are accepted into the new generation with a probability equal to their 
        computed fitness in the Wagner type of selection.
    - Offspring are added into the new generation with a probability proportional to the
        previous generation's fitness in the Roulette type of selection.
    - Noise is applied after mutation but before development.
    - Stability is determined by the `develop` function; a stable phenotype is any 
        non-`nothing` return value.

    # TODO
    - Integrate asynchronous development possibility
    - Integrate possibility of other activation functions
    """

    function create_offspring(pop::ArtificialPop, params::SimulationParameters;
        buffer1::Vector{Float64}=Vector{Float64}(undef,params.number_genes),
        buffer2::Vector{Float64}=Vector{Float64}(undef, params.number_genes),
        )
        params.selection_type in valid_selection_types ||
            throw(ArgumentError("$(params.selection_type) is not a valid selection type. " *
                "Try with something in $valid_selection_types"))
        @unpack generations, max_steps, mutation_prob, noise_dist, number_genes, pop_size,
            selection_pressure, selection_type, unstable_fitness, weights_dist = params
        @unpack matrices, initial_state, optimal_phenotype = pop  # avoid overwriting
        
        completion_gen = 0.0
        fitness = Vector{Float64}(undef, pop_size)
        offspring = Vector{Matrix{Float64}}(undef, pop_size)
        steps = Vector{Union{Int,Nothing}}(undef,pop_size)
        noisy_W = Matrix{Float64}(undef, number_genes, number_genes)

        # ---Wagner-like selection--- 
        if selection_type == :wagner
            for i in 1:pop_size
                survival = false 
                while !survival
                # Note: This loop may run forever, but it always halts in practice. 
                    parent_i, parent_j = rand(1:pop_size, 2)
                    W_candidate = recombine_rows(matrices[parent_i], matrices[parent_j])  # Sexual recombination

                    mutation!(W_candidate, mutation_prob, weights_dist)  # Mutation
                    copyto!(noisy_W, W_candidate)
                    apply_noise!(noisy_W, noise_dist)  # Noisy gene interactions

                    expressed_phenotype, path_length = develop(noisy_W, initial_state, max_steps;
                        buffer1=buffer1, buffer2=buffer2) 
                    fit = indiv_fitness(expressed_phenotype, optimal_phenotype, 
                        selection_pressure, unstable_fitness) # Compute fitness
                    
                    if rand() < fit  # Decide if offspring is added to the next generation
                        offspring[i] = W_candidate 
                        fitness[i] = fit
                        steps[i] = path_length

                        survival = true
                    end
                end 
            end
            completion_gen = 1.0 - count(isnothing, steps) / pop_size
            
            return offspring, fitness, steps, completion_gen
        # ---Roulette selection---
        elseif selection_type == :roulette
            for i in 1:pop_size
                copyto!(noisy_W, matrices[i])
                apply_noise!(noisy_W, noise_dist)  # Noisy gene interactions
                expressed_phenotype, path_length = develop(noisy_W, initial_state, max_steps;
                buffer1=buffer1, buffer2=buffer2)
                fit = indiv_fitness(expressed_phenotype, optimal_phenotype,
                    selection_pressure, unstable_fitness) # Compute fitness
                
                fitness[i] = fit
                steps[i] = path_length
            end

            normalized_fitness = fitness / sum(fitness)  
            parents_indices = sample(1:pop_size, Weights(normalized_fitness), (2,pop_size);
                replace=true)
            # Choose parents with a probability proportional to their fitness
            completion_gen = 1.0 - count(isnothing, steps) / pop_size  # stable development

            for i in 1:pop_size  # Populate next generation
                parent_i, parent_j = parents_indices[1,i], parents_indices[2,i]
                W_candidate = recombine_rows(matrices[parent_i], matrices[parent_j])
                # Sexual recombination
                mutation!(W_candidate, mutation_prob, weights_dist)  # Mutation
                offspring[i] = W_candidate
            end 

            return offspring, fitness, steps, completion_gen
        end
    end

    """
        run_simulation(params::SimulationParameters)::SimulationData

    Simulate the evolution of gene regulatory networks (GRNs) using noisy gene-gene 
    interactions. The simulation subjects a population of GRNs to selection, mutation, and
    recombination, measuring network stability and phenotype expression over generations.

    # Arguments
    - `params::SimulationParameters`: Simulation parameters

    # Returns
    SimulationData with simulation results:
    - `"matrices_history"::Array{Matrix{Float64},2}`: Weight matrices, shape (generations, pop_size)
    - `"fitness"::Matrix{Float64}`: Individual fitness values, shape (generations, pop_size)
    - `"path_length"::Matrix{Union{Int,Nothing}}`: Steps to stability, shape (generations, pop_size)
    - `"completion"::Vector{Float64}`: Fraction stable per generation, length generations
    - `"initial_state"::Vector{Int}`: Initial gene expression state
    - `"phenotypic_optima"::Vector{Int}`: Target phenotype vector

    # Notes
    - The type of selection determines which is the fitness of the first generation. 
        Wagner-like selection records the fitness of the offspring, and Roulette selection
        records the fitness of the parents. 
    
    # TODO
    - Make sure it works with asynchronous development
    - Integrate possibility of other activation functions
    """

    function run_simulation(params::SimulationParameters)::SimulationData
        @unpack generations, initial_density, initial_pop_type, max_steps, mutation_prob, 
            noise_dist, number_genes, pop_size, selection_pressure, selection_type, 
            unstable_fitness, weights_dist = params

        simulation_data = SimulationData(
            completion_history = zeros(Float64, generations),
            fitness_history = Matrix{Float64}(undef, generations, pop_size),
            matrices_history = Array{Matrix{Float64}}(undef, generations, pop_size),
            path_length_history = Matrix{Any}(undef, generations, pop_size),
            initial_state = Vector{Float64}(undef, number_genes),
            optimal_phenotype = Vector{Float64}(undef, number_genes),
        )
        noisy_W = Matrix{Float64}(undef, number_genes, number_genes)  # Preallocate memory
        buffer1 = Vector{Float64}(undef, number_genes)
        buffer2 = Vector{Float64}(undef, number_genes)

        # Helper function
        function record_generation!(gen::Int, offspring::Vector{Matrix{Float64}},
            fit::Vector{Float64}, steps::Vector{Union{Int,Nothing}}, completion::Int;
            simulation_data::SimulationData=simulation_data)

            simulation_data.matrices_history[gen, :] .= offspring
            simulation_data.fitness_history[gen, :] .= fit
            simulation_data.path_length_history[gen, :] .= steps
            simulation_data.completion_history[gen] .= completion / pop_size

            return nothing
        end

        initial_state, optimal_phenotype, matrices = initialize_population(params)
        population = ArtificialPop(pop_size = pop_size, number_genes = number_genes,
            matrices = matrices, initial_state = initial_state,
            optimal_phenotype = optimal_phenotype)
        simulation_data.initial_state = initial_state
        simulation_data.optimal_phenotype = optimal_phenotype

        if selection_type == :wagner
            for (index, matrix) in enumerate(population.matrices) # First-generation data
                # Note: initialize_population does not compute fitness, path length or 
                # completion for the initial population
                copyto!(noisy_W, matrix)
                apply_noise!(noisy_W, noise_dist)
                phenotype, path_length = develop(noisy_W, population.initial_state, max_steps;
                    buffer1=buffer1, buffer2=buffer2)
                fit = indiv_fitness(phenotype, population.optimal_phenotype, 
                    selection_pressure, unstable_fitness)

                simulation_data.fitness_history[1,index] .= fit
                simulation_data.path_length_history[1, index] .= path_length
                simulation_data.matrices_history[1, index] .= matrix
            end
            simulation_data.completion_history[1] .= 1.0 - count(isnothing,
                simulation_data.path_length_history[1,:]) / pop_size
            start_gen = 2
        else
            start_gen = 1
        end

        for gen in start_gen:generations  # Run simulation
            offspring, fit, steps, completion_gen = create_offspring(population, params)
            record_generation!(gen, offspring, fit, steps, completion_gen)
            copyto!(population.matrices, offspring)
        end

        return simulation_data
    end
end