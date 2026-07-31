"""
    BooleanNetwork

Provides tools and structures to run an evolutionary algorithm of gene-regulatory networks

# Exported functions

- `run_simulation`: Executes a single simulation run with a given set of parameters
"""
module BooleanNetwork

using Distributions
using LinearAlgebra
using Parameters
using Random
using StatsBase

using Base: @kwdef

export run_simulation

    """
    --------------------
    UTILITIES
    --------------------
    """

    """
        SimulationParameters

    Holds configuration settings and initial conditions for a simulation run.

    # Fields
    - `generations::Int`: Total number of evolutionary generations to simulate. 
    - `initial_density::Float64`: Initial connectivity density of networks in the population.
    - `initial_pop_type::String`: Initial population stability.
    - `max_steps::Int`: Maximum steps in phenotype expression.
    - `noise_dist::X`: Probability distribution for interaction strength noise. 
    - `number_genes::Int`: Number of genes per regulatory network. 
    - `pop_size::Int`: Number of individuals in the population. 
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
        initial_pop_type::String = "stable"
        max_steps::Int = 100
        noise_dist::X = Bernoulli(1.0)  # standard is no noise
        number_genes::Int = 10
        pop_size::Int = 300
        selection_pressure::Float64 = 10.0
        unstable_fitness::Float64 = exp(-10.0)
        weights_dist::W = Normal(0.0, 1.0)
    end

    """
        hamming_distance(v1::AbstractVector, v2::AbstractVector, N_target::Integer) -> Float64

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
        return (size_vectors - matching_genes) / (2 * N_target)
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

        noise_matrix = rand(noise_dist,size(W))
        W = W.*noise_matrix # apply noise directly

        return nothing
    end

    """
        generate_random_matrix(
            number_genes::Int,noise_dist::Distribution,density<:Real
        ) -> Matrix{Float64}

    Generate a random gene regulatory matrix with a specified connection density.
    Matrix entries are sampled from `noise_dist`, and each entry is independently
    retained with probability `density` or set to zero otherwise.

    # Arguments
    - `number_genes`: Number of genes
    - `noise_dist`: Distribution used to sample nonzero interaction strengths
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
        number_genes::Int, noise_dist::Distribution, density::Real
        )::Matrix{Float64}
        0.0 <= density <= 1.0 ||
            throw(DomainError("density must be between 0.0 and 1.0, got $density"))
        adjacency = Bernoulli(density) 
        return rand(noise_dist, (number_genes,number_genes)) .* rand(adjacency,(number_genes,number_genes))
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

    #TODO
    - Add asynchornous development mode
    - Add possibility to switch domains [1.0, -1.0] <-> [1.0, 0.0]
    """
    function initialize_population(
        params::SimulationParameters
        )::Tuple{Vector{Float64}, Vector{Float64}, Vector{Matrix{Float64}}}
        # ---Constants---
        initial_state = rand([1.0,-1.0], params.number_genes) 
        optimal_phenotype = rand([1.0, -1.0], params.number_genes)
        # TODO - Add possibility of switching [1.0,-1.0] domain to [1.0,0.0]
        buffer1 = Vector{Float64}(undef, N)
        buffer2 = Vector{Float64}(undef, N)
        
        # ---Useful methods---
        is_stable(matrix) = isnothing(develop(
            matrix, initial_state, params.max_steps, buffer1, buffer2)[1])
        stable_matrix() = begin
            while true
                candidate = generate_random_matrix(params.number_genes,
                    params.noise_dist, params.initial_density)
                stability_test = is_stable(candidate)
                stability_test && return candidate
            end
        end
        unstable_matrix() = begin
            while true
                candidate = generate_random_matrix(params.number_genes,
                    params.noise_dist, params.initial_density)
                stability_test = !is_stable(candidate)
                stability_test && return candidate
            end
        end

        # ---Generating populations---
        if mode == "random"  
        # No stability checks
            matrices = [
                generate_random_matrix(params.number_genes, 
                    params.noise_dist, params.initial_density) 
                for _ in 1:pop_size
            ]
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "stable"  
        # Deterministic development reaches a stable state
            matrices = [stable_matrix() for _ in 1:pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "unstable"  
        # Deterministic development does not reach a stable state
            matrices = [unstable_matrix() for _ in 1:pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "optimal clones"  
        # Find a stable matrix and clone it with its expressed phenotype set as optimal
            candidate = stable_matrix()
            expressed_phenotype, _ = develop(candidate, initial_state, params.max_steps;
                buffer1=buffer1, buffer2=buffer2)
            matrices = [copy(candidate) for _ in 1:params.pop_size]
            return (initial_state, expressed_phenotype, matrices)

        elseif mode == "nonoptimal clones"
        # Find a stable matrix and clone it. The expressed phenotype is not necessarily optimal
            candidate = stable_matrix()
            matrices = [copy(candidate) for _ in 1:params.pop_size]
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "ensemble sample"
        # Find many matrices in which the expressed phenotype is optimal
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            population_count = 0
            attempts = 0
            attempt_limit = 5 * 2^(2*params.number_genes)  # Max attempts before restart (Wagner, 1996)
            
            while population_count < pop_size
                candidate = generate_random_matrix(params.number_genes, 
                    params.noise_dist, params.initial_density)
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
            error("Unknown mode: $mode")
        end 
    end

    # Progress Mark - July 31, 2026

    """
    -----------------
    STRUCTURES
    -----------------
    """

    """
        artificial_pop: A population of gene regulatory networks
        (Mutable)

    Properties:
    - pop_size: Number of networks in the population
    - N_regulator: Number of regulator genes
    - N_target: Number of target genes
    - matrices: Weight matrices representing the GRNs
    - initial_state: Initial gene expression state
    - phenotypic_optima: Target gene expression configuration

    Performance optimizations:
    - Uses type-stable Vector{Matrix{Float64}} for matrices
    - Direct matrix storage without wrapper objects
    - Memory-efficient representation
    """
    @with_kw mutable struct artificial_pop
        pop_size::Int64;
        N_regulator::Int64;
        N_target::Int64;
        matrices::Vector{Matrix{Float64}};  # Changed from pop_ens to matrices
        initial_state::Vector{Int};
        phenotypic_optima::Vector{Int};
    end

    """
        replace_matrices!(pop::artificial_pop, matrices::Vector{<:AbstractMatrix}) -> Nothing

    Efficiently replace weight matrices in a population using optimized iteration
    and in-place updates. Uses type stability and explicit typing.

    # Arguments
    - `pop`: Population to update
    - `matrices`: New weight matrices to assign

    """
        function replace_matrices!(pop::artificial_pop, matrices::Vector{<:AbstractMatrix})
            @inbounds copyto!(pop.matrices, matrices)
            return nothing
        end
    """
    ---------------------------
    MUTATION AND RECOMBINATION
    --------------------------
    """

    """
        reg_mutation!(W::NetworkMatrix, mr::Float64, σr::Float64, pr::Float64) -> Nothing

    Mutate network weights using a resampling method.

    # Arguments
    - `W`: Weight matrix to modify
    - `mr`: Mean of weight distribution
    - `σr`: Standard deviation of weight distribution
    - `pr`: Probability of mutation occurring

    """

    function reg_mutation!(W::NetworkMatrix, pr::Float64, d::Normal)
        # Early exit if no mutation
        if pr == 0.0
            return nothing
        end

        # Find nonzero elements (only once)
        nz_inds = findall(!iszero, W)
        if isempty(nz_inds)
            return nothing
        end

        # Decide which non-zero elements to mutate
        n_elements = length(nz_inds)
        apply_mutation = rand(n_elements) .< pr # Creates a boolean mask

        # Apply mutation if any are selected
        if any(apply_mutation)
            # Get the number of mutations to apply
            n_mutations = count(apply_mutation)

            # Resample weights for the selected indices
            W[nz_inds[apply_mutation]] .= rand(d, n_mutations)
        end

        return nothing
    end

    """
        con_mutation!(W::NetworkMatrix,
                    pc::Float64,
                    d::Normal -> Nothing

    Mutate network connectivity by modifying edges (adding or removing).

    # Arguments
    - `W`: Weight matrix to modify
    - `pc`: Probability of connectivity mutation
    - `mr`: Mean of weight distribution for new edges
    - `σr`: Standard deviation of weight distribution for new edges

    # NOTES:
        -   The function is currently deprecated and is kept here to further
            investigate other types of mutations later on
    """
    function con_mutation!(W::NetworkMatrix, pc::Float64, d::Normal)
        if pc == 0.0
            return nothing
        end

        n_total_elements = length(W) 

        # 1. Compute how many mutations to perform
        n_mutations = rand(Binomial(n_total_elements, pc))

        if n_mutations == 0
            return nothing
        end

        # 2. Decide which indices to mutate
        indices_to_mutate = sample(1:n_total_elements, n_mutations, replace=false)

        # 3. Apply the mutations
        for idx in indices_to_mutate
            if W[idx] != 0.0 # Turn off the edge
                W[idx] = 0.0
            else # Turn on the edge
                W[idx] = rand(d) # rand(d) is called, getting a new value
            end
        end
        return nothing
    end

    """
        indiv_fitness(expressed_phenotype::Union{Vector{<:Real}, Nothing},
                    optimal_phenotype::Vector{<:Real},
                    N_target::Integer,
                    s::Real,
                    distance::Function,
                    unstable_fitness::Real) -> Float64

    Calculate individual fitness of a member in the population

    # Arguments
    - `expressed_phenotype`: Phenotype vector or nothing if unstable
    - `optimal_phenotype`: Target phenotype vector
    - `N_target`: Number of target genes
    - `s`: Selection strength
    - `distance`: Distance function
    - `unstable_fitness`: Fitness value for unstable phenotypes

    # Returns
    - Calculated fitness value
    """
    function indiv_fitness(expressed_phenotype::Union{Vector{<:Real}, Nothing},
                        optimal_phenotype::Vector{<:Real},
                        N_target::Integer,
                        s::Real,
                        distance::Function,
                        unstable_fitness::Real)::Float64
        if expressed_phenotype !== nothing
            # Use @fastmath for optimized math operations
            @fastmath begin
                dist = distance(expressed_phenotype, optimal_phenotype, N_target)
                return exp(-s * dist)
            end
        else 
            return unstable_fitness
        end
    end

    """
        recombine_rows(A::AbstractMatrix{T},
                    B::AbstractMatrix{T},
                    p_rec::Real) where T<:Real -> Matrix{T}

    Recombine rows from two matrices.

    # Arguments
    - `A`, `B`: Source matrices of same size
    - `p_rec`: Probability of selecting a row from matrix B

    # Returns
    - New matrix with rows selected from A or B
    """
    function recombine_rows(A::AbstractMatrix{T},
                          B::AbstractMatrix{T},
                          p_rec::Real) where T<:Real
        @assert size(A) == size(B) "Matrices must have the same size"

        # Early return for no recombination
        if iszero(p_rec)
            return copy(A)  # Return a copy of A
        end

        m, n = size(A)
        C = similar(A)

        # Use in-place row copies. This avoids creating temporary views and
        # keeps operations type-stable while minimizing allocations.
        @inbounds for i in 1:m
            if rand() > p_rec
                C[i, :] .= A[i, :]
            else
                C[i, :] .= B[i, :]
            end
        end

        return C
    end

    """
        create_offspring(pop::artificial_pop, activation, distance, params::Dict) 
            -> Tuple{
                Vector{Matrix{Float64}},  # offspring
                Vector{Float64},          # fitness
                Vector{Any},              # steps
                Int,                      # completion_gen
                Matrix{Int}               # parents
            }

    Generates a new generation of offspring matrices from an existing `artificial_pop`
    using recombination, mutation, and fitness-based selection.

    # Arguments
    - `pop::artificial_pop`: Population containing individuals and their properties.
    - `activation`: Activation function used during development.
    - `distance`: Distance metric for computing fitness.
    - `params::Dict`: Dictionary with the following keys:
        - `"s"::Float64`: Selection strength.
        - `"mr"::Float64`: Mutation rate for regulatory weights.
        - `"σr"::Float64`: Standard deviation of weight mutations.
        - `"pr"::Float64`: Probability of regulatory weight mutation.
        - `"unstable_fitness"::Float64`: Fitness assigned to unstable phenotypes.
        - `"p_rec"::Float64`: Probability of recombination per row.
        - `"pc"::Float64`: Probability of connectivity mutation.
        - `"noise_prob"::Float64`: Probability of noise applied to weights.
        - `"noise_dist"`: Distribution from which noise is drawn.
        - `"max_steps"::Int`: Maximum number of steps to attempt reaching a stable state.

    # Returns
    A tuple containing:
    1. `Vector{Matrix{Float64}}`: Offspring weight matrices.
    2. `Vector{Float64}`: Fitness of each offspring.
    3. `Vector{Union{Int,Nothing}}`: Number of steps each offspring took to reach a stable state (`nothing` if unstable).
    4. `Int`: Number of offspring that reached stability (`completion_gen`).
    5. `Matrix{Int}`: Parent indices for each offspring (`pop_size × 2`).

    # Notes
    - Offspring are accepted into the new generation with a probability equal to their computed fitness.
    - Stability is determined by the `develop` function; a stable phenotype is any non-`nothing` return value.
    - Noise is applied after mutation but before development.
    """

    function create_offspring(pop::artificial_pop, activation,distance, params)

        s = params["s"]
        mr = params["mr"]
        σr = params["σr"]
        pr = params["pr"]
        unstable_fitness = params["unstable_fitness"]
        p_rec = params["p_rec"]
        pc = params["pc"]
        noise_prob = params["noise_prob"]
        noise_dist = params["noise_dist"]
        max_steps = params["max_steps"]

        pop_size = pop.pop_size
        phenotypic_optima = pop.phenotypic_optima
        initial_state = pop.initial_state
        matrices = pop.matrices
        N_target = pop.N_target
        N_genes = pop.N_target + pop.N_regulator

        survival = false 

        # Store offspring matrices
        offspring = Vector{Matrix{Float64}}(undef, pop_size)

        # Measures
        fitness = Vector{Float64}(undef,pop_size)
        completion_gen = 0
        steps = Vector{Union{Int,Nothing}}(undef,pop_size)
        parents = Matrix{Int}(undef,pop_size, 2)
        noisy_W = Matrix{Float64}(undef, N_genes, N_genes)

        # Cache Normal distribution
        d_norm = Normal(mr, σr)

        for i in 1:pop_size
            survival = false 
            while !survival 
                parent_i, parent_j = rand(1:pop_size, 2)
                # recombine
                    W_candidate = recombine_rows(matrices[parent_i], matrices[parent_j], p_rec)
                
                # mutate 
                reg_mutation!(W_candidate, pr, d_norm)
                
                # Mutate connectivity of W_candidate (use cached distribution)
                con_mutation!(W_candidate, pc, d_norm)

                # Make noise
                copyto!(noisy_W,W_candidate)
                apply_noise!(noisy_W,noise_prob,noise_dist)

                # find stable state
                phenotype, path_length = develop(noisy_W, initial_state, max_steps, activation)
                # phenotype, path_length = develop_asynchronous(noisy_W, initial_state, max_steps, activation)

                # compute fitness
                fit = indiv_fitness(phenotype, phenotypic_optima, N_target, s, distance, unstable_fitness)
                
                # decide if the offspring survives
                if rand() < fit
                    offspring[i] = W_candidate
                    fitness[i] = fit
                    steps[i] = path_length
                    if phenotype !== nothing
                        completion_gen += 1
                    end
                    survival = true
                    parents[i,:] .= (parent_i,parent_j)
                end
            end 
        end 
        return offspring, fitness, steps, completion_gen, parents
    end

    """
    -----------------------
    EVOLUTIONARY ALGORITHM
    -----------------------
    """
    
    """
        run_simulation(parameters::Dict{String,Any}; distance::Function=hamming_distance)::Dict{String,Any}

    Simulate the evolution of gene regulatory networks (GRNs) using Wagner's algorithm with noise
    extensions. The simulation evolves a population of GRNs through selection, mutation, and
    recombination, measuring network stability and phenotype expression over generations.

    # Arguments
    - `parameters::Dict{String,Any}`: Simulation parameters, merged with STANDARD_PARAMETERS.
        Required keys:
        - `"G"::Int`: Number of generations to simulate
        - `"pop_size"::Int`: Population size
        - `"N_target"::Int`: Number of target genes
        - `"N_regulator"::Int`: Number of non-target genes
        Network parameters:
        - `"c"::Float64`: Initial matrix density ∈ [0,1]
        - `"mr"::Float64`: Mean of weight distribution
        - `"σr"::Float64`: Standard deviation of weight distribution
        - `"max_steps"::Int`: Maximum steps before declaring instability
        Evolution parameters:
        - `"pr"::Float64`: Regular mutation probability ∈ [0,1]
        - `"p_rec"::Float64`: Recombination probability ∈ [0,1]
        - `"s"::Float64`: Selection pressure (fitness scaling)
        Initial state parameters:
        - `"p_init"::Float64`: Proportion of +1 in initial state ∈ [0,1]
        - `"p_phen"::Float64`: Proportion of +1 in target phenotype ∈ [0,1]
        - `"mode"::String`: Initialization mode, one of:
            - "random": Random matrices without stability checks
            - "stable": Only stable matrices
            - "unstable": Only unstable matrices
            - "optimal clones": Population of identical stable matrices
            - "nonoptimal clones": Population of identical stable matrices
            - "ensemble sample": Matrices that reach target phenotype
        Noise parameters:
        - `"noise_prob"::Float64`: Probability of noise per weight ∈ [0,1]
        - `"noise_dist"::Distribution`: Distribution for multiplicative noise

    # Optional Arguments
    - `distance::Function=hamming_distance`: Distance metric for phenotype comparison

    # Returns
    Dictionary with simulation results:
    - `"matrices"::Array{Matrix{Float64},2}`: Weight matrices, shape (G, pop_size)
    - `"fitness"::Matrix{Float64}`: Individual fitness values, shape (G, pop_size)
    - `"path_length"::Matrix{Union{Int,Nothing}}`: Steps to stability, shape (G, pop_size)
    - `"completion"::Vector{Float64}`: Fraction stable per generation, length G
    - `"initial_state"::Vector{Int}`: Initial gene expression state
    - `"phenotypic_optima"::Vector{Int}`: Target phenotype vector

    # References
    Wagner, A. (1996). Does evolutionary plasticity evolve? Evolution, 50(3), 1008-1023.
    """

    function run_simulation(parameters::Dict; distance::Function=hamming_distance)::Dict{String,Any}

        # Merge supplied parameters with STANDARD_PARAMETERS to ensure sensible defaults
        p = merge(STANDARD_PARAMETERS, parameters)

        # PARAMETER ASSIGNMENT (local, clearer names)
        G = p["G"]
        pop_size = p["pop_size"]
        N_target = p["N_target"]
        N_regulator = p["N_regulator"]
        N_genes = N_target + N_regulator
        p_init = p["p_init"]
        p_phen = p["p_phen"]
        max_steps = p["max_steps"]
        s = p["s"]
        unstable_fitness = p["unstable_fitness"]

        noise_prob = p["noise_prob"]
        noise_dist = p["noise_dist"]

        # INITIALIZATION
        initial_state, phenotypic_optima, matrices = initialize_population(parameters, make_initial_state,make_optimal_phenotype, activation)
        population = artificial_pop(
                pop_size = pop_size,
                N_regulator = N_regulator,
                N_target = N_target,
                matrices = matrices,
                initial_state = initial_state,
                phenotypic_optima = phenotypic_optima
            )

        # MEASUREMENTS DECLARATIONS

        ## __ individual measures __ 
        fitness_history = Matrix{Float64}(undef, G,pop_size)
        path_length_history = Matrix{Any}(undef, G,pop_size)
        matrices_history = Array{Matrix{Float64}}(undef, G, pop_size)

        ## __ aggregate measures __
        completion = zeros(G)
        
        noisy_W = Matrix{Float64}(undef, N_genes, N_genes) # preallocate memory

        # Helper: record generation results 
        function record_generation!(gen::Int, offspring::Vector{Matrix{Float64}}, fit::Vector{Float64}, steps::Vector{Union{Int,Nothing}}, completion_gen::Int,
                matrices_history::Array{Matrix{Float64}}, fitness_history::Matrix{Float64}, path_length_history::Matrix{Any}, completion::Vector{Float64})

            matrices_history[gen,:] .= offspring
            fitness_history[gen,:] .= fit
            path_length_history[gen,:] .= steps
            completion[gen] = completion_gen / size(fit, 1)

            return nothing
        end

        # Initial population measurements
        for (index, matrix) in enumerate(population.matrices)
            copyto!(noisy_W, matrix) # copy contents
            apply_noise!(noisy_W, noise_prob, noise_dist)
            phenotype, path_length = develop(noisy_W, population.initial_state, max_steps, activation)
            # phenotype, path_length = develop_asynchronous(noisy_W, population.initial_state, max_steps, activation)

            fit = indiv_fitness(phenotype, phenotypic_optima, N_target, s, distance, unstable_fitness)
            fitness_history[1,index] = fit
            if phenotype !== nothing # how many stable phenotypes there are
                completion[1] += 1 / pop_size
            end
            path_length_history[1,index] = path_length
            matrices_history[1,index] = matrix
        end

        # RUN SIMULATION
        for gen in 2:G

            completion_gen = 0
            # compute the next generation (recombination, mutation, and fitness survival are implicit)
            offspring, fit, steps, completion_gen, parents = create_offspring(population, activation, distance, parameters)
            
            # store historic measures
            record_generation!(gen, offspring, fit, steps, completion_gen,
                            matrices_history, fitness_history, path_length_history, 
                            completion)

            # update matrices
            replace_matrices!(population, offspring)
        end

        data = Dict("matrices"  => matrices_history,
                    "fitness" => fitness_history,
                    "path_length" => path_length_history,
                    "completion" => completion,
                    "initial_state" => population.initial_state,
                    "phenotypic_optima" => population.phenotypic_optima) #TODO - might change data type with changing environments

        return data
    end
end