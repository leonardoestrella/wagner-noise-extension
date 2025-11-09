"""
BooleanNetwork

A module implementing Wagner's GRN evolution algorithm with noise extensions.
This implementation focuses on memory efficiency and performance while maintaining
the exact evolutionary dynamics of the original algorithm.
"""
module BooleanNetwork

using LinearAlgebra
using Random
using Distributions
using Parameters
using StatsBase

export run_simulation

# Type aliases for improved performance and code clarity
const NetworkMatrix = Matrix{Float64}    # Weight matrix type
const GeneState = Union{Vector{Int64}, Vector{Float64}}         # Gene expression state type
const Population = Vector{NetworkMatrix} # Collection of networks
const NoiseDistribution = Union{Distribution{Univariate,Continuous},Distribution{Univariate,Discrete}} # Distribution for noise

    """
    --------------------
    STANDARD PARAMETERS
    --------------------
    """

    const STANDARD_PARAMETERS = Dict{String,Any}(
        "G" => 500, # Number of generations per run
        "max_steps" => 100, # Maximum number of steps before declaring a state unstable
        "s" => 10, # selection pressure
        "unstable_fitness" => exp(-10), # fitness for unstable matrices
        "mode" => "stable", # initialization mode 
        "pop_size" => 300,
        "N_target" => 10, # Number of target genes
        "N_regulator" => 0, # Number of non-target genes (not used, yet)
        "c" => 1.0, # initial matrix density
        "p_init" => 0.5, # proportion of +1 in initial gene expression state
        "p_phen" => 0.5, # proportion of +1 in target optimal phenotype
        "mr" => 0.0,"σr" => 1.0, # weights distribution parameters
        "pr" => 0.01, # regular mutation probability
        "pc" => 0.0, # connectivity mutation probability (deprecated)
        "p_rec" => 0.5, # 0 for no recombination, 0.5 for unbiased recombination
        "noise_prob" => 1.0, # probability of each weight to be affected by noise
        "noise_dist" => Bernoulli(1.0) # no noise
    )

    """
    --------------------
    DYNAMICS
    -------------------
    """

    """
        activation(x::AbstractVecOrMat{<:Real}) -> AbstractVecOrMat{Float64}

    Vectorized activation function that converts continuous values to discrete states {-1,1}.

    # Arguments
    - `x`: Input vector or matrix of real numbers

    # Returns
    - Vector or matrix of same shape as input with values ∈ {-1.0, 1.0}
    """
    function activation(x::AbstractVecOrMat{<:Real})
        return sign.(x)
    end

    """
        hamming_distance(v1::AbstractVector, v2::AbstractVector, N_target::Integer) -> Float64

    Compute normalized Hamming distance between two gene state vectors.

    # Arguments
    - `v1`, `v2`: Vectors containing gene states (1 or -1)
    - `N_target`: Number of target genes to consider

    # Returns
    - Normalized distance in [0,1] where 0 means identical and 1 means opposite
    """
    function hamming_distance(v1::AbstractVector, v2::AbstractVector, N_target::Integer)
        @assert length(v1) >= N_target && length(v2) >= N_target "Vectors must be at least N_target long"
        # Use views to avoid allocation and enable SIMD
        v1_view = @view v1[1:N_target]
        v2_view = @view v2[1:N_target]
        # Vectorized dot product
        matching_genes = dot(v1_view, v2_view)
        return (N_target - matching_genes) / (2 * N_target)
    end

    """
        develop(W::NetworkMatrix, 
                initial_state::GeneState, 
                max_steps::Integer,
                activation::Function;
                buffer1::Vector{Float64}=Vector{Float64}(undef, size(W,1)),
                buffer2::Vector{Float64}=Vector{Float64}(undef, size(W,1))
                ) -> Tuple{Union{Vector{Float64},Nothing}, Union{Int,Nothing}}

    Develop network phenotype through iterative matrix multiplication until stability
    or max_steps reached. 

    # Arguments
    - `W`: Weight matrix of the network
    - `initial_state`: Initial gene expression state
    - `max_steps`: Maximum iterations before declaring instability
    - `activation`: Activation function to apply at each step
    - `buffer1`, `buffer2`: Pre-allocated buffers for intermediate states

    # Returns
    - Tuple of (final_state, steps_taken) or (nothing, nothing) if unstable

    """
    function develop(W::NetworkMatrix,
                    initial_state::GeneState,
                    max_steps::Integer,
                    activation::Function;
                    buffer1::Vector{Float64}=Vector{Float64}(undef, size(W,1)),
                    buffer2::Vector{Float64}=Vector{Float64}(undef, size(W,1)))
        
        # Initialize buffers with initial state
        copyto!(buffer1, float.(initial_state))
        
        # First iteration
        mul!(buffer2, W, buffer1)  # In-place matrix multiplication
        buffer2 .= activation(buffer2)  # In-place activation
        
        # Iterate until stability or max_steps
        for step in 1:max_steps
            if buffer1 == buffer2
                return buffer2, step
            end
            buffer1, buffer2 = buffer2, buffer1  # Swap buffers
            mul!(buffer2, W, buffer1)  # In-place matrix multiplication
            buffer2 .= activation(buffer2)  # In-place activation
        end
        
        return nothing, nothing
    end

    """
        apply_noise!(W::NetworkMatrix, 
                    noise_prob::Float64, 
                    noise_dist::NoiseDistribution) -> Nothing

    Apply multiplicative noise to nonzero elements of the weight matrix.
    Optimized to minimize memory allocations and maximize vectorization.

    # Arguments
    - `W`: Weight matrix to modify
    - `noise_prob`: Probability of applying noise to each nonzero element
    - `noise_dist`: Distribution to sample noise values from (E[η] ≈ 1)
    """
    function apply_noise!(W::NetworkMatrix, 
                        noise_prob::Float64, 
                        noise_dist::NoiseDistribution)
        # Early exit if no noise
        if noise_prob == 0.0
            return nothing
        end
        
        # Find nonzero elements
        nonzero_idxs = findall(!iszero, W)
        if isempty(nonzero_idxs)
            return nothing
        end
        
        n_elements = length(nonzero_idxs)

        # Optimize the common special-case where every nonzero element is affected
        if noise_prob >= 1.0
            noise_values = rand(noise_dist, n_elements)
            W[nonzero_idxs] .*= noise_values
            return nothing
        end

        # For partial-noise probability, generate boolean mask then apply only
        # to the selected subset to reduce unnecessary random draws.
        apply_noise = rand(n_elements) .< noise_prob
        if any(apply_noise)
            noise_values = rand(noise_dist, count(apply_noise))
            W[nonzero_idxs[apply_noise]] .*= noise_values
        end
        
        return nothing
    end

    """
    --------------------
    INITIALIZATION
    --------------------
    """

    """
        generate_random_matrix(N::Integer, mr::Real, σr::Real, c::Real) -> Matrix{Float64}

    Generate a random weight matrix for a gene regulatory network.
    
    # Arguments
    - `N`: Matrix dimensions (N × N)
    - `mr`: Mean of the normal distribution for weights
    - `σr`: Standard deviation of the normal distribution for weights
    - `c`: Connection probability (controls sparsity)
    
    # Returns
    - An N × N matrix with weights drawn from Normal(mr, σr) and masked by Bernoulli(c)
    """
    function generate_random_matrix(N::Integer, mr::Real, σr::Real, c::Real)::Matrix{Float64}
        # Pre-allocate result matrix
        result = zeros(Float64, N, N)
        
        # Cache distributions for repeated use
        norm_dist = Normal(mr, σr)
        
        # Generate all weights at once
        n_elements = N * N
        active_elements = rand(n_elements) .< c
        n_active = count(active_elements)
        
        if n_active > 0
            # Only generate random weights for active connections
            active_weights = rand(norm_dist, n_active)
            
            # Place weights in active positions
            result[active_elements] .= active_weights
        end
        
        return result
    end

    """
        check_stability(W::NetworkMatrix,
                    initial_state::GeneState,
                    max_steps::Integer,
                    activation::Function;
                    buffer1::Vector{Float64}=Vector{Float64}(undef, size(W,1)),
                    buffer2::Vector{Float64}=Vector{Float64}(undef, size(W,1))
                    ) -> Tuple{Bool, Union{Vector{Float64}, Nothing}, Union{Int, Nothing}}

    Check if a network reaches a stable state within max_steps.

    # Arguments
    - `W`: Weight matrix to check
    - `initial_state`: Initial state vector
    - `max_steps`: Maximum steps before declaring instability
    - `activation`: Activation function to use
    - `buffer1`, `buffer2`: Pre-allocated buffers for development

    # Returns
    - Tuple of:
        1. Boolean indicating stability
        2. Final phenotype (or nothing if unstable)
        3. Steps taken (or nothing if unstable)

    """
    function check_stability(W::NetworkMatrix,
                           initial_state::GeneState,
                           max_steps::Integer,
                           activation::Function;
                           buffer1::Vector{Float64}=Vector{Float64}(undef, size(W,1)),
                           buffer2::Vector{Float64}=Vector{Float64}(undef, size(W,1))
                           )::Tuple{Bool, Union{Vector{Float64}, Nothing}, Union{Int, Nothing}}
        phenotype, steps = develop(W, initial_state, max_steps, activation; 
                                 buffer1=buffer1, buffer2=buffer2)
        return phenotype !== nothing, phenotype, steps
    end


    """
        sample_binary_state(N::Int, p::Real) -> Vector{Int}

    Generic helper that returns a length-N vector with entries in {1, -1}.
    The value 1 is sampled with probability p and -1 with probability 1-p.

    This is the single implementation used by both `make_initial_state`
    and `make_optimal_phenotype` to avoid duplication.
    """
    function sample_binary_state(N::Int, p::Real)::Vector{Int}
        # Use small pre-allocated arrays for values and weights to avoid repeated allocations
        vals = [1, -1]
        w = Weights([p, 1 - p])
        return sample(vals, w, N)
    end


    """
        make_initial_state(params::Dict) -> Vector{Int}

    Wrapper around `sample_binary_state` using parameters in `params`.
    """
    function make_initial_state(params::Dict)
        N = params["N_target"] + params["N_regulator"]
        p_init = params["p_init"]
        return sample_binary_state(N, p_init)
    end

    """
        make_optimal_phenotype(params::Dict) -> Vector{Int}

    Wrapper around `sample_binary_state` using parameters in `params`.
    """
    function make_optimal_phenotype(params::Dict)
        N_target = params["N_target"]
        p_phen = params["p_phen"]
        return sample_binary_state(N_target, p_phen)
    end

    """
        initialize_population(params::Dict,
                            initial_state_generator::Function,
                            optimal_phen_generator::Function,
                            activation::Function
                            ) -> Tuple{GeneState, GeneState, Vector{Matrix{Float64}}}

    Initialize a population of gene regulatory networks according to specified mode.
    Uses pre-allocated buffers and optimized matrix generation.

    # Arguments
    - `params`: Dictionary of simulation parameters
    - `initial_state_generator`: Function to generate initial states
    - `optimal_phen_generator`: Function to generate optimal phenotype
    - `activation`: Activation function for development

    # Returns
    - Tuple of:
        1. Initial state vector
        2. Optimal phenotype vector
        3. Vector of weight matrices
    """
    function initialize_population(params::Dict,
                                initial_state_generator::Function,
                                optimal_phen_generator::Function,
                                activation::Function
                                )::Tuple{GeneState, GeneState, Vector{Matrix{Float64}}}
        
        # Extract parameters
        mode = params["mode"]
        pop_size = params["pop_size"]
        N_target = params["N_target"]
        N_regulator = params["N_regulator"]
        N = N_target + N_regulator
        mr = params["mr"]
        σr = params["σr"]
        c = params["c"]
        max_steps = params["max_steps"]

        # Generate states
        initial_state = initial_state_generator(params)
        optimal_phenotype = optimal_phen_generator(params)
        
        # Pre-allocate development buffers
        buffer1 = Vector{Float64}(undef, N)
        buffer2 = Vector{Float64}(undef, N)
        
        phenotype = nothing

        if mode == "random"
            # For random mode, just generate matrices without stability checks
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            for i in 1:pop_size
                matrices[i] = generate_random_matrix(N, mr, σr, c)
            end
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "stable"
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            
            # Find stable matrices using pre-allocated buffers
            for i in 1:pop_size
                while true
                    candidate = generate_random_matrix(N, mr, σr, c)
                    is_stable, phenotype, steps = check_stability(
                        candidate, initial_state, max_steps, activation;
                        buffer1=buffer1, buffer2=buffer2
                    )
                    
                    if is_stable
                        matrices[i] = copy(candidate)
                        break
                    end
                end
            end
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "unstable"
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            
            # Find unstable matrices using pre-allocated buffers
            for i in 1:pop_size
                while true
                    candidate = generate_random_matrix(N, mr, σr, c)
                    is_stable, phenotype, steps = check_stability(
                        candidate, initial_state, max_steps, activation;
                        buffer1=buffer1, buffer2=buffer2
                    )
                    
                    if !is_stable  # Want unstable matrices here
                        matrices[i] = copy(candidate)
                        break
                    end
                end
            end
            return (initial_state, optimal_phenotype, matrices)

        elseif mode == "optimal clones"
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            
            # Find a single stable matrix and clone it
            while true
                candidate = generate_random_matrix(N, mr, σr, c)
                is_stable, phenotype, steps = check_stability(
                    candidate, initial_state, max_steps, activation;
                    buffer1=buffer1, buffer2=buffer2
                )
                
                if is_stable
                    # Found a stable matrix - clone it for the population
                    for i in 1:pop_size
                        matrices[i] = copy(candidate)
                    end
                    return (initial_state, phenotype, matrices)  # Return the actual phenotype here
                end
            end

        elseif mode == "nonoptimal clones"
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            
            # Find a single stable matrix (non-optimal) and clone it
            while true
                candidate = generate_random_matrix(N, mr, σr, c)
                is_stable, phenotype, steps = check_stability(
                    candidate, initial_state, max_steps, activation;
                    buffer1=buffer1, buffer2=buffer2
                )
                
                if is_stable
                    # Found a stable matrix - clone it for the population
                    for i in 1:pop_size
                        matrices[i] = copy(candidate)
                    end
                    return (initial_state, optimal_phenotype, matrices)
                end
            end

        elseif mode == "ensemble sample"
            matrices = Vector{Matrix{Float64}}(undef, pop_size)
            population_count = 0
            attempts = 0
            attempt_limit = 5 * 2^(2*N)  # Max attempts before restart
            
            while population_count < pop_size
                candidate = generate_random_matrix(N, mr, σr, c)
                is_stable, phenotype, steps = check_stability(
                    candidate, initial_state, max_steps, activation;
                    buffer1=buffer1, buffer2=buffer2
                )
                
                if is_stable && phenotype == optimal_phenotype
                    # Found a matrix with the desired phenotype
                    population_count += 1
                    matrices[population_count] = candidate
                    attempts = 0
                else
                    attempts += 1
                end
                
                # Reset if we've tried too many times
                if attempts > attempt_limit
                    population_count = 0
                    attempts = 0
                    initial_state = initial_state_generator(params)
                    optimal_phenotype = optimal_phen_generator(params)
                end
            end
            
            return (initial_state, optimal_phenotype, matrices)
        else
            error("Unknown mode: $mode")
        end 
    end

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
                    "phenotypic_optima" => population.phenotypic_optima)

        return data
    end
end