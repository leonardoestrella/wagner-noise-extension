include("../src/wagner_algorithm.jl")
using Random
using Distributions
using LinearAlgebra
using Test

Random.seed!(12345)  # For reproducibility

@testset "generate_random_matrix" begin 
    N = 10
    weights_dist = Normal(0.0,1.0)
    W = BooleanNetwork.generate_random_matrix(N, weights_dist, 1.0)
    @test size(W) == (N, N)  # Test dimensions
    @test count(x -> x == 0, W) == 0  # Test that generated a fully-connected matrix
    
    conn_density = 0.3
    W_sparse = BooleanNetwork.generate_random_matrix(N, weights_dist, conn_density)
    zero_count = count(x -> x == 0, W_sparse)
    expected_zeros = round(Int, N * N * (1 - conn_density))
    @test abs(zero_count - expected_zeros) ≤ N # Allow small random variation
    
    μ = 1.0
    σ = 0.5
    alt_weights_dist = Normal(μ, σ)

    W_dist = BooleanNetwork.generate_random_matrix(N, alt_weights_dist, 1.0)
    nonzero_weights = filter(x -> x != 0, vec(W_dist))
    @test mean(nonzero_weights) ≈ μ atol=0.5
    @test std(nonzero_weights) ≈ σ atol=0.3  # Test other weight distributions

    @test_throws DomainError BooleanNetwork.generate_random_matrix(N, weights_dist, -0.1)
    @test_throws DomainError BooleanNetwork.generate_random_matrix(N, weights_dist, 1.1)
end

@testset "hamming_distance" begin
    v1 = [1.0, 1.0, 1.0, 1.0]
    v2 = [1.0, 1.0, 1.0, 1.0]
    @test BooleanNetwork.hamming_distance(v1, v2) == 0.0  # Identical vectors

    v3 = [-1.0, -1.0, -1.0, -1.0]
    @test BooleanNetwork.hamming_distance(v1, v3) == 1.0  # Opposite vectors

    v4 = [1.0, 1.0, -1.0, -1.0]
    @test BooleanNetwork.hamming_distance(v1, v4) == 0.5  # Half differ

    @test_throws DimensionMismatch BooleanNetwork.hamming_distance([1.0, -1.0], [1.0, -1.0, 1.0])
end

@testset "develop" begin
    # Fixed point: identity matrix keeps the state unchanged
    N = 4
    W_identity = Matrix{Float64}(I, N, N)
    state = [1.0, -1.0, 1.0, -1.0]
    final_state, steps = BooleanNetwork.develop(W_identity, state, 10)
    @test final_state == state
    @test steps == 0

    # Limit cycle / non-convergence: pure sign-flip oscillates forever
    W_flip = [0.0 -1.0; -1.0 0.0]
    flip_state = [1.0, 1.0]
    final_flip, steps_flip = BooleanNetwork.develop(W_flip, flip_state, 10)
    @test final_flip === nothing
    @test steps_flip === nothing

    # max_steps exhausted without stabilizing (same oscillation, tiny budget)
    final_short, steps_short = BooleanNetwork.develop(W_flip, flip_state, 1)
    @test final_short === nothing
    @test steps_short === nothing

    # Dimension errors
    non_square = ones(2, 3)
    @test_throws DimensionMismatch BooleanNetwork.develop(non_square, [1.0, -1.0], 10)
    @test_throws DimensionMismatch BooleanNetwork.develop(W_identity, [1.0, -1.0], 10)
    @test_throws DimensionMismatch BooleanNetwork.develop(
        W_identity, state, 10; buffer1=Vector{Float64}(undef, 2))
    @test_throws DimensionMismatch BooleanNetwork.develop(
        W_identity, state, 10; buffer2=Vector{Float64}(undef, 2))
end

@testset "indiv_fitness" begin
    optimal = [1.0, 1.0, 1.0, 1.0]
    selection_pressure = 5.0
    unstable_fitness = exp(-10.0)

    @test BooleanNetwork.indiv_fitness(optimal, optimal, selection_pressure, unstable_fitness) ≈ 1.0
    # Unstable phenotype returns the baseline fitness
    @test BooleanNetwork.indiv_fitness(nothing, optimal, selection_pressure, unstable_fitness) == unstable_fitness

    half_off = [1.0, 1.0, -1.0, -1.0]
    expected = exp(-selection_pressure * BooleanNetwork.hamming_distance(half_off, optimal))
    @test BooleanNetwork.indiv_fitness(half_off, optimal, selection_pressure, unstable_fitness) ≈ expected
end

### THIS HAS NOT BEEN IMPLEMENTED - AUG 13, 2026
# @testset "develop_asynchronous" begin  
#     activation = sign

#     # 1) Basic behavior: asynchronous updates should reach the expected fixed point
#     W = [0.0 2.0;
#          2.0 0.0]
#     initial_state = [1, -1]
#     final_state, sweeps = BooleanNetwork.develop_asynchronous(W, initial_state, 10, activation)
#     @test final_state !== nothing
#     @test sweeps == 2
#     @test final_state == [-1.0, -1.0]

#     # 2) Termination case: identity matrix with all-ones state should stabilize immediately
#     N = 6
#     W_identity = Matrix{Float64}(I, N, N)
#     ones_state = ones(Int, N)
#     final_identity, sweeps_identity = BooleanNetwork.develop_asynchronous(
#         W_identity, ones_state, 100, activation
#     )
#     @test final_identity !== nothing
#     @test sweeps_identity == 1
#     @test final_identity == ones(Float64, N)

#     # 3) Input immutability: function must not mutate initial_state
#     immutable_check_state = [1, -1, 1, -1]
#     original_copy = copy(immutable_check_state)
#     _ = BooleanNetwork.develop_asynchronous(Matrix{Float64}(I, 4, 4), immutable_check_state, 5, activation)
#     @test immutable_check_state == original_copy
# end

@testset "mutation!" begin
    N = 5
    W = ones(N, N)
    μ = 0.0
    σ = 1.0
    weights_dist = Normal(μ, σ)
    
    W_point = copy(W)
    BooleanNetwork.mutation!(W_point, 0.5, weights_dist)
    @test any(x -> x != 1.0, W_point)  # Some elements should be mutated
    
    W_point_off = copy(W)
    BooleanNetwork.mutation!(W_point_off, 0.0, weights_dist)
    @test all(x -> x == 1.0, W_point_off)  # No element should change

    W_point_all = copy(W)
    BooleanNetwork.mutation!(W_point_all, 1.0, weights_dist)
    @test all(x -> x != 1.0, vec(W_point_all))  # Every element should change
end

@testset "apply_noise!" begin
    N = 5
    W = ones(N, N)
    noise_dist = Gamma(1.0,1.0)

    W_noise = copy(W)
    BooleanNetwork.apply_noise!(W_noise, noise_dist)
    @test all(x -> x != 1.0, W_noise)  # All weights should have changed
    @test all(sign.(W_noise) .== sign.(W))  # No signs should have been changed
    
    W_noise_off = copy(W)
    BooleanNetwork.apply_noise!(W_noise_off, Bernoulli(1.0))
    @test all(x -> x == 1.0, W_noise_off)  # No noise was applied
end

@testset "recombine_rows" begin
    N = 10
    A = ones(N, N)
    B = -ones(N, N)
    C = BooleanNetwork.recombine_rows(A, B)
    @test any(row -> all(x -> x == 1.0, row), eachrow(C))  # Some rows from A
    @test any(row -> all(x -> x == -1.0, row), eachrow(C)) # Some rows from B

    C_off_A = BooleanNetwork.recombine_rows(A, A)
    C_off_B = BooleanNetwork.recombine_rows(B, B)
    @test all(C_off_A .== A)
    @test all(C_off_B .== B)
end

@testset "initialize_population" begin
    N = 5
    weights_dist = Normal(0.0, 1.0)
    max_steps = 100

    for pop_type in BooleanNetwork.valid_initial_pop_types
        params = BooleanNetwork.SimulationParameters(number_genes=N, pop_size=10,
            max_steps=max_steps, weights_dist=weights_dist, initial_pop_type=pop_type)
        initial_state, optimal_phenotype, matrices = BooleanNetwork.initialize_population(params)

        @test length(initial_state) == N
        @test length(optimal_phenotype) == N
        @test length(matrices) == params.pop_size
        @test all(W -> size(W) == (N, N), matrices)

        if pop_type == :stable
            @test all(W -> !isnothing(BooleanNetwork.develop(W, initial_state, max_steps)[1]),
                matrices)
        elseif pop_type == :unstable
            @test all(W -> isnothing(BooleanNetwork.develop(W, initial_state, max_steps)[1]),
                matrices)
        elseif pop_type == :optimal_clones
            @test all(W -> W == matrices[1], matrices)  # All clones identical
            expressed, _ = BooleanNetwork.develop(matrices[1], initial_state, max_steps)
            @test expressed == optimal_phenotype
        elseif pop_type == :nonoptimal_clones
            @test all(W -> W == matrices[1], matrices)  # All clones identical
        elseif pop_type == :ensemble_sample
            @test all(matrices) do W
                phenotype, steps = BooleanNetwork.develop(W, initial_state, max_steps)
                !isnothing(steps) && phenotype == optimal_phenotype
            end
        end
    end

    bad_params = BooleanNetwork.SimulationParameters(initial_pop_type=:not_a_type)
    @test_throws ArgumentError BooleanNetwork.initialize_population(bad_params)
end

@testset "create_offspring" begin
    N = 5
    pop_size = 10
    weights_dist = Normal(0.0, 1.0)

    for selection_type in BooleanNetwork.valid_selection_types
        params = BooleanNetwork.SimulationParameters(number_genes=N, pop_size=pop_size,
            weights_dist=weights_dist, selection_type=selection_type)
        matrices = [BooleanNetwork.generate_random_matrix(N, weights_dist, 1.0)
            for _ in 1:pop_size]
        initial_state = rand([1.0, -1.0], N)
        optimal_phenotype = rand([1.0, -1.0], N)
        pop = BooleanNetwork.ArtificialPop(pop_size=pop_size, number_genes=N,
            matrices=matrices, initial_state=initial_state, optimal_phenotype=optimal_phenotype)

        offspring, fitness, steps, completion_gen = BooleanNetwork.create_offspring(pop, params)

        @test length(offspring) == pop_size
        @test all(W -> size(W) == (N, N), offspring)
        @test length(fitness) == pop_size
        @test all(f -> 0.0 <= f <= 1.0, fitness)
        @test length(steps) == pop_size
        @test length(completion_gen) == pop_size
    end

    bad_params = BooleanNetwork.SimulationParameters(selection_type=:not_a_type)
    matrices = [BooleanNetwork.generate_random_matrix(5, weights_dist, 1.0) for _ in 1:5]
    pop = BooleanNetwork.ArtificialPop(pop_size=5, number_genes=5, matrices=matrices,
        initial_state=rand([1.0, -1.0], 5), optimal_phenotype=rand([1.0, -1.0], 5))
    @test_throws ArgumentError BooleanNetwork.create_offspring(pop, bad_params)
end

@testset "run_simulation" begin
    params = BooleanNetwork.SimulationParameters(generations=20, pop_size=20, number_genes=5)
    result = BooleanNetwork.run_simulation(params)

    # Check that dimensions are correct
    @test size(result.completion_history) == (params.generations, params.pop_size)
    @test size(result.fitness_history) == (params.generations, params.pop_size)
    @test size(result.matrices_history) == (params.generations, params.pop_size)
    @test size(result.path_length_history) == (params.generations, params.pop_size)
    @test all(W -> size(W) == (params.number_genes, params.number_genes),
        result.matrices_history)

    # Test with noise
    params = BooleanNetwork.SimulationParameters(generations=20, pop_size=20,
        number_genes=5, noise_dist=Gamma(1.0,1.0))
    result_noise = BooleanNetwork.run_simulation(params)

    @test size(result_noise.matrices_history) == (params.generations, params.pop_size)
    @test all(W -> size(W) == (params.number_genes, params.number_genes),
        result_noise.matrices_history)

    # All initial population types
    for pop_type in BooleanNetwork.valid_initial_pop_types
        params_pop = BooleanNetwork.SimulationParameters(generations=5, pop_size=10,
            number_genes=5, initial_pop_type=pop_type)
        result_pop = BooleanNetwork.run_simulation(params_pop)
        @test size(result_pop.matrices_history) == (params_pop.generations, params_pop.pop_size)
    end

    # All selection types
    for selection_type in BooleanNetwork.valid_selection_types
        params_sel = BooleanNetwork.SimulationParameters(generations=5, pop_size=10,
            number_genes=5, selection_type=selection_type)
        result_sel = BooleanNetwork.run_simulation(params_sel)
        @test size(result_sel.matrices_history) == (params_sel.generations, params_sel.pop_size)
    end
end

println("test_core.jl: all tests completed")
