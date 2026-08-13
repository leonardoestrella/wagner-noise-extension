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

@testset "run_simulation" begin
    params = BooleanNetwork.SimulationParameters(generations=20, pop_size=20, number_genes=5)
    result = BooleanNetwork.run_simulation(params)
    
    # Check that dimensions are correct
    @test length(result.completion_history) == params.generations
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
end

"""
Missing tests:
- The simulation runs with all initial types of population
- The simulation runs with all types of selection 
-> Note they don't have to be combinatorially tested. They are independent
- The simulation runs with many values of initial_density, mutation_prob, selection_pressure,
    unstable_fitness, and parametrizations of weights_dist and noise_dist
"""

println("test_core.jl: all tests completed")
