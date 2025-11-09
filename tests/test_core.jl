include("../src/wagner_algorithm.jl")
using Random
using Distributions

# For reproducibility
Random.seed!(12345)

# Test generate_random_matrix
@testset "generate_random_matrix" begin
    # Test dimensions
    N = 10
    W = BooleanNetwork.generate_random_matrix(N, 0.0,1.0,1.0)
    @test size(W) == (N, N)
    
    # Test connection density
    conn_density = 0.3
    W_sparse = BooleanNetwork.generate_random_matrix(N, 0.0, 1.0, conn_density)
    zero_count = count(x -> x == 0, W_sparse)
    expected_zeros = round(Int, N * N * (1 - conn_density))
    @test abs(zero_count - expected_zeros) ≤ N # Allow small random variation
    
    # Test weight distribution
    μ = 1.0
    σ = 0.5
    W_dist = BooleanNetwork.generate_random_matrix(N, μ, σ, 1.0)
    nonzero_weights = filter(x -> x != 0, vec(W_dist))
    @test mean(nonzero_weights) ≈ μ atol=0.5
    @test std(nonzero_weights) ≈ σ atol=0.3
end

# Test check_stability
@testset "check_stability" begin
    # Test stable state
    N = 5
    W = [1.0 0.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0 0.0;
         0.0 0.0 1.0 0.0 0.0;
         0.0 0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 0.0 1.0]
    s = ones(N)
    max_steps = 10 
    activation = x -> sign.(x)

    # Test stable state
    @test BooleanNetwork.check_stability(W, s, max_steps,activation)[1]
    
    # Test unstable state
    W_unstable = -W
    @test !BooleanNetwork.check_stability(W_unstable, s, max_steps, activation)[1]
end

# Test mutation and noise operations 
@testset "mutation_operations" begin
    N = 5
    W = ones(N, N)
    μ = 0.0
    σ = 1.0
    
    # Test point mutations
    W_point = copy(W)
    BooleanNetwork.reg_mutation!(W_point, 0.5, Distributions.Normal(μ, σ))
    @test any(x -> x != 1.0, W_point) # Some elements should be mutated
    
    # Test turning off mutation: nothing should change when p=0.0
    W_point_off = copy(W)
    BooleanNetwork.reg_mutation!(W_point_off, 0.0, Distributions.Normal(μ, σ))
    @test all(x -> x == 1.0, W_point_off)

    # Test mutating all elements of W (p=1.0): every element should be changed
    W_point_all = copy(W)
    BooleanNetwork.reg_mutation!(W_point_all, 1.0, Distributions.Normal(μ, σ))
    @test all(x -> x != 1.0, vec(W_point_all))

    # Test noise application
    W_noise = copy(W)
    BooleanNetwork.apply_noise!(W_noise, 0.5, Distributions.Gamma(1.0,1.0))
    @test any(x -> x != 1.0, W_noise)
    # The applied noise distribution (Gamma) is positive, so signs should be preserved
    @test all(sign.(W_noise) .== sign.(W))
    
    # Test turning off noise
    W_noise_off = copy(W)
    BooleanNetwork.apply_noise!(W_noise_off, 0.0, Distributions.Gamma(1.0,1.0))
    @test all(x -> x == 1.0, W_noise_off)
    # Test putting noise in all elements of W
    W_noise_all = copy(W)
    BooleanNetwork.apply_noise!(W_noise_all, 1.0, Distributions.Gamma(1.0,1.0))
    @test all(x -> x != 1.0, vec(W_noise_all))

    # Test recombination
    A = ones(N, N)
    B = -ones(N, N)
    C = BooleanNetwork.recombine_rows(A, B, 0.5)
    @test any(row -> all(x -> x == 1.0, row), eachrow(C))  # Some rows from A
    @test any(row -> all(x -> x == -1.0, row), eachrow(C)) # Some rows from B

    # Test turning off recombination 
    # Turning off recombination returns A; full recombination returns B
    C_off = BooleanNetwork.recombine_rows(A, B, 0.0)
    C_full = BooleanNetwork.recombine_rows(A, B, 1.0)
    @test all(C_off .== A)
    @test all(C_full .== B)
end

# Test full simulation
@testset "run_simulation" begin
    # Basic simulation parameters

    params = BooleanNetwork.STANDARD_PARAMETERS
    N = 5
    P = 10
    G = 20
    params["N_target"] = N
    params["pop_size"] = P
    params["G"] = G
    
    # Run simulation with defaults
    result = BooleanNetwork.run_simulation(
    params
    )
    
    # 1. Check that the container's dimensions are correct
    @test size(result["matrices"]) == (G, P)

    # 2. Check that every matrix inside the container is of size (N, N)
    @test all(W -> size(W) == (N, N), result["matrices"])

    # Test with noise
    params["noise_prob"] = 1.0
    params["noise_dist"] = Gamma(1.0, 1.0)
    result_noise = BooleanNetwork.run_simulation(
        params
    )

    @test size(result_noise["matrices"]) == (G, P)
    @test all(W -> size(W) == (N, N), result_noise["matrices"])
end

println("test_core.jl: all tests completed")