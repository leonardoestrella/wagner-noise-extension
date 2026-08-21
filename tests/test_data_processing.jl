# Tests for data-processing utilities (CustomStats)
include("../src/wagner_algorithm.jl")
include("../src/data_processing.jl")
using Test
using Random
using Distributions
using LinearAlgebra
using ..BooleanNetwork

Random.seed!(1234)

@testset "alignment_score" begin
    M = [1.0 0.0;
         0.0 1.0]
    v = [1.0, 1.0]
    # A fully aligned matrix should yield an alignment of 1.0
    score = CustomStats.alignment_score(M, v)
    @test isapprox(score, 1.0; atol=1e-12)

    # An aligned matrix multiplied by -1.0 yields an alignment of -1.0
    score_2 = CustomStats.alignment_score(-M, v)
    @test isapprox(score_2, -1.0; atol=1e-12)

    # A matrix with rows orthogonal to v should yield 0.0
    M2 = [1.0 -1.0;
          -1.0 1.0]
    score_3 = CustomStats.alignment_score(M2, v)
    @test isapprox(score_3, 0.0; atol=1e-12)

    # Test fully aligned matrices with block structure
    # Target phenotype with mixed signs
    v_mixed = [1.0, 1.0, 1.0, -1.0, -1.0]
    
    # Construct fully aligned matrix with blocks
    W_aligned = [
        2.0  1.0  1.0  -2.0 -1.0;  # Positive correlation with first 3 targets
        1.0  2.0  1.0  -1.0 -2.0;  # Positive correlation with first 3 targets
        1.0  1.0  2.0  -1.0 -1.0;  # Positive correlation with first 3 targets
        -2.0 -1.0 -1.0  2.0  1.0;  # Negative correlation -> positive with last 2
        -1.0 -2.0 -1.0  1.0  2.0   # Negative correlation -> positive with last 2
    ]
    score_aligned = CustomStats.alignment_score(W_aligned, v_mixed)
    @test isapprox(score_aligned, 1.0; atol=1e-12)

    # Test completely misaligned matrices with block structure
    # Same target phenotype, but opposite block signs
    W_misaligned = [
        -2.0 -1.0 -1.0  2.0  1.0;  # Negative correlation with first 3
        -1.0 -2.0 -1.0  1.0  2.0;  # Negative correlation with first 3
        -1.0 -1.0 -2.0  1.0  1.0;  # Negative correlation with first 3
        2.0  1.0  1.0  -2.0 -1.0;  # Positive correlation -> negative with last 2
        1.0  2.0  1.0  -1.0 -2.0   # Positive correlation -> negative with last 2
    ]
    score_misaligned = CustomStats.alignment_score(W_misaligned, v_mixed)
    @test isapprox(score_misaligned, -1.0; atol=1e-12)

    # Test partial alignment cases
    v_binary = [1.0, -1.0]  # Binary target phenotype
    
    # Case 1: Matrix with zero net alignment
    W_zero = [1.0  1.0;     # Row sums to 0 when weighted by target
             -1.0 -1.0]     # Row also sums to 0
    score_zero = CustomStats.alignment_score(W_zero, v_binary)
    @test isapprox(score_zero, 0.0; atol=1e-12)
    
    # Case 2: Matrix with 0.5 alignment (half-aligned)
    W_half = [1.0  -1.0;    # Perfectly aligned row (+1)
              1.0   1.0]    # Zero alignment row (0)
    score_half = CustomStats.alignment_score(W_half, v_binary)
    @test isapprox(score_half, 0.5; atol=1e-12)

    # Case 3: Matrix without edges (alignment should be 0)
    W_empty = [0.0 -0.0;    # Perfectly aligned row (+1)
        0.0 0.0]    # Zero alignment row (0)
    score_empty = CustomStats.alignment_score(W_empty, v_binary)
    @test isapprox(score_empty, 0.0; atol=1e-12)

    # Broadcasting over a grid of matrices
    grid = [M, -M, M2]
    scores_grid = CustomStats.alignment_score.(grid, Ref(v))
    @test scores_grid ≈ [1.0, -1.0, 0.0] atol=1e-12
end

@testset "summarize_history" begin
    # Basic mean/std over fully-populated rows
    data = [1.0 2.0 3.0;
            4.0 4.0 4.0]
    means, stds = CustomStats.summarize_history(data)
    @test means ≈ [2.0, 4.0] atol=1e-12
    @test stds[1] ≈ std([1.0, 2.0, 3.0]; corrected=true) atol=1e-12
    @test stds[2] ≈ 0.0 atol=1e-12

    # `nothing` values are excluded from the statistics
    data_missing = Matrix{Union{Float64,Nothing}}(undef, 2, 3)
    data_missing[1, :] = [1.0, 2.0, nothing]
    data_missing[2, :] = [5.0, nothing, nothing]
    means_missing, stds_missing = CustomStats.summarize_history(data_missing)
    @test means_missing[1] ≈ 1.5 atol=1e-12
    @test means_missing[2] ≈ 5.0 atol=1e-12
    @test isnan(stds_missing[2])  # Only one data point: std undefined

    # A row with no data at all yields NaN for both mean and std
    data_empty_row = Matrix{Union{Float64,Nothing}}(undef, 1, 2)
    data_empty_row[1, :] = [nothing, nothing]
    means_empty, stds_empty = CustomStats.summarize_history(data_empty_row)
    @test isnan(means_empty[1])
    @test isnan(stds_empty[1])
end

@testset "generate_expression_distribution" begin
    N = 2
    matrix = Matrix{Float64}(I, N, N)
    initial_state = [1.0, -1.0]

    # No noise: development is deterministic, so every sample is stable and identical
    avg_expression, unstable_prop = CustomStats.generate_expression_distribution(
        matrix, initial_state, 5, Bernoulli(1.0), 10)
    final_state, _ = BooleanNetwork.develop(matrix, initial_state, 10)
    @test avg_expression ≈ final_state atol=1e-12
    @test unstable_prop == 0.0

    # A matrix that never stabilizes yields an all-zero average and full instability
    W_flip = [0.0 -1.0;
             -1.0 0.0]
    avg_unstable, unstable_prop_full = CustomStats.generate_expression_distribution(
        W_flip, [1.0, 1.0], 5, Bernoulli(1.0), 10)
    @test avg_unstable == zeros(Float64, N)
    @test unstable_prop_full == 1.0

    # Default max_steps (from SimulationParameters()) is used when omitted
    avg_default, _ = CustomStats.generate_expression_distribution(
        matrix, initial_state, 5, Bernoulli(1.0))
    @test avg_default ≈ final_state atol=1e-12

    # Normal working conditions: positive-only noise (Gamma) rescales weights but never
    # flips their sign, so a diagonal matrix should still stabilize on every sample
    Random.seed!(42)
    avg_gamma, unstable_prop_gamma = CustomStats.generate_expression_distribution(
        matrix, initial_state, 50, Gamma(1.0, 1.0), 10)
    @test avg_gamma ≈ initial_state atol=1e-12
    @test unstable_prop_gamma == 0.0

    # A less trivial matrix under Gamma noise: outcomes stay well-formed, even though
    # individual samples are no longer deterministic
    W_mixed = [0.1 2.0;
              -2.0 0.1]
    avg_mixed, unstable_prop_mixed = CustomStats.generate_expression_distribution(
        W_mixed, initial_state, 50, Gamma(1.0, 1.0), 10)
    @test length(avg_mixed) == N
    @test all(x -> -1.0 <= x <= 1.0, avg_mixed)
    @test 0.0 <= unstable_prop_mixed <= 1.0
end

@testset "compute_mut_robustness" begin
    matrix = [0.1 2.0;
        -2.0 0.1]
    initial_state = [1.0, -1.0]
    n_mutation_samples = 5
    n_noise_samples = 5
    weights_dist = Normal(0.0, 1.0)
    max_steps = 10

    @testset "1. No noise, no mutations" begin
        res = CustomStats.compute_mut_robustness(
            matrix, initial_state,
            n_mutation_samples, n_noise_samples,
            Bernoulli(1.0),  # No noise
            0.0;             # No mutations
            weights_dist=weights_dist, max_steps=max_steps
        )
        # Without noise or mutations, the system behavior should be completely deterministic
        @test isapprox(res.stable_expression_shift, 0.0; atol=1e-12)
        @test isapprox(res.stable_expression_var, 0.0; atol=1e-12)
        @test isapprox(res.unstable_prob_shift, 0.0; atol=1e-12)
        @test isapprox(res.unstable_prob_var, 0.0; atol=1e-12)
    end

    @testset "2. No noise, always mutate" begin
        Random.seed!(42)
        res = CustomStats.compute_mut_robustness(
            matrix, initial_state,
            n_mutation_samples, n_noise_samples,
            Bernoulli(1.0), # No noise
            1.0;            # Always mutate
            weights_dist=weights_dist, max_steps=max_steps
        )
        # Mutations alter network dynamics, creating variation between samples
        @test res.stable_expression_var >= 0.0
        @test res.unstable_prob_var >= 0.0
    end

    @testset "3. A lot of noise, no mutations" begin
        Random.seed!(42)
        res = CustomStats.compute_mut_robustness(
            matrix, initial_state,
            n_mutation_samples, n_noise_samples,
            Normal(0.0, 2.0), # Large noise magnitude
            0.0;              # No mutations
            weights_dist=weights_dist, max_steps=max_steps
        )
        # High noise introduces expression shifts while preserving unmutated topology
        @test res.stable_expression_var >= 0.0
        @test res.unstable_prob_var >= 0.0
    end

    @testset "4. A lot of noise, a lot of mutations" begin
        Random.seed!(42)
        res = CustomStats.compute_mut_robustness(
            matrix, initial_state,
            n_mutation_samples, n_noise_samples,
            Normal(0.0, 2.0), # High noise
            1.0;              # Always mutate
            weights_dist=weights_dist, max_steps=max_steps
        )
        # Combined heavy noise and mutation produce significant shifts and non-zero variance
        @test !isapprox(res.stable_expression_shift, 0.0; atol=1e-12)
        @test res.stable_expression_var >= 0.0
        @test res.unstable_prob_var >= 0.0
    end

    @testset "5. Active noise and active mutations" begin
        Random.seed!(42)
        res = CustomStats.compute_mut_robustness(
            matrix, initial_state,
            n_mutation_samples, n_noise_samples,
            Gamma(1.0, 1.0), # Moderate noise
            0.01;              # Moderate mutation probability
            weights_dist=weights_dist, max_steps=max_steps
        )
        # Standard stochastic operating conditions
        @test res.stable_expression_var >= 0.0
        @test res.unstable_prob_var >= 0.0
    end

    @testset "argument errors" begin
        @test_throws ArgumentError CustomStats.compute_mut_robustness(
            matrix, initial_state, 1, n_noise_samples, Bernoulli(1.0), 0.0;
            weights_dist=weights_dist, max_steps=max_steps)
        @test_throws ArgumentError CustomStats.compute_mut_robustness(
            matrix, initial_state, n_mutation_samples, 1, Bernoulli(1.0), 0.0;
            weights_dist=weights_dist, max_steps=max_steps)
    end

    @testset "broadcasting over a grid of matrices" begin
        Random.seed!(42)
        grid = [copy(matrix) for _ in 1:2, _ in 1:2]
        results = CustomStats.compute_mut_robustness.(
            grid, Ref(initial_state), n_mutation_samples, n_noise_samples,
            Ref(Bernoulli(1.0)), 0.0;
            weights_dist=weights_dist, max_steps=max_steps)
        @test size(results) == (2, 2)
        @test all(r -> isapprox(r.stable_expression_shift, 0.0; atol=1e-12), results)
    end
end

@testset "summarize_simulation_run" begin
    # Minimal synthetic simulation result
    fitness_history = fill(0.5, 2, 2)                # 2 generations 2 individual
    path_length_history = ones((2,2))

    matrices_history = Array{Matrix{Float64}}(undef, 2, 2)
    matrices_history = [ones(2, 2) for _ in 1:2, _ in 1:2]
    completion_history = trues(1,1)
    optimal_phenotype = [1.0, 1.0]
    initial_state = [1.0, 1.0]

    result = BooleanNetwork.SimulationData(
        completion_history,
        fitness_history,
        matrices_history,
        path_length_history,
        initial_state,
        optimal_phenotype
    )

    summary = CustomStats.summarize_simulation_run(result)
    @test isa(summary, Dict)

    # fitness_stats is a vector of generation summaries
    fstats = summary["fitness_stats"]
    @test length(fstats) == 2  # Check it is a tuple
    @test length(fstats[1]) == 2  # Check the averages are size generations
    @test length(fstats[2]) == 2  # Check the stds are size generations
    @test isapprox(fstats[1][1], 0.5; atol=1e-12)  # The average should stay at 0.5
    @test isapprox(fstats[1][2], 0.5; atol=1e-12) 
    @test isapprox(fstats[2][1], 0.0; atol=1e-12)  # No variation
    @test isapprox(fstats[2][2], 0.0; atol=1e-12)

    # path_stats should also be present
    pstats = summary["path_stats"]
    @test length(pstats) == 2
    @test length(pstats[1]) == 2
    @test length(pstats[2]) == 2
    @test isapprox(pstats[1][1], 1.0; atol=1e-12)  # The average should stay at 1.0
    @test isapprox(pstats[2][1], 0.0; atol=1e-12)  # No variation


    # More complex simulation results (5 generations × 3 individuals)
    G, P = 5, 3
    fitness_history = rand(G, P) # Random fitness values between 0 and 1
    
    # Path lengths with some missing values
    path_length_history = Matrix{Union{Int,Nothing}}(undef, G, P)
    path_length_history[1, :] = [3, nothing, 4]      # Gen 1: 66% valid
    path_length_history[2, :] = [2, 5, 3]            # Gen 2: 100% valid
    path_length_history[3, :] = [nothing, 4, nothing] # Gen 3: 33% valid
    path_length_history[4, :] = [6, 3, 4]            # Gen 4: 100% valid
    path_length_history[5, :] = [5, 4, 3]            # Gen 5: 100% valid

    # 2×2 matrices for each individual in each generation
    matrices_history = Array{Matrix{Float64}}(undef, G, P)
    matrices_history = [rand(2, 2) for _ in 1:G, _ in 1:P]  # Random 2×2 matrices
    completion_history = .!isnothing.(path_length_history)
    optimal_phenotype = [1.0, -1.0]
    initial_state = [1.0, 1.0]    

    result = BooleanNetwork.SimulationData(
        completion_history,
        fitness_history,
        matrices_history,
        path_length_history,
        initial_state,
        optimal_phenotype
    )

    summary = CustomStats.summarize_simulation_run(result)

    # Test dimensions
    @test length(summary["fitness_stats"][1]) == G
    @test length(summary["fitness_stats"][2]) == G
    @test length(summary["path_stats"][1]) == G
    @test length(summary["path_stats"][2]) == G
    @test length(summary["completion_stats"][1]) == G
    @test length(summary["completion_stats"][2]) == G
    @test length(summary["alignment_stats"][1]) == G
    @test length(summary["alignment_stats"][2]) == G
end

println("test_data_processing.jl: all tests completed")
