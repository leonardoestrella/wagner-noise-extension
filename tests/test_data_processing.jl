# Tests for data-processing utilities (CustomStats)
include("../src/data_processing.jl")
using Test
using Random


# Deterministic RNG where needed
Random.seed!(1234)

@testset "compute_percentile_stats" begin
    data = [1.0, 2.0, 3.0, nothing]
    stats = CustomStats.compute_percentile_stats(data)

    @test isapprox(stats.mean, 2.0; atol=1e-12)
    @test isapprox(stats.p50, 2.0; atol=1e-12)
    @test isapprox(stats.validity_pct, 75.0; atol=1e-12)
end

@testset "compute_alignment_score" begin
    M = [1.0 0.0;
         0.0 1.0]
    v = [1.0, 1.0]
    score = CustomStats.compute_alignment_score(M, v)
    @test isapprox(score, 1.0; atol=1e-12)

    # A matrix with rows orthogonal to v should yield 0
    M2 = [1.0 -1.0;
          -1.0 1.0]
    score2 = CustomStats.compute_alignment_score(M2, v)
    @test isapprox(score2, 0.0; atol=1e-12)

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
    score_aligned = CustomStats.compute_alignment_score(W_aligned, v_mixed)
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
    score_misaligned = CustomStats.compute_alignment_score(W_misaligned, v_mixed)
    @test isapprox(score_misaligned, -1.0; atol=1e-12)

    # Test partial alignment cases
    v_binary = [1.0, -1.0]  # Binary target phenotype
    
    # Case 1: Matrix with zero net alignment
    W_zero = [1.0  1.0;     # Row sums to 0 when weighted by target
             -1.0 -1.0]     # Row also sums to 0
    score_zero = CustomStats.compute_alignment_score(W_zero, v_binary)
    @test isapprox(score_zero, 0.0; atol=1e-12)
    
    # Case 2: Matrix with 0.5 alignment (half-aligned)
    W_half = [1.0  -1.0;    # Perfectly aligned row (+1)
              1.0   1.0]    # Zero alignment row (0)
    score_half = CustomStats.compute_alignment_score(W_half, v_binary)
    @test isapprox(score_half, 0.5; atol=1e-12)

    # Case 3: Matrix without edges (alignment should be 0)
    W_empty = [0.0 -0.0;    # Perfectly aligned row (+1)
        0.0 0.0]    # Zero alignment row (0)
    score_empty = CustomStats.compute_alignment_score(W_empty, v_binary)
    @test isapprox(score_empty, 0.0; atol=1e-12)
end

@testset "summarize_simulation_run" begin
    # Minimal synthetic simulation result
    fitness = fill(0.5, 1, 1)                # 1 generation × 1 individual
    path_length = Matrix{Int64}(undef, 1, 1)
    path_length[1,1] = 3

    matrices = Array{Matrix{Float64}}(undef, 1, 1)
    matrices[1,1] = [1.0 0.0; 0.0 1.0]

    completion = [1.0]
    phen_opt = [1.0, 1.0]

    result = Dict(
        "fitness" => fitness,
        "path_length" => path_length,
        "matrices" => matrices,
        "completion" => completion,
        "phenotypic_optima" => phen_opt
    )

    summary = CustomStats.summarize_simulation_run(result)
    @test isa(summary, Dict)

    # fitness_stats is a vector of generation summaries
    fstats = summary["fitness_stats"]
    @test length(fstats) == 1
    @test isapprox(fstats[1].mean, 0.5; atol=1e-12)
    @test isapprox(fstats[1].validity_pct, 100.0; atol=1e-12)

    # path_stats should also be present
    pstats = summary["path_stats"]
    @test length(pstats) == 1
    @test isapprox(pstats[1].p50, 3.0; atol=1e-12)

    # More complex simulation results (5 generations × 3 individuals)
    G, P = 5, 3
    fitness = rand(G, P) # Random fitness values between 0 and 1
    
    # Path lengths with some missing values
    path_length = Matrix{Union{Int,Nothing}}(undef, G, P)
    path_length[1,:] = [3, nothing, 4]      # Gen 1: 66% valid
    path_length[2,:] = [2, 5, 3]            # Gen 2: 100% valid
    path_length[3,:] = [nothing, 4, nothing] # Gen 3: 33% valid
    path_length[4,:] = [6, 3, 4]            # Gen 4: 100% valid
    path_length[5,:] = [5, 4, 3]            # Gen 5: 100% valid

    # 2×2 matrices for each individual in each generation
    matrices = Array{Matrix{Float64}}(undef, G, P)
    for g in 1:G, p in 1:P
        matrices[g,p] = rand(2, 2)  # Random 2×2 matrices
    end

    completion = fill(0.8, G)  # 80% completion rate each generation
    phen_opt = [1.0, -1.0]    # Binary phenotype target

    result = Dict(
        "fitness" => fitness,
        "path_length" => path_length,
        "matrices" => matrices,
        "completion" => completion,
        "phenotypic_optima" => phen_opt
    )

    summary = CustomStats.summarize_simulation_run(result)

    # Test dimensions
    @test length(summary["fitness_stats"]) == G
    @test length(summary["path_stats"]) == G
    @test length(summary["completion"]) == G
    @test length(summary["alignment_stats"]) == G

    # Test validity tracking in path_stats
    path_validities = [stats.validity_pct for stats in summary["path_stats"]]
    @test isapprox(path_validities[1], 66.67; atol=0.1) # Gen 1: 2/3 valid
    @test isapprox(path_validities[2], 100.0; atol=0.1) # Gen 2: all valid
    @test isapprox(path_validities[3], 33.33; atol=0.1) # Gen 3: 1/3 valid
    @test isapprox(path_validities[4], 100.0; atol=0.1) # Gen 4: all valid
    @test isapprox(path_validities[5], 100.0; atol=0.1) # Gen 5: all valid
end

println("test_data_processing.jl: all tests completed")