# Minimal unit tests for core helpers in BooleanNetwork
# Include the source so the `BooleanNetwork` module is available
include("../src/wagner_algorithm.jl")

# Deterministic RNG for reproducibility
using Random
import Distributions
Random.seed!(12345)

# Test sample_binary_state edge cases
s = BooleanNetwork.sample_binary_state(10, 0.0)
@assert length(s) == 10
@assert all(x -> x == -1, s)

s2 = BooleanNetwork.sample_binary_state(8, 1.0)
@assert all(x -> x == 1, s2)

s3 = BooleanNetwork.sample_binary_state(100, 0.5)
@assert length(s3) == 100

# Test recombine_rows preserves rows for p_rec == 0 and p_rec == 1
A = ones(4,4)
B = -ones(4,4)
C = BooleanNetwork.recombine_rows(A, B, 0.0)
@assert all(C .== A)

C2 = BooleanNetwork.recombine_rows(A, B, 1.0)
@assert all(C2 .== B)

# Test apply_noise! special-case (noise_prob == 1.0)
W = ones(3,3)
BooleanNetwork.apply_noise!(W, 1.0, Distributions.Normal(2.0, 0.0))
# With Normal(2.0,0.0) all draws equal 2.0
@assert all(abs.(W .- 2.0) .< 1e-12)

println("test_operations.jl: all assertions passed")
