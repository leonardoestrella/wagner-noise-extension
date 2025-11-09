using Test

# Tests for wagner_algorithm.jl 
println("Running all tests for BooleanNetwork...")

# Run basic operations tests
include("test_operations.jl")

# Run comprehensive core tests
include("test_core.jl")

# Tests for data_processiong.jl
println("Running all tests for CustomStats")

# Run data-processing tests
include("test_data_processing.jl")

println("All tests completed.")
