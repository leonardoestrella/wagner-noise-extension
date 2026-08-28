#!/usr/bin/env julia
"""
    ExperimentHandler

Compute-only runner for the two manuscript experiments. Returns experiment results without
plotting or saving figures.

# Exported structures
- `Experiment1Config`: Sets up the configurations for experiment 1. See # Fields
- `Experiment2Config`: Sets up the configurations for experiment 2. See # Fields
- `Experiment1Result`: Contains the results of experiment 1. 
- `Experiment2Result`: Contains the results of experiment 2.
- `MotifStats`: Holds summary statistics for network motif counts

# Exported functions:
- `run_experiment_1`: Runs experiment 1, consisting of evolving populations across noise
    noise levels. It computes population level averages and standard deviations of fitness,
    path lengths, completion, and alignment score. It also computes mutational robustness of
    a sample of matrices, specified in its configurations. It also stores some matrices, 
    specified at the sample points and sample sizes, for further processing.
- `run_experiment_2`: Runs experiment 2, consisting of producing different types of
    populations (random, evolved without noise, and evolved with high noise) and getting the
    average network motif counts found in each population.
- `build_noise_scenarios`: Builds a vector of different noise levels and their labels. 

# Notes
- Uses Threads for parallel processing.
- The module assumes BooleanNetwork was already loaded in Main.  
"""
module ExperimentHandler

using Base.Threads
using Distributions
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using StatsBase
using UnPack
using PyCall: PyVector, pyimport, PyObject

include("../src/wagner_algorithm.jl")
using Main.BooleanNetwork: SimulationData, SimulationParameters
using Main.BooleanNetwork: run_simulation, initialize_population
include("../src/data_processing.jl")
using Main.CustomStats: summarize_history, summarize_simulation_run, alignment_score, 
    compute_mut_robustness

export Experiment1Config, Experiment2Config, Experiment1Result, Experiment2Result, MotifStats
export run_experiment_1, run_experiment_2, build_noise_scenarios

    const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
    const PY_SRC_PATH = joinpath(ROOT_DIR, "src")
    const PY_SYS_PATH = PyVector(pyimport("sys")["path"])

    if !(PY_SRC_PATH in PY_SYS_PATH)
        push!(PY_SYS_PATH, PY_SRC_PATH)
    end

    const MotifSearch = pyimport("motif_search")  # motif_search.py
    const PyBuiltins = pyimport("builtins")
    const FFL_TYPES = collect(String.(PyVector(MotifSearch["FFL_loops_names"])))

    # ------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------
    struct NoiseScenario
        label::String
        variance::Float64
        distribution::Distribution
    end

    Base.@kwdef struct Experiment1Config
        # SimulationParameters
        simulation_parameters::SimulationParameters = SimulationParameters()

        # Variables
        number_mutation_samples::Int = 5
        number_noise_samples::Int = 30
        sample_points::Vector{Int} = [1,500]
        sample_sizes:: Int = 30
        standard_noise::Distribution = Gamma(1.0,1.0)
        thetas::Vector{Float64} = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        trials::Int = 30
    end

    struct Experiment1Result  # Do not delete. Serves as a reference in the data analysis
        scenarios::Vector{NoiseScenario}

        # Summary statistics mapped by metric keys (:avg, :std, :sem, :size).
        # Each Matrix{Float64} has dimensions: (n_scenarios, generations), 
        # aggregated across trials and populations.
        fitness::Dict{Symbol, Matrix{Float64}}
        path_length::Dict{Symbol, Matrix{Float64}}
        completion::Dict{Symbol, Matrix{Float64}}
        alignment::Dict{Symbol, Matrix{Float64}}

        # Sampled population matrices indexed by coordinate.
        # Array dimensions: (n_scenarios, trials, sample_points, sample_sizes).
        matrix_samples::Array{Matrix{Float64}, 4}

        # Population-level mutational robustness metrics.
        # Each Array{Float64, 3} has dimensions: (n_scenarios, trials, sample_points),
        # aggregated across samples ⊂ populations.
        mut_rob::NamedTuple{
            (:stable_expression_shift, :stable_expression_var, :unstable_prob_shift, :unstable_prob_var),
            NTuple{4,Array{Float64,3}}
        }
    end

    function build_noise_scenarios(config::Experiment1Config)
        scenarios = NoiseScenario[NoiseScenario("Noiseless", 0.0, Bernoulli(1.0))]
        append!(
            scenarios,
            [NoiseScenario(@sprintf("θ = %.2f", θ), θ, Gamma(1 / θ, θ)) for θ in config.thetas]
        )
        return scenarios
    end

    """
        aggregate_trials(data::AbstractMatrix{Float64}
        ) -> NamedTuple{(:avg,:std,:sem,:size), NTuple{4,Vector{Float64}}}

    Summarize a `(trials, generations)` matrix into per-generation statistics across trials,
    ignoring `NaN` entries (which arise, e.g., when no individual in a trial's population
    stabilized in a given generation).

    # Arguments
    - `data`: Matrix of per-trial, per-generation values, shape `(trials, generations)`.

    # Returns
    A `NamedTuple` with, for each generation:
    - `avg`: Mean across trials.
    - `std`: Sample standard deviation across trials.
    - `sem`: Standard error of the mean, `std / sqrt(size)`.
    - `size`: Number of non-`NaN` trials.

    # Notes
    - Entries are `NaN` if there are 0 valid trials, and `std`/`sem` are `NaN` if there is
        only 1 valid trial.
    """
    function aggregate_trials(
        data::AbstractMatrix{Float64}
    )::NamedTuple{(:avg, :std, :sem, :size),NTuple{4,Vector{Float64}}}
        n_generations = size(data, 2)
        avg = Vector{Float64}(undef, n_generations)
        sd = Vector{Float64}(undef, n_generations)
        sem = Vector{Float64}(undef, n_generations)
        n = Vector{Float64}(undef, n_generations)

        for gen in 1:n_generations
            valid = filter(!isnan, view(data, :, gen))
            count = length(valid)
            n[gen] = count
            if count == 0
                avg[gen], sd[gen], sem[gen] = NaN, NaN, NaN
            elseif count == 1
                avg[gen], sd[gen], sem[gen] = valid[1], NaN, NaN
            else
                avg[gen] = mean(valid)
                sd[gen] = std(valid; corrected=true)
                sem[gen] = sd[gen] / sqrt(count)
            end
        end

        return (avg=avg, std=sd, sem=sem, size=n)
    end

    """
        run_experiment_1(config::Experiment1Config) -> Experiment1Result

    Run the first experiment of the model. It subjects populations to various levels of
    noise throughout evolution, recording the average, standard deviation, standard error,
    and sample size of population fitness, path length, completion percentages, and
    alignment score per generation, and stores samples of given sizes at some generations.
    It also computes the mutational robustness tuple (stable expression shift, stable
    expression variance, unstable probability shift, unstable probability variance) for the
    sampled matrices, aggregated across the sampled population.

    # Arguments
    - `config::Experiment1Config`: Experiment configuration, including simulation
        parameters, the noise scenarios to sweep (via `thetas`), the number of `trials` per
        scenario, and the sampling settings (`sample_sizes`, `sample_points`).

    # Returns
    - `Experiment1Result` holding the noise scenarios tested, the per-generation summary
        statistics for fitness/path length/completion/alignment, the sampled population
        matrices, and their mutational robustness metrics.

    # Notes
    - Trials for a given noise scenario run in parallel via `@threads`.
    - Mutational robustness is only computed for the sampled matrices in `matrix_samples`,
        since it is expensive to evaluate over the whole population.
    """
    function run_experiment_1(config::Experiment1Config)::Experiment1Result
        @unpack generations = config.simulation_parameters
        @unpack trials, sample_sizes, sample_points, number_mutation_samples,
        number_noise_samples = config

        scenarios = build_noise_scenarios(config)
        n_scenarios = length(scenarios)
        n_sample_points = length(sample_points)
        metric_names = (:avg, :std, :sem, :size)

        # Preallocate variables
        fitness = Dict(name => Matrix{Float64}(undef, n_scenarios, generations) for name in metric_names)
        path_length = Dict(name => Matrix{Float64}(undef, n_scenarios, generations) for name in metric_names)
        completion = Dict(name => Matrix{Float64}(undef, n_scenarios, generations) for name in metric_names)
        alignment = Dict(name => Matrix{Float64}(undef, n_scenarios, generations) for name in metric_names)
        matrix_samples = Array{Matrix{Float64},4}(undef, n_scenarios, trials, n_sample_points, sample_sizes)
        mut_rob = (
            stable_expression_shift=Array{Float64,3}(undef, n_scenarios, trials, n_sample_points),
            stable_expression_var=Array{Float64,3}(undef, n_scenarios, trials, n_sample_points),
            unstable_prob_shift=Array{Float64,3}(undef, n_scenarios, trials, n_sample_points),
            unstable_prob_var=Array{Float64,3}(undef, n_scenarios, trials, n_sample_points)
        )

        progress = Progress(n_scenarios; desc="Experiment 1 – noise schedules")
        for (noise_idx, scenario) in enumerate(scenarios)
            all_fit = zeros(trials, generations)
            all_path = fill(NaN, trials, generations)
            all_completion = zeros(trials, generations)
            all_alignment = zeros(trials, generations)

            @threads for trial_idx in 1:trials
                local_params = deepcopy(config.simulation_parameters)
                local_params.noise_dist = scenario.distribution
                local_data = run_simulation(local_params)
                local_summary = summarize_simulation_run(local_data)  # Computes averages and
                # sample standard deviations of the population across timesteps
                local_sample_matrices = local_data.matrices_history[sample_points,1:sample_sizes]
                # Sample sample_sizes matrices at generations indicated by sample_points

                all_fit[trial_idx, :] .= local_summary["fitness_stats"][1]
                all_path[trial_idx, :] .= local_summary["path_stats"][1]
                all_completion[trial_idx, :] .= local_summary["completion_stats"][1]
                all_alignment[trial_idx, :] .= local_summary["alignment_stats"][1]
                matrix_samples[noise_idx, trial_idx, :, :] .= local_sample_matrices

                local_mut_rob = compute_mut_robustness.(local_sample_matrices,
                    Ref(local_data.initial_state), Ref(number_mutation_samples),
                    Ref(number_noise_samples), Ref(local_params.noise_dist),
                    Ref(local_params.mutation_prob); weights_dist=local_params.weights_dist,
                    max_steps=local_params.max_steps)
                # Computes mutational robustness metrics per sampled matrix, shape
                # (n_sample_points, sample_sizes)

                for field in keys(mut_rob)
                    values = getfield.(local_mut_rob, field)
                    getfield(mut_rob, field)[noise_idx, trial_idx, :] .= vec(mean(values, dims=2))
                    # Aggregate mutational robustness across the sampled population
                end
            end

            for (metric_dict, data) in (
                (fitness, all_fit), (path_length, all_path),
                (completion, all_completion), (alignment, all_alignment)
            )  # Record the data
                stats = aggregate_trials(data)
                for name in metric_names
                    metric_dict[name][noise_idx, :] .= getfield(stats, name)
                end
            end

            next!(progress)
        end

        return Experiment1Result(scenarios, fitness, path_length, completion, alignment,
            matrix_samples, mut_rob)
    end
    # ------------------------------------------------------------------
    # Experiment 2 
    # ------------------------------------------------------------------
    Base.@kwdef struct Experiment2Config
        # SimulationParameters
        simulation_parameters::SimulationParameters = SimulationParameters()

        # Variables
        trials::Int = 30
        max_loop_size::Int = 5
        sample_size::Int = 30
        high_noise_dist::Distribution = Gamma(1.0/8.0, 8.0)
    end

    struct MotifStats
        ffl::Dict{String, Vector{Float64}}
        feedback_reinforcing::Dict{Int, Vector{Float64}}
        feedback_balancing::Dict{Int, Vector{Float64}}
    end

    struct Experiment2Result
        random::MotifStats
        noiseless::MotifStats
        noisy::MotifStats
    end

    function ensure_vector!(store::Dict{K, Vector{Float64}}, key::K, trials::Int) where {K}
        if !haskey(store, key)
            store[key] = zeros(trials)
        end
    end

    """
        sample_matrices(matrices::AbstractVector{<:AbstractMatrix{Float64}}, sample_size::Int
        ) -> Vector{Matrix{Float64}}

    Take the first `sample_size` matrices from a population, or all of them if the
    population is smaller than `sample_size`.

    # Arguments
    - `matrices`: Population of weight matrices to sample from.
    - `sample_size`: Maximum number of matrices to return.

    # Returns
    - `Vector{Matrix{Float64}}`: Up to `sample_size` matrices from `matrices`.
    """
    function sample_matrices(
        matrices::AbstractVector{<:AbstractMatrix{Float64}}, sample_size::Int
    )::Vector{Matrix{Float64}}
        limit = min(sample_size, length(matrices))
        return [Matrix(matrices[i]) for i in 1:limit]
    end

    """
        random_matrix_generator(params::SimulationParameters, config::Experiment2Config) -> Function

    Build a generator that draws a sample of unevolved random matrices, used as a
    motif-statistics baseline.

    # Arguments
    - `params::SimulationParameters`: Simulation parameters; `initial_pop_type` should be
        `:random` so that no stability check is applied.
    - `config::Experiment2Config`: Experiment configuration, providing `sample_size`.

    # Returns
    - A zero-argument function returning `(matrices, optimal_phenotype)`.
    """
    function random_matrix_generator(params::SimulationParameters, sample_size::Int)
        function generator()
            _, optimal_phenotype, matrices = initialize_population(params)
            return sample_matrices(matrices, sample_size), optimal_phenotype
        end
        return generator
    end

    """
        evolved_matrix_generator(params::SimulationParameters, noise_dist::Distribution,
            config::Experiment2Config) -> Function

    Build a generator that evolves a population under `noise_dist` and draws a sample of the
    final generation's matrices.

    # Arguments
    - `params::SimulationParameters`: Simulation parameters (its `noise_dist` is
        overridden by `noise_dist` for each generated run).
    - `noise_dist::Distribution`: Noise distribution applied during evolution.
    - `config::Experiment2Config`: Experiment configuration, providing `sample_size`.

    # Returns
    - A zero-argument function returning `(matrices, optimal_phenotype)`.
    """
    function evolved_matrix_generator(
        params::SimulationParameters, noise_dist::Distribution, sample_size::Int
    )
        function generator()
            local_params = deepcopy(params)
            local_params.noise_dist = noise_dist
            simulation_data = run_simulation(local_params)
            final_matrices = simulation_data.matrices_history[end, :]
            return sample_matrices(final_matrices, sample_size), simulation_data.optimal_phenotype
        end
        return generator
    end

    function compute_ffl_type_counts(matrices::Vector{Matrix{Float64}}, phen_opt)
        if isempty(matrices)
            return Dict{String, Vector{Float64}}()
        end

        counts = Dict(key => zeros(length(matrices)) for key in FFL_TYPES)
        for (idx, W) in enumerate(matrices)
            clean = copy(W)
            clean[diagind(clean)] .= 0
            _, py_counts = MotifSearch[:count_ffl_types](clean; visualize=false, phen_opt=phen_opt)
            counts_dict = py_string_float_dict(py_counts)
            total = sum(values(counts_dict))
            norm = total > 0 ? total : 1.0
            for key in FFL_TYPES
                value = get(counts_dict, key, 0.0)
                counts[key][idx] = value / norm
            end
        end
        return counts
    end

    function compute_fbck_type_counts(matrices::Vector{Matrix{Float64}}; max_size::Int=4)
        counts = Dict{Int, Matrix{Float64}}()
        for size in 1:max_size
            counts[size] = zeros(length(matrices), 2)
        end

        for (row_idx, W) in enumerate(matrices)
            _, py_counts = MotifSearch[:count_feedback_loops](W; max_size=max_size)
            counts_dict = py_int_pyobject_dict(py_counts)
            for size in 1:max_size
                if haskey(counts_dict, size)
                    type_dict = py_string_float_dict(counts_dict[size])
                    pos = get(type_dict, "Reinforcing Feedback", 0.0)
                    neg = get(type_dict, "Balancing Feedback", 0.0)
                    total = pos + neg
                    if total > 0
                        counts[size][row_idx, :] .= [pos / total, neg / total]
                    end
                end
            end
        end
        return counts
    end

    function summarize_expectations(data::Dict{K, Vector{Float64}}) where {K}
        Dict(k => (mean(v), std(v) / sqrt(length(v))) for (k, v) in data)
    end

    """
        aggregate_motif_statistics(generator::Function, config::Experiment2Config; desc::String
        ) -> MotifStats

    Repeatedly draw a sample of matrices from `generator` and average FFL-type and
    feedback-loop-type proportions across `config.trials` samples.

    # Arguments
    - `generator::Function`: Zero-argument function returning `(matrices, optimal_phenotype)`,
        e.g. from `random_matrix_generator` or `evolved_matrix_generator`.
    - `config::Experiment2Config`: Experiment configuration (`trials`, `max_loop_size`).

    # Keywords
    - `desc::String`: Progress bar description.

    # Returns
    - `MotifStats` with per-trial averages of FFL-type and feedback-loop-type proportions.
    """
    function aggregate_motif_statistics(generator::Function, config::Experiment2Config; desc::String)
        averages_ffl = Dict{String, Vector{Float64}}()
        averages_feedback_reinforcing = Dict(n => zeros(config.trials) for n in 1:config.max_loop_size)
        averages_feedback_balancing = Dict(n => zeros(config.trials) for n in 1:config.max_loop_size)

        progress = Progress(config.trials; desc=desc)
        for trial in 1:config.trials
            matrices, phen_opt = generator()
            ffl_counts = compute_ffl_type_counts(matrices, phen_opt)
            for (key, values) in ffl_counts
                ensure_vector!(averages_ffl, key, config.trials)
                averages_ffl[key][trial] = mean(values)
            end

            fbck_counts = compute_fbck_type_counts(matrices; max_size=config.max_loop_size)
            for size in 1:config.max_loop_size
                averages_feedback_reinforcing[size][trial] = mean(view(fbck_counts[size], :, 1))
                averages_feedback_balancing[size][trial] = mean(view(fbck_counts[size], :, 2))
            end
            next!(progress)
        end

        return MotifStats(averages_ffl, averages_feedback_reinforcing, averages_feedback_balancing)
    end

    """
        run_experiment_2(config::Experiment2Config, target_scenario::NoiseScenario) -> Experiment2Result

    Run the second experiment of the model. It compares feed-forward-loop and feedback-loop
    motif statistics between random matrices, matrices evolved without noise, and matrices
    evolved under `target_scenario`.

    # Arguments
    - `config::Experiment2Config`: Experiment configuration.
    - `target_scenario::NoiseScenario`: Noise scenario used for the "noisy" evolved
        population (see `build_noise_scenarios`).

    # Returns
    - `Experiment2Result` with motif statistics for the random, noiseless, and noisy
        populations.

    ### PROGRESS MARK - AUG 27, 2026
    The experiment 2 needs revision in how are frequencies computed. I need theoretical
    background. 
    """
    function run_experiment_2(config::Experiment2Config)
        @unpack simulation_parameters, trials, max_loop_size, sample_size, high_noise_dist = config

        random_gen = random_matrix_generator(simulation_parameters, sample_size)
        noiseless_gen = evolved_matrix_generator(simulation_parameters, Bernoulli(1.0), sample_size)
        noisy_gen = evolved_matrix_generator(simulation_parameters, high_noise_dist, sample_size)

        random_stats = aggregate_motif_statistics(random_gen, config; desc="Random matrices")
        noiseless_stats = aggregate_motif_statistics(noiseless_gen, config; desc="Noiseless evolution")
        noisy_stats = aggregate_motif_statistics(noisy_gen, config; desc="High-noise evolution")

        return Experiment2Result(random_stats, noiseless_stats, noisy_stats)
    end

    # ------------------------------------------------------------------
    # PyCall interop helpers
    # ------------------------------------------------------------------

    python_list(obj) = PyBuiltins["list"](obj)

    function python_items(py_dict)
        if py_dict isa PyObject
            return PyVector(python_list(py_dict.items()))
        elseif py_dict isa AbstractDict
            return collect(py_dict)
        else
            error("Unsupported dictionary type: $(typeof(py_dict))")
        end
    end

    function py_string_float_dict(py_dict)::Dict{String,Float64}
        result = Dict{String,Float64}()
        for item in python_items(py_dict)
            key_obj, value_obj = item isa Tuple ? item : (item[1], item[2])
            key = convert(String, key_obj)
            value = convert(Float64, value_obj)
            result[key] = value
        end
        return result
    end

    function py_int_pyobject_dict(py_dict)::Dict{Int,PyObject}
        result = Dict{Int,PyObject}()
        for item in python_items(py_dict)
            key_obj, value_obj = item isa Tuple ? item : (item[1], item[2])
            key = convert(Int, key_obj)
            result[key] = value_obj
        end
        return result
    end
end # module