#!/usr/bin/env julia
"""
Compute-only runner for the two manuscript experiments.
Returns experiment results without plotting or saving figures.
"""

"""
JULY 28, 2026 - It is currently under refactoring. Leo will delete
plot_payload to simplify code, and will use JLD2
"""

module ExperimentHandler

using Distributions
using Statistics
using StatsBase
using Random
using Printf
using ProgressMeter
using JSON3
using StructTypes
import PyCall
using PyCall: PyVector, pyimport, PyObject
using LinearAlgebra
using Base.Threads


export run_experiment1, run_experiment2, Experiment1Config, Experiment2Config, build_noise_scenarios
export save_experiment_results, load_experiment_results

    const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))

    include(joinpath(ROOT_DIR, "src", "wagner_algorithm.jl"))
    using .BooleanNetwork
    include(joinpath(ROOT_DIR, "src", "data_processing.jl"))
    using .CustomStats # obtains population-level statistics

    # Make sure PyCall can import motif_search.py that lives under src/.
    const PY_SRC_PATH = joinpath(ROOT_DIR, "src")

    const PY_SYS_PATH = PyVector(pyimport("sys")["path"])
    if !(PY_SRC_PATH in PY_SYS_PATH)
        push!(PY_SYS_PATH, PY_SRC_PATH)
    end
    const MotifSearch = pyimport("motif_search")
    const PyBuiltins = pyimport("builtins")
    const FFL_TYPES = collect(String.(PyVector(MotifSearch["FFL_loops_names"])))

    # ------------------------------------------------------------------
    # Experiment 1 helpers
    # ------------------------------------------------------------------

    struct NoiseScenario
        label::String
        variance::Float64
        distribution::Distribution
    end

    Base.@kwdef struct Experiment1Config #TODO - replace with a "loader" in each experiment
        generations::Int = 500 # 500
        pop_size::Int = 300 # 300
        mode::String = "stable"
        trials::Int = 30 # 30
        thetas::Vector{Float64} = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        robustness_n_mutations::Int = 5
        robustness_n_noise_masks::Int = 30
        standard_noise::Distribution = Gamma(1.0,1.0)
        N_target::Int = 10
        # TODO - incorporate type of initial network topology and connectivity
    end

    struct PopulationRobustnessSummary # TODO - revise with theoretical discussion in methods section
        stable_expression_shift_population::Matrix{Float64}
        stable_expression_variance_population::Matrix{Float64}
        unstable_probability_shift_population::Matrix{Float64}
        unstable_probability_variance_population::Matrix{Float64}
    end

    struct Experiment1Result
        scenarios::Vector{NoiseScenario}
        averages::Dict{Symbol,Matrix{Float64}}
        sems::Dict{Symbol,Matrix{Float64}}
        stds::Dict{Symbol,Matrix{Float64}}
        sizes::Dict{Symbol,Matrix{Float64}}
        final_alignments::Matrix{Float64}
        initial_robustness::PopulationRobustnessSummary
        final_robustness::PopulationRobustnessSummary
        #TODO - add samples of matrices from populations in the generation
        config::Experiment1Config
    end

    # ------------------------------------------------------------------
    # Experiment 2 helpers
    # ------------------------------------------------------------------

    Base.@kwdef struct Experiment2Config
        trials::Int = 30 # 30
        max_loop_size::Int = 5 # 5
        sample_size::Int = 30 # 30
    end

    struct MotifStats # TODO - revise with motif-search algorithms
        ffl::Dict{String, Vector{Float64}}
        fbcks_reinf::Dict{Int, Vector{Float64}}
        fbcks_balanc::Dict{Int, Vector{Float64}}
    end

    struct Experiment2Result
        random::MotifStats
        noiseless::MotifStats
        noisy::MotifStats
        config::Experiment2Config
    end

    # ------------------------------------------------------------------
    # General functions
    # ------------------------------------------------------------------


    function build_noise_scenarios(config::Experiment1Config)
        scenarios = NoiseScenario[NoiseScenario("Noiseless", 0.0, Bernoulli(1.0))]
        append!(
            scenarios,
            [NoiseScenario(@sprintf("θ = %.2f", θ), θ, Gamma(1 / θ, θ)) for θ in config.thetas]
        )
        return scenarios
    end

    function mean_path_length(row) # TODO - This function should already be handled in CustomStats
        values = Float64[]
        for val in row
            if val === nothing || (val isa Missing)
                continue
            end
            num = Float64(val)
            if isnan(num)
                continue
            end
            push!(values, num)
        end
        return isempty(values) ? NaN : mean(values)
    end

    """
        function column_mean_and_stds
    
    computes the mean and standard deviation per column in a data matrix. it ignores
    the entries with missing values, and counts the number of non-missing values.
            (THIS MIGHT BE THE SOURCE OF ERROR FOR θ=1.00)

    args
        - data: a matrix with numerical values
    returns
        - means: a vector with the averages
        - means: a vector with the standard deviations
        - sizes: a vector with the number of non-missing values per column
    """
    # previously: column_mean_and_sem #TODO - remove this comment
    function column_mean_and_stds(data::Matrix{Float64}) # This function should already be handled in Custom Stats
        cols = size(data, 2)
        means = Vector{Float64}(undef, cols)
        stds = Vector{Float64}(undef, cols)
        sizes = Vector{Float64}(undef, cols)
        for col in 1:cols
            col_data = view(data, :, col)
            mask = .!isnan.(col_data)
            clean = col_data[mask]
            if isempty(clean)
                means[col] = NaN
                stds[col] = NaN
            else
                means[col] = mean(clean)
                stds[col] = std(clean) 
                sizes[col] = length(clean)
            end
        end
        return means, stds, sizes
    end

    """
        run_experiment1(config::Experiment1Config)

    Run the first experiment of the model. 
    It subjects populations to various levels of noise throughout evolution.
    It records the average and standard deviation of population's fitness,
    path length, completion percentages, alignment score, and computes
    the mutational robustness tuple (initial expression shift, initial unstable shift,
    final expression shift, initial unstable shift)
    """
    function run_experiment1(config::Experiment1Config)

        # TODO - Add loader of configuration 

        params = deepcopy(BooleanNetwork.STANDARD_PARAMETERS)
        params["G"] = config.generations
        params["pop_size"] = config.pop_size
        params["mode"] = config.mode
        params["N_target"] = config.N_target


        # Pre-allocate variables to hold data
        scenarios = build_noise_scenarios(config)
        gens = params["G"]
        metric_names = (:fit, :path, :completion, :alignment)
        averages = Dict(name => zeros(length(scenarios), gens) for name in metric_names)
        sems = Dict(name => zeros(length(scenarios), gens) for name in metric_names)

        stds = Dict(name => zeros(length(scenarios), gens) for name in metric_names)
        sizes = Dict(name => zeros(length(scenarios), gens) for name in metric_names)

        final_alignments = zeros(length(scenarios), config.trials)

        init_stab_expr_shift = zeros(length(scenarios), config.trials)
        init_stab_expr_var = zeros(length(scenarios), config.trials)
        init_prob_expr_shift = zeros(length(scenarios), config.trials)
        init_prob_expr_var = zeros(length(scenarios), config.trials)

        final_stab_expr_shift = zeros(length(scenarios), config.trials)
        final_stab_expr_var = zeros(length(scenarios), config.trials)
        final_prob_expr_shift = zeros(length(scenarios), config.trials)
        final_prob_expr_var = zeros(length(scenarios), config.trials)

        progress = Progress(length(scenarios); desc="Experiment 1 – noise schedules")
        for (noise_idx, scenario) in enumerate(scenarios)
            all_fit = zeros(config.trials, gens)
            all_path = fill(NaN, config.trials, gens)
            all_completion = zeros(config.trials, gens)
            all_alignment = zeros(config.trials, gens)

          # Run experiments in parallel
            @threads for trial_idx in 1:config.trials 
                # local parameter selection
                local_params = deepcopy(params)
                local_params["noise_dist"] = scenario.distribution 

                exp_data = BooleanNetwork.run_simulation(local_params) # Experiment results

                # extract and compute metrics
                fitness_run = exp_data["fitness"]
                path_length_run = exp_data["path_length"]
                completion = Float64.(exp_data["completion"])
                alignments = CustomStats.compute_all_alignments(
                    exp_data["matrices"],
                    exp_data["phenotypic_optima"]
                )

                # TODO - Factorize into functions that can be tested
                avg_fit = vec(mean(fitness_run, dims=2))
                avg_alignment = vec(mean(alignments, dims=2))
                path_means = map(mean_path_length, eachrow(path_length_run))

                all_fit[trial_idx, :] = avg_fit
                all_path[trial_idx, :] = collect(path_means)
                all_completion[trial_idx, :] = completion
                all_alignment[trial_idx, :] = avg_alignment
                final_alignments[noise_idx, trial_idx] = avg_alignment[end]

                initial_pop = collect(exp_data["matrices"][1, :])
                final_pop = collect(exp_data["matrices"][gens, :])
                initial_robustness = CustomStats.compute_population_mut_robustness(
                    initial_pop,
                    exp_data["initial_state"],
                    config.robustness_n_mutations,
                    config.robustness_n_noise_masks,
                    config.standard_noise;
                    mut_prob=local_params["pr"],
                    mr=local_params["mr"],
                    sigma_r=local_params["σr"],
                    noise_prob=local_params["noise_prob"],
                    max_steps=local_params["max_steps"],
                    activation=BooleanNetwork.activation
                )
                final_robustness = CustomStats.compute_population_mut_robustness(
                    final_pop,
                    exp_data["initial_state"],
                    config.robustness_n_mutations,
                    config.robustness_n_noise_masks,
                    config.standard_noise;
                    mut_prob=local_params["pr"],
                    mr=local_params["mr"],
                    sigma_r=local_params["σr"],
                    noise_prob=local_params["noise_prob"],
                    max_steps=local_params["max_steps"],
                    activation=BooleanNetwork.activation
                )

                # TODO - Incorporate in a for loop
                init_stab_expr_shift[noise_idx, trial_idx] = mean(getfield.(initial_robustness,:stable_expression_shift))
                init_stab_expr_var[noise_idx, trial_idx] = mean(getfield.(initial_robustness, :stable_expression_variance))
                init_prob_expr_shift[noise_idx, trial_idx] = mean(getfield.(initial_robustness, :unstable_probability_shift))
                init_prob_expr_var[noise_idx, trial_idx] = mean(getfield.(initial_robustness, :unstable_probability_variance))

                final_stab_expr_shift[noise_idx, trial_idx] = mean(getfield.(final_robustness, :stable_expression_shift))
                final_stab_expr_var[noise_idx, trial_idx] = mean(getfield.(final_robustness, :stable_expression_variance))
                final_prob_expr_shift[noise_idx, trial_idx] = mean(getfield.(final_robustness, :unstable_probability_shift))
                final_prob_expr_var[noise_idx, trial_idx] = mean(getfield.(final_robustness, :unstable_probability_variance))

            end

            metric_data = Dict(
                :fit => all_fit,
                :path => all_path,
                :completion => all_completion,
                :alignment => all_alignment,
            )

            for metric in (:fit, :path, :completion, :alignment)
                local_avg, local_std, local_sizes = column_mean_and_stds(metric_data[metric])
                averages[metric][noise_idx, :] .= local_avg
                sems[metric][noise_idx, :] .= local_std ./ sqrt.(local_sizes)
                stds[metric][noise_idx, :] .= local_std
                sizes[metric][noise_idx, :] .= local_sizes
            end

            #TODO - PRIORITY - Sample some of the matrices here

            next!(progress)
        end

        initial_summary = PopulationRobustnessSummary(init_stab_expr_shift, init_stab_expr_var, init_prob_expr_shift, init_prob_expr_var)
        final_summary = PopulationRobustnessSummary(final_stab_expr_shift, final_stab_expr_var, final_prob_expr_shift, final_prob_expr_var)
        return Experiment1Result(scenarios, averages, sems, stds, sizes, final_alignments, initial_summary, final_summary, config)
    end

    # ------------------------------------------------------------------
    # Experiment 2 helpers
    # ------------------------------------------------------------------

    function ensure_vector!(store::Dict{K, Vector{Float64}}, key::K, trials::Int) where {K}
        if !haskey(store, key)
            store[key] = zeros(trials)
        end
    end

    function select_matrices(collection, sample_size::Int)
        # TODO - Remove this function
        limit = min(sample_size, length(collection))
        return [Matrix(collection[i]) for i in 1:limit]
    end

    function random_matrix_generator(base_params::Dict, config::Experiment2Config)
        #TODO - refactor with evolved_matrix_generator into a single function
        function generator()
            phen = sample([1, -1], Weights([base_params["p_phen"], 1 - base_params["p_phen"]]), base_params["N_target"])
            _, _, matrices = BooleanNetwork.initialize_population(
                base_params,
                BooleanNetwork.make_initial_state,
                BooleanNetwork.make_optimal_phenotype,
                BooleanNetwork.activation
            ) #TODO - confusing function name. 
            return select_matrices(matrices, config.sample_size), phen # TODO - remove select_amatrices (they are already sorted randomly!)
        end
        return generator
    end

    function evolved_matrix_generator(base_params::Dict, dist, config::Experiment2Config)
        function generator()
            params = deepcopy(base_params)
            params["noise_dist"] = dist
            run_data = BooleanNetwork.run_simulation(params)
            matrices = run_data["matrices"]
            pop_count = size(matrices, 2)
            limit = min(config.sample_size, pop_count)
            slice = [Matrix(matrices[end, idx]) for idx in 1:limit]
            return slice, run_data["phenotypic_optima"]
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

    function aggregate_motif_statistics(generator::Function, config::Experiment2Config; desc::String)
        averages_ffl = Dict{String, Vector{Float64}}()
        averages_reinf = Dict(n => zeros(config.trials) for n in 1:config.max_loop_size)
        averages_balanc = Dict(n => zeros(config.trials) for n in 1:config.max_loop_size)

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
                averages_reinf[size][trial] = mean(view(fbck_counts[size], :, 1))
                averages_balanc[size][trial] = mean(view(fbck_counts[size], :, 2))
            end
            next!(progress)
        end

        return MotifStats(averages_ffl, averages_reinf, averages_balanc)
    end

    function run_experiment2(config::Experiment2Config, base_params::Dict, noise_scenarios::Vector{NoiseScenario})
        random_gen = random_matrix_generator(base_params, config)
        noiseless_gen = evolved_matrix_generator(base_params, Bernoulli(1.0), config)
        noisy_gen = evolved_matrix_generator(base_params, noise_scenarios[end].distribution, config) #TODO - require a specific noise value

        random_stats = aggregate_motif_statistics(random_gen, config; desc="Random matrices")
        noiseless_stats = aggregate_motif_statistics(noiseless_gen, config; desc="Noiseless evolution")
        noisy_stats = aggregate_motif_statistics(noisy_gen, config; desc="High-noise evolution")

        return Experiment2Result(random_stats, noiseless_stats, noisy_stats, config)
    end

    # ------------------------------------------------------------------
    # Serialization helpers (JSON)
    # ------------------------------------------------------------------

    # TODO - Can I avoid the JSON altogheter with JDL2?

    const GAMMA_ALPHA_KEYS = ("alpha", "shape", "k", "α", "Чс")
    const GAMMA_THETA_KEYS = ("theta", "scale", "θ", "Чч")

    function first_key_value(obj, keys)
        for key in keys
            if haskey(obj, key)
                return obj[key]
            end
        end
        return nothing
    end

    StructTypes.StructType(::Type{Experiment1Result}) = StructTypes.Struct()
    StructTypes.StructType(::Type{Experiment2Result}) = StructTypes.Struct()
    StructTypes.StructType(::Type{Distribution}) = StructTypes.CustomStruct()

    function StructTypes.lower(dist::Distribution)
        if dist isa Gamma
            shape, scale = params(dist)
            return (; type="Gamma", shape=Float64(shape), scale=Float64(scale))
        elseif dist isa Bernoulli
            return (; type="Bernoulli", p=Float64(dist.p))
        else
            error("Unsupported distribution type for serialization: $(typeof(dist))")
        end
    end

    function StructTypes.construct(::Type{Distribution}, obj)
        if haskey(obj, "type")
            dtype = String(obj["type"])
            if dtype == "Bernoulli"
                return Bernoulli(Float64(obj["p"]))
            elseif dtype == "Gamma"
                return Gamma(Float64(obj["shape"]), Float64(obj["scale"]))
            else
                error("Unsupported distribution type in JSON: $(dtype)")
            end
        end

        # Legacy payloads without a type field.
        if haskey(obj, "p")
            return Bernoulli(Float64(obj["p"]))
        end

        alpha = first_key_value(obj, GAMMA_ALPHA_KEYS)
        theta = first_key_value(obj, GAMMA_THETA_KEYS)
        if alpha === nothing || theta === nothing
            error("Unsupported distribution payload keys: $(collect(keys(obj)))")
        end
        return Gamma(Float64(alpha), Float64(theta))
    end

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

    # # ------------------------------------------------------------------
    # # Save/load helpers
    # # ------------------------------------------------------------------

    # function save_experiment_results(path::AbstractString, result)
    #     # TODO - Remove. They will be handled via JLD2
    #     JSON3.write(path, result)
    #     return nothing
    # end

    # function load_experiment_results(path::AbstractString, ::Type{T}) where {T}
    #     return JSON3.read(read(path, String), T)
    # end

end # module
