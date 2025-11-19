#!/usr/bin/env julia
"""
End-to-end runner for the two manuscript experiments.
Generates all figures with improved readability and saves them under figures/.
"""

using Plots
using LaTeXStrings
using Distributions
using Statistics
using StatsBase
using Random
using DataFrames
using GLM
using Printf
using HypothesisTests
using ProgressMeter
import PyCall
using PyCall: PyVector, pyimport, PyObject
using LinearAlgebra
using Base.Threads

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const FIGURES_DIR = joinpath(ROOT_DIR, "figures")
mkpath(FIGURES_DIR)
const ROBUSTNESS_FIG_DIR = joinpath(FIGURES_DIR, "scatterplots_mutational_robustness")
mkpath(ROBUSTNESS_FIG_DIR)

include(joinpath(ROOT_DIR, "src", "wagner_algorithm.jl"))
using .BooleanNetwork
include(joinpath(ROOT_DIR, "src", "data_processing.jl"))
using .CustomStats

# Make sure PyCall can import motif_search.py that lives under src/.
const PY_SRC_PATH = joinpath(ROOT_DIR, "src")

const PY_SYS_PATH = PyVector(pyimport("sys")["path"])
if !(PY_SRC_PATH in PY_SYS_PATH)
    push!(PY_SYS_PATH, PY_SRC_PATH)
end
const MotifSearch = pyimport("motif_search")
const PyBuiltins = pyimport("builtins")
const FFL_TYPES = collect(String.(PyVector(MotifSearch["FFL_loops_names"])))

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

function py_string_float_dict(py_dict)::Dict{String, Float64}
    result = Dict{String, Float64}()
    for item in python_items(py_dict)
        key_obj, value_obj = item isa Tuple ? item : (item[1], item[2])
        key = convert(String, key_obj)
        value = convert(Float64, value_obj)
        result[key] = value
    end
    return result
end

function py_int_pyobject_dict(py_dict)::Dict{Int, PyObject}
    result = Dict{Int, PyObject}()
    for item in python_items(py_dict)
        key_obj, value_obj = item isa Tuple ? item : (item[1], item[2])
        key = convert(Int, key_obj)
        result[key] = value_obj
    end
    return result
end

default(dpi=300, legendfontsize=10, guidefont=font(11), tickfont=font(9), margin=8Plots.mm, grid=false)
const BONFERRONI_TESTS = 3 # 3 tests!

# ------------------------------------------------------------------
# Experiment 1 helpers
# ------------------------------------------------------------------

struct NoiseScenario
    label::String
    variance::Float64
    distribution::Distribution
end

Base.@kwdef struct Experiment1Config
    generations::Int = 500 # 500
    pop_size::Int = 300 # 300
    mode::String = "stable"
    trials::Int = 30 # 30
    thetas::Vector{Float64} = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    robustness_n_mutations::Int = 5
    robustness_n_noise_masks::Int = 5
end

Base.@kwdef struct Experiment2Config
    trials::Int = 30 # 30
    max_loop_size::Int = 5 # 5
    sample_size::Int = 30 # 30
end

struct PopulationRobustnessSummary
    mean_expression_shift::Matrix{Float64}
    mean_unstable_shift::Matrix{Float64}
end

struct Experiment1Result
    scenarios::Vector{NoiseScenario}
    averages::Dict{Symbol, Matrix{Float64}}
    sems::Dict{Symbol, Matrix{Float64}}
    final_alignments::Matrix{Float64}
    initial_robustness::PopulationRobustnessSummary
    final_robustness::PopulationRobustnessSummary
    config::Experiment1Config
end

struct MotifStats
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

function build_noise_scenarios(config::Experiment1Config)
    scenarios = NoiseScenario[NoiseScenario("Noiseless", 0.0, Bernoulli(1.0))]
    append!(
        scenarios,
        [NoiseScenario(@sprintf("θ = %.2f", θ), θ, Gamma(1 / θ, θ)) for θ in config.thetas]
    )
    return scenarios
end

function mean_path_length(row) # This function should already be handled in CustomStats
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

function column_mean_and_sem(data::Matrix{Float64}) # This function should already be handled in Custom Stats
    cols = size(data, 2)
    means = Vector{Float64}(undef, cols)
    sems = Vector{Float64}(undef, cols)
    for col in 1:cols
        col_data = view(data, :, col)
        mask = .!isnan.(col_data)
        clean = col_data[mask]
        if isempty(clean)
            means[col] = NaN
            sems[col] = NaN
        else
            means[col] = mean(clean)
            sems[col] = std(clean) / sqrt(length(clean))
        end
    end
    return means, sems
end

function run_experiment1(config::Experiment1Config)
    params = deepcopy(BooleanNetwork.STANDARD_PARAMETERS)
    params["G"] = config.generations
    params["pop_size"] = config.pop_size
    params["mode"] = config.mode

    scenarios = build_noise_scenarios(config)
    gens = params["G"]
    metric_names = (:fit, :path, :completion, :alignment)
    averages = Dict(name => zeros(length(scenarios), gens) for name in metric_names)
    sems = Dict(name => zeros(length(scenarios), gens) for name in metric_names)
    final_alignments = zeros(length(scenarios), config.trials)
    init_expr_shift = zeros(length(scenarios), config.trials)
    init_unstable = zeros(length(scenarios), config.trials)
    final_expr_shift = zeros(length(scenarios), config.trials)
    final_unstable = zeros(length(scenarios), config.trials)

    progress = Progress(length(scenarios); desc="Experiment 1 – noise schedules")
    for (noise_idx, scenario) in enumerate(scenarios)
        all_fit = zeros(config.trials, gens)
        all_path = fill(NaN, config.trials, gens)
        all_completion = zeros(config.trials, gens)
        all_alignment = zeros(config.trials, gens)

        @threads for trial_idx in 1:config.trials
            local_params = deepcopy(params)
            local_params["noise_dist"] = scenario.distribution

            exp_data = BooleanNetwork.run_simulation(local_params)
            fitness_run = exp_data["fitness"]
            path_length_run = exp_data["path_length"]
            completion = Float64.(exp_data["completion"])
            alignments = CustomStats.compute_all_alignments(
                exp_data["matrices"],
                exp_data["phenotypic_optima"]
            )

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
                scenario.distribution;
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
                scenario.distribution;
                mut_prob=local_params["pr"],
                mr=local_params["mr"],
                sigma_r=local_params["σr"],
                noise_prob=local_params["noise_prob"],
                max_steps=local_params["max_steps"],
                activation=BooleanNetwork.activation
            )

            init_expr_shift[noise_idx, trial_idx] = mean(getfield.(initial_robustness, :mean_expression_shift))
            init_unstable[noise_idx, trial_idx] = mean(getfield.(initial_robustness, :mean_unstable_shift))
            final_expr_shift[noise_idx, trial_idx] = mean(getfield.(final_robustness, :mean_expression_shift))
            final_unstable[noise_idx, trial_idx] = mean(getfield.(final_robustness, :mean_unstable_shift))
        end

        means, sem_vals = column_mean_and_sem(all_fit)
        averages[:fit][noise_idx, :] .= means
        sems[:fit][noise_idx, :] .= sem_vals

        means, sem_vals = column_mean_and_sem(all_path)
        averages[:path][noise_idx, :] .= means
        sems[:path][noise_idx, :] .= sem_vals

        means, sem_vals = column_mean_and_sem(all_completion)
        averages[:completion][noise_idx, :] .= means
        sems[:completion][noise_idx, :] .= sem_vals

        means, sem_vals = column_mean_and_sem(all_alignment)
        averages[:alignment][noise_idx, :] .= means
        sems[:alignment][noise_idx, :] .= sem_vals

        next!(progress)
    end

    initial_summary = PopulationRobustnessSummary(init_expr_shift, init_unstable)
    final_summary = PopulationRobustnessSummary(final_expr_shift, final_unstable)
    return Experiment1Result(scenarios, averages, sems, final_alignments, initial_summary, final_summary, config)
end

function add_variance_colorbar!(plt, scenarios, colorscheme)
    if isempty(scenarios)
        return
    end
    min_v = minimum(s.variance for s in scenarios)
    max_v = maximum(s.variance for s in scenarios)
    plot!(
        plt,
        0:1,
        fill(NaN, 2);
        zcolor=[min_v, max_v],
        c=colorscheme,
        label="",
        colorbar=true,
        colorbar_title=L"Noise Variance ($\theta$)"
    )
end

function plot_metric_many_trials(max_gens, scenarios, metric_avgs, metric_sems;
        xlabel="", ylabel="", colorscheme=cgrad(:blues), init_gen=1, title="")
    gens = init_gen:max_gens
    plt = plot(
        1:max_gens,
        metric_avgs[1, 1:max_gens];
        ribbon=metric_sems[1, 1:max_gens],
        color=:black,
        linestyle=:dot,
        linewidth=2,
        label=scenarios[1].label,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=:best
    )

    noisy_scenarios = scenarios[2:end]
    if !isempty(noisy_scenarios)
        min_var = minimum(s.variance for s in noisy_scenarios)
        max_var = maximum(s.variance for s in noisy_scenarios)
        denom = max(max_var - min_var, eps())
        for (offset, scenario) in enumerate(noisy_scenarios)
            rel_idx = offset + 1
            norm_val = (scenario.variance - min_var) / denom
            line_color = get(colorscheme, clamp(norm_val, 0.0, 1.0))
            plot!(
                plt,
                gens,
                metric_avgs[rel_idx, init_gen:max_gens];
                ribbon=metric_sems[rel_idx, init_gen:max_gens] .* 1.96,
                color=line_color,
                linewidth=2,
                label=rel_idx == 2 ? "Noisy" : ""
            )
        end
        add_variance_colorbar!(plt, noisy_scenarios, colorscheme)
    end
    return plt
end

function save_plot(plot_obj, filename)
    filepath = joinpath(FIGURES_DIR, filename)
    savefig(plot_obj, filepath)
    println("Saved $(filepath)")
end

sanitize_filename(name::String) = replace(lowercase(name), r"[^a-z0-9]+" => "_")

function mutational_scatter_plot(
    xvals,
    yvals;
    xlabel::String,
    ylabel::String,
    title::String,
    xrange::Tuple{Float64,Float64}=(-0.05, 1.05),
    yrange::Tuple{Float64,Float64}=(-0.05, 1.05)
)
    plt = scatter(
        xvals,
        yvals;
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label="",
        legend=false,
        markersize=6,
        grid=false,
        xlims=xrange,
        ylims=yrange
    )
    plot!(
        [xrange[1], xrange[2]],
        [yrange[1], yrange[2]];
        color=:black,
        linestyle=:dash,
        label="",
        grid=false
    )
    return plt
end

function save_mutational_robustness_plots(result::Experiment1Result)
    expr_init = result.initial_robustness.mean_expression_shift
    expr_final = result.final_robustness.mean_expression_shift
    stab_init = result.initial_robustness.mean_unstable_shift
    stab_final = result.final_robustness.mean_unstable_shift

    for idx in eachindex(result.scenarios)
        scenario = result.scenarios[idx]
        label_token = sanitize_filename(scenario.label)
        expr_plot = mutational_scatter_plot(
            expr_init[idx, :],
            expr_final[idx, :];
            xlabel="Initial mean expression shift",
            ylabel="Final mean expression shift",
            title="Expression shift"
        )

        stab_plot = mutational_scatter_plot(
            stab_init[idx, :],
            stab_final[idx, :];
            xlabel="Initial mean unstable shift",
            ylabel="Final mean unstable shift",
            title="Stability shift"
        )

        combined = plot(expr_plot, stab_plot; layout=(1, 2), size=(950, 400), grid=false)
        savefig(combined, joinpath(ROBUSTNESS_FIG_DIR, "robustness_$(label_token).png"))
    end

    theta_vals = [scenario.variance for scenario in result.scenarios]
    expr_change = [mean(expr_final[idx, :] .- expr_init[idx, :]) for idx in eachindex(result.scenarios)]
    stab_change = [mean(stab_final[idx, :] .- stab_init[idx, :]) for idx in eachindex(result.scenarios)]

    expr_vs_theta = scatter(
        sqrt.(theta_vals),
        expr_change;
        xlabel="Noise variance (√θ)",
        ylabel="Δ mean expression shift",
        title="Expression shift change vs √θ",
        legend=false,
        markersize=7,
        grid=false
    )
    savefig(expr_vs_theta, joinpath(ROBUSTNESS_FIG_DIR, "expression_vs_theta.png"))

    stab_vs_theta = scatter(
        sqrt.(theta_vals),
        stab_change;
        xlabel="Noise variance (√θ)",
        ylabel="Δ mean unstable shift",
        title="Stability shift change vs √θ",
        legend=false,
        markersize=7,
        grid=false
    )
    savefig(stab_vs_theta, joinpath(ROBUSTNESS_FIG_DIR, "stability_vs_theta.png"))
end

function save_experiment1_figures(result::Experiment1Result)
    gens = result.config.generations
    scenarios = result.scenarios
    mode_tag = lowercase(replace(result.config.mode, " " => "_"))
    mode_title = titlecase(result.config.mode)

    p1 = plot_metric_many_trials(
        gens,
        scenarios,
        result.averages[:fit],
        result.sems[:fit];
        xlabel="Generation",
        ylabel="Average Population Fitness",
        colorscheme=cgrad(:blues)
    )
    save_plot(p1, "fitness_evolution_$(mode_tag).png")

    p2 = plot_metric_many_trials(
        gens,
        scenarios,
        result.averages[:path],
        result.sems[:path];
        xlabel="Generation",
        ylabel="Average Mean Path Length ($(mode_title))",
        colorscheme=cgrad(:greens),
        init_gen=2
    )
    save_plot(p2, "path_evolution_$(mode_tag).png")

    p3 = plot_metric_many_trials(
        min(15, gens),
        scenarios,
        result.averages[:completion],
        result.sems[:completion];
        xlabel="Generation",
        ylabel="Average Completion",
        colorscheme=cgrad(:reds)
    )
    save_plot(p3, "completion_evolution_$(mode_tag).png")

    p4 = plot_metric_many_trials(
        gens,
        scenarios,
        result.averages[:alignment],
        result.sems[:alignment];
        xlabel="Generation",
        ylabel="Average Mean Alignment",
        colorscheme=cgrad(:reds)
    )
    save_plot(p4, "alignment_evolution_$(mode_tag).png")
end

function alignment_tests(result::Experiment1Result; bonferroni_factor::Int=1)
    noiseless = result.final_alignments[1, :]
    noisy = vec(result.final_alignments[2:end, :])
    test = UnequalVarianceTTest(noiseless, noisy)
    println("\nWelch's t-test (noiseless vs noisy final alignment)")
    println(test)
    raw_p = pvalue(test)
    corrected_p = min(raw_p * bonferroni_factor, 1.0)
    println("p-value: $(raw_p)")
    if bonferroni_factor > 1
        println("Bonferroni-corrected p-value (x$(bonferroni_factor)): $(corrected_p)")
    end
end

function alignment_regression_plot(result::Experiment1Result)
    config = result.config
    noisy_means = vec(mean(result.final_alignments[2:end, :], dims=2))
    noisy_sems = vec(std(result.final_alignments[2:end, :], dims=2) ./ sqrt(config.trials))
    x_vals = sqrt.(config.thetas)

    df = DataFrame(Noise=x_vals, Alignment=noisy_means)
    model = lm(@formula(Alignment ~ Noise), df)
    y_pred = predict(model)
    ci_95 = noisy_sems .* 1.96

    xmin, xmax = extrema(x_vals)
    xpad = (xmax - xmin) * 0.1
    shared_xlims = (xmin - xpad, xmax + xpad)

    top_ylims = (
        minimum(noisy_means .- ci_95) - 0.05,
        maximum(noisy_means .+ ci_95) + 0.05
    )

    p_top = scatter(
        x_vals,
        noisy_means;
        yerror=ci_95,
        label="Measured Averages",
        ylabel="Avg Final Alignment",
        legend=:bottomright,
        markersize=8,
        grid=false,
        #grid=:y,
        #gridalpha=0.4,
        ylims=top_ylims,
        xlims=shared_xlims,
        xformatter=_ -> ""
    )
    plot!(
        p_top,
        x_vals,
        y_pred;
        linewidth=3,
        color=:red,
        label=@sprintf("Regression (R² = %.3f)", r2(model))
    )

    p_bottom = plot(
        x_vals,
        zeros(length(x_vals));
        xlabel=L"Noise Standard Deviation ($\sigma = \sqrt{\theta}$)",
        legend=false,
        ylims=(0, 0.01),
        yticks=[0],
        xlims=shared_xlims
    )

    final_plot = plot(p_top, p_bottom; layout=@layout([a{0.85h}; b{0.15h}]))
    save_plot(final_plot, "alignment_regression_broken_axis.png")
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
    limit = min(sample_size, length(collection))
    return [Matrix(collection[i]) for i in 1:limit]
end

function random_matrix_generator(base_params::Dict, config::Experiment2Config)
    function generator()
        phen = sample([1, -1], Weights([base_params["p_phen"], 1 - base_params["p_phen"]]), base_params["N_target"])
        _, _, matrices = BooleanNetwork.initialize_population(
            base_params,
            BooleanNetwork.make_initial_state,
            BooleanNetwork.make_optimal_phenotype,
            BooleanNetwork.activation
        )
        return select_matrices(matrices, config.sample_size), phen
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

function create_summary_plots(data_dict::Dict{String, Vector{Float64}},
        null_expectation::Dict{String, Tuple{Float64, Float64}})
    sorted_keys = sort(collect(keys(data_dict)))
    means = [mean(data_dict[k]) for k in sorted_keys]
    ses = [std(data_dict[k]) / sqrt(length(data_dict[k])) for k in sorted_keys]
    p50_lower = [quantile(data_dict[k], 0.25) for k in sorted_keys]
    p50_upper = [quantile(data_dict[k], 0.75) for k in sorted_keys]
    p95_lower = [quantile(data_dict[k], 0.025) for k in sorted_keys]
    p95_upper = [quantile(data_dict[k], 0.975) for k in sorted_keys]

    null_means = [get(null_expectation, k, (0.0, 0.0))[1] for k in sorted_keys]
    null_ses = [get(null_expectation, k, (0.0, 0.0))[2] for k in sorted_keys]
    colors = [occursin("incoherent", lowercase(k)) ? :red : :blue for k in sorted_keys]

    p_left = scatter(
        sorted_keys,
        means;
        yerror=ses .* 1.96,
        markersize=8,
        markerstrokewidth=2,
        color=colors,
        label="Data",
        xlabel="FFL Type",
        ylabel="Average (%)",
        xrotation=45,
        legend=:bottomright,
        background_color_legend=:transparent,
        #grid=:y,
        #gridstyle=:dash,
        grid = false,
        left_margin=12Plots.mm,
        bottom_margin=10Plots.mm,
        errorbarcolor=:black,
        errorbarlinewidth=2,
        errorbarcap=:round
    )
    scatter!(
        p_left,
        sorted_keys,
        null_means;
        yerror=null_ses .* 1.96,
        marker=:x,
        markersize=9,
        color=:black,
        label="Null Model",
        errorbarcolor=:black,
        errorbarlinewidth=2,
        errorbarcap=:round
    )

    ymax = maximum(p95_upper) * 1.1
    p_right = plot(
        xlabel="FFL Type",
        ylabel="Value",
        xrotation=45,
        legend=:outertopright,
        background_color_legend=:transparent,
        ylim=(0, ymax),
        left_margin=12Plots.mm,
        bottom_margin=12Plots.mm
    )
    for (idx, key) in enumerate(sorted_keys)
        label95 = idx == 1 ? "95% Percentile" : ""
        label50 = idx == 1 ? "50% Percentile" : ""
        labelNull = idx == 1 ? "Null ± 2 SE" : ""

        plot!(p_right, [key, key], [p95_lower[idx], p95_upper[idx]]; color=:gray, lw=6, alpha=0.5, label=label95)
        plot!(p_right, [key, key], [p50_lower[idx], p50_upper[idx]]; color=colors[idx], lw=8, alpha=0.8, label=label50)
        plot!(p_right, [key, key], [null_means[idx] - 1.96 * null_ses[idx], null_means[idx] + 1.96 * null_ses[idx]];
            color=:black, lw=2, alpha=0.9, label=labelNull)
        scatter!(p_right, [key], [null_means[idx]]; color=:black, marker=:x, label="")
    end

    return plot(p_left, p_right; layout=(1, 2), size=(1100, 500))
end

function create_summary_plots_fbcks(expectations_noiseless::Dict{Int, Tuple{Float64, Float64}},
        expectations_random::Dict{Int, Tuple{Float64, Float64}},
        loop_type::String;
        max_size::Int=4,
        color=:blue)
    sizes = 1:max_size
    mean_noiseless = [expectations_noiseless[s][1] for s in sizes]
    sem_noiseless = [expectations_noiseless[s][2] for s in sizes]
    mean_random = [expectations_random[s][1] for s in sizes]
    sem_random = [expectations_random[s][2] for s in sizes]

    lower_bounds = min.(mean_noiseless .- 1.96 .* sem_noiseless, mean_random .- 1.96 .* sem_random)
    upper_bounds = max.(mean_noiseless .+ 1.96 .* sem_noiseless, mean_random .+ 1.96 .* sem_random)
    global_min = minimum(lower_bounds)
    global_max = maximum(upper_bounds)
    padding = max(0.05, 0.1 * (global_max - global_min))
    ymin = clamp(global_min - padding, 0.0, 1.0)
    ymax = clamp(global_max + padding, 0.0, 1.0)

    plt = plot(
        sizes,
        mean_random;
        yerror=sem_random .* 1.96,
        seriestype=:scatter,
        marker=:x,
        markersize=9,
        color=:gray,
        label="Null comparison",
        xlabel="Loop Size",
        ylabel="Proportion",
        title=loop_type,
        ylim=(ymin, ymax),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        legend=:bottomright,
        background_color_legend=:transparent,
        errorbarcolor=:gray,
        errorbarlinewidth=2,
        errorbarcap=:round
    )
    scatter!(
        plt,
        sizes,
        mean_noiseless;
        yerror=sem_noiseless .* 1.96,
        marker=:circle,
        markersize=8,
        color=color,
        label="Evolved Comparisons",
        errorbarcolor=color,
        errorbarlinewidth=2,
        errorbarcap=:round
    )
    return plt
end

function run_experiment2(config::Experiment2Config, base_params::Dict, noise_scenarios::Vector{NoiseScenario})
    random_gen = random_matrix_generator(base_params, config)
    noiseless_gen = evolved_matrix_generator(base_params, Bernoulli(1.0), config)
    noisy_gen = evolved_matrix_generator(base_params, noise_scenarios[end].distribution, config)

    random_stats = aggregate_motif_statistics(random_gen, config; desc="Random matrices")
    noiseless_stats = aggregate_motif_statistics(noiseless_gen, config; desc="Noiseless evolution")
    noisy_stats = aggregate_motif_statistics(noisy_gen, config; desc="High-noise evolution")

    return Experiment2Result(random_stats, noiseless_stats, noisy_stats, config)
end

function save_experiment2_figures(result::Experiment2Result)
    expect_random = summarize_expectations(result.random.ffl)
    expect_noiseless = summarize_expectations(result.noiseless.ffl)
    expect_noisy = summarize_expectations(result.noisy.ffl)

    plot_noiseless = create_summary_plots(result.noiseless.ffl, expect_random)
    save_plot(plot_noiseless, "noiseless_vs_random_matrices.png")

    plot_noisy = create_summary_plots(result.noisy.ffl, expect_noiseless)
    save_plot(plot_noisy, "high_noise_vs_noiseless.png")

    expect_random_reinf = summarize_expectations(result.random.fbcks_reinf)
    expect_random_balanc = summarize_expectations(result.random.fbcks_balanc)
    expect_noiseless_reinf = summarize_expectations(result.noiseless.fbcks_reinf)
    expect_noiseless_balanc = summarize_expectations(result.noiseless.fbcks_balanc)
    expect_noisy_reinf = summarize_expectations(result.noisy.fbcks_reinf)
    expect_noisy_balanc = summarize_expectations(result.noisy.fbcks_balanc)

    # max_size = min(result.config.max_loop_size, 4)
    max_size = result.config.max_loop_size
    p1 = create_summary_plots_fbcks(expect_noiseless_reinf, expect_random_reinf, "Reinforcing, Noiseless"; max_size=max_size, color=:blue)
    p2 = create_summary_plots_fbcks(expect_noiseless_balanc, expect_random_balanc, "Balancing, Noiseless"; max_size=max_size, color=:red)
    p3 = create_summary_plots_fbcks(expect_noisy_reinf, expect_noiseless_reinf, "Reinforcing, Very Noisy"; max_size=max_size, color=:blue)
    p4 = create_summary_plots_fbcks(expect_noisy_balanc, expect_noiseless_balanc, "Balancing, Very Noisy"; max_size=max_size, color=:red)

    final_plot = plot(p1, p2, p3, p4; layout=(2, 2), size=(900, 700))
    save_plot(final_plot, "concentrations_fbcks.png")
end

function coherent_fraction_samples(stats::MotifStats)
    if isempty(stats.ffl)
        return zeros(Float64, 0)
    end
    trials = length(first(values(stats.ffl)))
    samples = zeros(Float64, trials)
    for (key, values) in stats.ffl
        if occursin("Coherent", key)
            samples .+= values
        end
    end
    return samples
end

function motif_welch_tests(result::Experiment2Result; bonferroni_factor::Int=1)
    noisy_samples = coherent_fraction_samples(result.noisy)
    noiseless_samples = coherent_fraction_samples(result.noiseless)
    random_samples = coherent_fraction_samples(result.random)
    if isempty(noiseless_samples) || isempty(random_samples)
        println("Insufficient motif samples to run Welch's test.")
        return
    end
    test1 = UnequalVarianceTTest(noiseless_samples, random_samples)
    println("\nWelch's t-test (noiseless vs random coherent FFL concentration)")
    println(test1)
    raw_p = pvalue(test1)
    corrected_p = min(raw_p * bonferroni_factor, 1.0)
    println("p-value: $(raw_p)")
    if bonferroni_factor > 1
        println("Bonferroni-corrected p-value (x$(bonferroni_factor)): $(corrected_p)")
    end

    test2 = UnequalVarianceTTest(noisy_samples, noiseless_samples)
    println("\nWelch's t-test (very noisy vs noiseless coherent FFL concentration)")
    println(test2)
    raw_p = pvalue(test2)
    corrected_p = min(raw_p * bonferroni_factor, 1.0)
    println("p-value: $(raw_p)")
    if bonferroni_factor > 1
        println("Bonferroni-corrected p-value (x$(bonferroni_factor)): $(corrected_p)")
    end
end

# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

function main()
    exp1_config = Experiment1Config()
    exp1_result = run_experiment1(exp1_config)
    save_experiment1_figures(exp1_result)
    save_mutational_robustness_plots(exp1_result)
    alignment_tests(exp1_result; bonferroni_factor=BONFERRONI_TESTS)
    alignment_regression_plot(exp1_result)

    base_params = deepcopy(BooleanNetwork.STANDARD_PARAMETERS)
    base_params["G"] = exp1_config.generations
    base_params["pop_size"] = exp1_config.pop_size
    base_params["mode"] = exp1_config.mode

    exp2_config = Experiment2Config()
    noise_scenarios = build_noise_scenarios(exp1_config)
    exp2_result = run_experiment2(exp2_config, base_params, noise_scenarios)
    save_experiment2_figures(exp2_result)
    motif_welch_tests(exp2_result; bonferroni_factor=BONFERRONI_TESTS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
