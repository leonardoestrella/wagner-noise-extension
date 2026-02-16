module PlotPayload

using JSON3

export PlotScenario
export PlotPopulationRobustnessSummary
export PlotExperiment1Config
export PlotExperiment1Result
export PlotMotifStats
export PlotExperiment2Config
export PlotExperiment2Result
export save_exp1_plot_payload
export save_exp2_plot_payload
export load_exp1_plot_payload
export load_exp2_plot_payload

Base.@kwdef struct PlotScenario
    label::String
    variance::Float64
end

struct PlotPopulationRobustnessSummary
    mean_expression_shift::Matrix{Float64}
    mean_unstable_shift::Matrix{Float64}
end

Base.@kwdef struct PlotExperiment1Config
    generations::Int
    pop_size::Int
    mode::String
    trials::Int
    thetas::Vector{Float64}
    robustness_n_mutations::Int
    robustness_n_noise_masks::Int
end

struct PlotExperiment1Result
    scenarios::Vector{PlotScenario}
    averages::Dict{Symbol, Matrix{Float64}}
    sems::Dict{Symbol, Matrix{Float64}}
    final_alignments::Matrix{Float64}
    initial_robustness::PlotPopulationRobustnessSummary
    final_robustness::PlotPopulationRobustnessSummary
    config::PlotExperiment1Config
end

struct PlotMotifStats
    ffl::Dict{String, Vector{Float64}}
    fbcks_reinf::Dict{Int, Vector{Float64}}
    fbcks_balanc::Dict{Int, Vector{Float64}}
end

Base.@kwdef struct PlotExperiment2Config
    trials::Int
    max_loop_size::Int
    sample_size::Int
end

struct PlotExperiment2Result
    random::PlotMotifStats
    noiseless::PlotMotifStats
    noisy::PlotMotifStats
    config::PlotExperiment2Config
end

matrix_to_rows(mat::AbstractMatrix{<:Real}) = [collect(Float64.(view(mat, i, :))) for i in axes(mat, 1)]
floatvec(x) = Float64.(collect(x))

function to_plain_json(x)
    if x isa JSON3.Object
        out = Dict{String,Any}()
        for (k, v) in pairs(x)
            out[String(k)] = to_plain_json(v)
        end
        return out
    elseif x isa JSON3.Array
        return [to_plain_json(v) for v in x]
    else
        return x
    end
end

function matrix_from_payload(data, nrows::Int, ncols::Int)
    rows = collect(data)
    if nrows == 0
        return zeros(Float64, 0, ncols)
    end

    if !isempty(rows) && rows[1] isa AbstractVector
        mat = Matrix{Float64}(undef, nrows, ncols)
        for i in 1:nrows
            row = Float64.(collect(rows[i]))
            if length(row) != ncols
                error("Matrix row $(i) has length $(length(row)); expected $(ncols).")
            end
            mat[i, :] = row
        end
        return mat
    end

    flat = Float64.(rows)
    expected = nrows * ncols
    if length(flat) != expected
        error("Flattened matrix length $(length(flat)) does not match expected $(expected).")
    end
    return reshape(flat, nrows, ncols)
end

function exp1_to_plot_payload(result)::Dict{String,Any}
    return Dict(
        "schema_version" => "plot_payload_v1",
        "kind" => "experiment1",
        "scenarios" => [
            Dict(
                "label" => String(s.label),
                "variance" => Float64(s.variance),
            ) for s in result.scenarios
        ],
        "averages" => Dict(String(k) => matrix_to_rows(v) for (k, v) in result.averages),
        "sems" => Dict(String(k) => matrix_to_rows(v) for (k, v) in result.sems),
        "final_alignments" => matrix_to_rows(result.final_alignments),
        "initial_robustness" => Dict(
            "mean_expression_shift" => matrix_to_rows(result.initial_robustness.mean_expression_shift),
            "mean_unstable_shift" => matrix_to_rows(result.initial_robustness.mean_unstable_shift),
        ),
        "final_robustness" => Dict(
            "mean_expression_shift" => matrix_to_rows(result.final_robustness.mean_expression_shift),
            "mean_unstable_shift" => matrix_to_rows(result.final_robustness.mean_unstable_shift),
        ),
        "config" => Dict(
            "generations" => Int(result.config.generations),
            "pop_size" => Int(result.config.pop_size),
            "mode" => String(result.config.mode),
            "trials" => Int(result.config.trials),
            "thetas" => collect(Float64.(result.config.thetas)),
            "robustness_n_mutations" => Int(result.config.robustness_n_mutations),
            "robustness_n_noise_masks" => Int(result.config.robustness_n_noise_masks),
        ),
    )
end

function motif_stats_to_plot_payload(stats)::Dict{String,Any}
    return Dict(
        "ffl" => Dict(String(k) => floatvec(v) for (k, v) in stats.ffl),
        "fbcks_reinf" => Dict(string(k) => floatvec(v) for (k, v) in stats.fbcks_reinf),
        "fbcks_balanc" => Dict(string(k) => floatvec(v) for (k, v) in stats.fbcks_balanc),
    )
end

function exp2_to_plot_payload(result)::Dict{String,Any}
    return Dict(
        "schema_version" => "plot_payload_v1",
        "kind" => "experiment2",
        "random" => motif_stats_to_plot_payload(result.random),
        "noiseless" => motif_stats_to_plot_payload(result.noiseless),
        "noisy" => motif_stats_to_plot_payload(result.noisy),
        "config" => Dict(
            "trials" => Int(result.config.trials),
            "max_loop_size" => Int(result.config.max_loop_size),
            "sample_size" => Int(result.config.sample_size),
        ),
    )
end

function parse_int_key(k)
    if k isa Integer
        return Int(k)
    end
    return parse(Int, String(k))
end

function normalize_exp1_payload(raw::Dict{String,Any})::PlotExperiment1Result
    scenarios = [
        PlotScenario(
            label=String(s["label"]),
            variance=Float64(s["variance"]),
        ) for s in raw["scenarios"]
    ]

    cfg = raw["config"]
    generations = Int(cfg["generations"])
    trials = Int(cfg["trials"])
    n_scenarios = length(scenarios)

    averages = Dict{Symbol,Matrix{Float64}}()
    for (k, v) in raw["averages"]
        averages[Symbol(k)] = matrix_from_payload(v, n_scenarios, generations)
    end

    sems = Dict{Symbol,Matrix{Float64}}()
    for (k, v) in raw["sems"]
        sems[Symbol(k)] = matrix_from_payload(v, n_scenarios, generations)
    end

    final_alignments = matrix_from_payload(raw["final_alignments"], n_scenarios, trials)
    initial_expr = matrix_from_payload(raw["initial_robustness"]["mean_expression_shift"], n_scenarios, trials)
    initial_unstable = matrix_from_payload(raw["initial_robustness"]["mean_unstable_shift"], n_scenarios, trials)
    final_expr = matrix_from_payload(raw["final_robustness"]["mean_expression_shift"], n_scenarios, trials)
    final_unstable = matrix_from_payload(raw["final_robustness"]["mean_unstable_shift"], n_scenarios, trials)

    config = PlotExperiment1Config(
        generations=generations,
        pop_size=Int(cfg["pop_size"]),
        mode=String(cfg["mode"]),
        trials=trials,
        thetas=Float64.(collect(cfg["thetas"])),
        robustness_n_mutations=Int(get(cfg, "robustness_n_mutations", 0)),
        robustness_n_noise_masks=Int(get(cfg, "robustness_n_noise_masks", 0)),
    )

    initial_summary = PlotPopulationRobustnessSummary(initial_expr, initial_unstable)
    final_summary = PlotPopulationRobustnessSummary(final_expr, final_unstable)

    return PlotExperiment1Result(
        scenarios,
        averages,
        sems,
        final_alignments,
        initial_summary,
        final_summary,
        config,
    )
end

function normalize_motif_stats_payload(stats::Dict{String,Any})::PlotMotifStats
    ffl = Dict{String,Vector{Float64}}(String(k) => floatvec(v) for (k, v) in stats["ffl"])

    fbcks_reinf = Dict{Int,Vector{Float64}}()
    for (k, v) in stats["fbcks_reinf"]
        fbcks_reinf[parse_int_key(k)] = floatvec(v)
    end

    fbcks_balanc = Dict{Int,Vector{Float64}}()
    for (k, v) in stats["fbcks_balanc"]
        fbcks_balanc[parse_int_key(k)] = floatvec(v)
    end

    return PlotMotifStats(ffl, fbcks_reinf, fbcks_balanc)
end

function normalize_exp2_payload(raw::Dict{String,Any})::PlotExperiment2Result
    cfg = raw["config"]
    config = PlotExperiment2Config(
        trials=Int(cfg["trials"]),
        max_loop_size=Int(cfg["max_loop_size"]),
        sample_size=Int(cfg["sample_size"]),
    )

    return PlotExperiment2Result(
        normalize_motif_stats_payload(raw["random"]),
        normalize_motif_stats_payload(raw["noiseless"]),
        normalize_motif_stats_payload(raw["noisy"]),
        config,
    )
end

function save_plot_payload(path::AbstractString, payload::Dict{String,Any})
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return nothing
end

save_exp1_plot_payload(path::AbstractString, result) = save_plot_payload(path, exp1_to_plot_payload(result))
save_exp2_plot_payload(path::AbstractString, result) = save_plot_payload(path, exp2_to_plot_payload(result))

function load_exp1_plot_payload(path::AbstractString)::PlotExperiment1Result
    raw = to_plain_json(JSON3.read(read(path, String)))
    return normalize_exp1_payload(raw)
end

function load_exp2_plot_payload(path::AbstractString)::PlotExperiment2Result
    raw = to_plain_json(JSON3.read(read(path, String)))
    return normalize_exp2_payload(raw)
end

end # module
