"""
    SolverAPI

This module provides an interface to solve optimization problems.

Exports `solve`, `print_model`, `response`, `serialize`, and `deserialize`.

Our format is based on JSON. For examples, see `tests/inputs` and `tests/outputs`.
"""
module SolverAPI

import MathOptInterface as MOI
import JSON3

export serialize, deserialize, solve, print_model, response

include("json_to_moi.jl") # Utilities for building MOI constraints/objectives from JSON.

# SolverAPI
# ========================================================================================

const Request = JSON3.Object
const Response = Dict{String,Any}

"""
    ErrorType

An `@enum` of possible errors types.

## Values

  - `InvalidFormat`: The request is syntactically incorrect. Expected
    fields might be missing or of the wrong type.
  - `InvalidModel`: The model is semantically incorrect. For example, an
    unknown `sense` was provided.
  - `Unsupported`: Some unsupported feature was used. That includes
    using unsupported attributes for a solver or specifying an
    unsupported solver.
  - `NotAllowed`: An operation is supported but cannot be applied in the
    current state of the model
  - `Domain`: A domain-specific, internal error occurred. For example, a
    function cannot be converted to either linear or quadratic form.
  - `Other`: An error that does not fit in any of the above categories.
"""
@enum ErrorType InvalidFormat InvalidModel Unsupported NotAllowed Domain Other

struct Error <: Exception
    type::ErrorType
    message::String
end

"""
    response([request], [model], [solver]; kw...)::Dict{String,Any}

Return a response

## Args

  - `request::SolverAPI.Request`: [optionally] The request that was received.
  - `model::MOI.ModelLike`: [optionally] The model that was solved.
  - `solver::MOI.AbstractOptimizer`: [optionally] The solver that was used.

## Keyword Args

  - `version::String`: The version of the API.
  - `termination_status::MOI.TerminationStatus`: The termination status of the solver.
  - `errors::SolverAPI.Error`: A vector of `Error`s.
"""
function response(;
    version = "0.1",
    termination_status = MOI.OTHER_ERROR,
    errors::Vector{Error} = Error[],
)
    pairs = Pair{String,Any}["version"=>version, "termination_status"=>termination_status]
    if !isempty(errors)
        push!(pairs, "errors" => errors)
    end
    return Dict(pairs)
end
response(error::Error; kw...) = response(; errors = [error], kw...)
function response(
    json::Request,
    model::MOI.ModelLike,
    solver::MOI.AbstractOptimizer;
    version = "0.1",
    kw...,
)
    res = Dict{String,Any}()

    res["version"] = version

    solver_name = MOI.get(solver, MOI.SolverName())
    solver_ver = try
        VersionNumber(MOI.get(solver, MOI.SolverVersion()))
    catch
        pkgversion(parentmodule(typeof(solver)))
    end
    res["solver_version"] = string(solver_name, '_', solver_ver)

    res["termination_status"] = string(MOI.get(model, MOI.TerminationStatus()))

    format = get(json.options, :print_format, nothing)
    if !isnothing(format)
        res["model_string"] = print_model(model, format)
    end

    if Bool(get(json.options, :print_only, false))
        return res
    end

    res["solve_time_sec"] = Float64(MOI.get(model, MOI.SolveTimeSec()))

    result_count = MOI.get(model, MOI.ResultCount())
    results = [Dict{String,Any}() for _ in 1:result_count]
    var_names = [string('\"', v, '\"') for v in json.variables]
    var_idxs = MOI.get(model, MOI.ListOfVariableIndices())

    for idx in 1:result_count
        r = results[idx]

        r["primal_status"] = string(MOI.get(model, MOI.PrimalStatus(idx)))

        # TODO: It is redundant to return the names for every result, since they are fixed -
        # try relying on fixed vector ordering and don't return names.
        r["names"] = var_names
        r["values"] = MOI.get(model, MOI.VariablePrimal(idx), var_idxs)

        if json.sense != "feas"
            r["objective_value"] = MOI.get(model, MOI.ObjectiveValue(idx))
        end
    end

    if !isempty(results)
        res["results"] = results
    end

    return res
end

"""
    serialize([io], response::Response)

Serialize the `response`.

## Args

  - `io::IO`: An optional IO stream to write to. If not provided, a
    `Vector{UInt8}` is returned.
  - `response`: A `SolverAPI.Response` to serialize.
"""
serialize(response::Response) = Vector{UInt8}(JSON3.write(response))
serialize(io::IO, response::Response) = JSON3.write(io, response)

"""
    deserialize(body::Vector{UInt8})::Request
    deserialize(body::String)::Request

Deserialize `body` to an `Request`. Returns an instances that can be
used with `solve` or `print_model`.
"""
deserialize(body::Vector{UInt8}) = JSON3.read(body)
deserialize(body::String) = deserialize(Vector{UInt8}(body))

"""
    solve([fn], request::Request, solver::MOI.AbstractOptimizer)::Response

Solve the optimization problem. Invalid specifications are handled
gracefully and included in `Response`.

## Args

  - `fn`: [Optionally] A function to call before solving the problem. This is useful
    for setting up custom solver options. The function is called with
    the `MOI.ModelLike` model and should return `Nothing`.
  - `request`: A `SolverAPI.Request` specifying the optimization problem.
  - `solver`: A `MOI.AbstractOptimizer` to use to solve the problem.
"""
function solve(fn, json::Request, solver::MOI.AbstractOptimizer)
    errors = validate(json)
    isempty(errors) || return response(; errors)

    # TODO (dba) `SolverAPI.jl` should be decoupled from any solver specific code.
    solver_info = Dict{Symbol,Any}()
    solver_name = lowercase(get(json.options, :solver, "highs")) 
    if solver_name in ["minizinc","csp2sat"]
        T = Int
        solver_info[:use_indicator] = false
    else
        T = Float64
        solver_info[:use_indicator] = true
    end

    model = MOI.instantiate(() -> solver; with_cache_type = T, with_bridge_type = T)

    try
        set_options!(model, json.options)
        load!(model, json, T, solver_info)
        fn(model)
        if !Bool(get(json.options, :print_only, false))
            MOI.optimize!(model)
        end
        return response(json, model, solver)
    catch e
        _err(E) = response(Error(E, sprint(Base.showerror, e)))
        if e isa MOI.UnsupportedError
            return _err(Unsupported)
        elseif e isa MOI.NotAllowedError
            return _err(NotAllowed)
        elseif e isa MOI.InvalidIndex ||
               e isa MOI.ResultIndexBoundsError ||
               e isa MOI.ScalarFunctionConstantNotZero ||
               e isa MOI.LowerBoundAlreadySet ||
               e isa MOI.UpperBoundAlreadySet ||
               e isa MOI.OptimizeInProgress ||
               e isa MOI.InvalidCallbackUsage
            return _err(Domain)
        elseif e isa ErrorException
            return _err(Other)
        elseif e isa Error
            return response(e)
        else
            rethrow()
        end
    finally
        MOI.empty!(model)
    end
end
solve(request::Request, solver::MOI.AbstractOptimizer) =
    solve(model -> nothing, request, solver)
solve(request::Dict, solver::MOI.AbstractOptimizer) =
    solve(JSON3.read(JSON3.write(request)), solver)

"""
    print_model(request::Request)::String
    print_model(model::MOI.ModelLike, format::String)::String

Print the `model`. `format` options (case-insensitive) are:

  - `MOI` (MathOptInterface default printout)
  - `LaTeX` (MathOptInterface default LaTeX printout)
  - `MOF` (MathOptFormat file)
  - `LP` (LP file)
  - `MPS` (MPS file)
  - `NL` (NL file)
"""
function print_model(model::MOI.ModelLike, format::String)
    format_lower = lowercase(format)
    if format_lower == "moi" || format_lower == "latex"
        mime = MIME(format_lower == "latex" ? "text/latex" : "text/plain")
        print_options = MOI.Utilities._PrintOptions(
            mime;
            simplify_coefficients = true,
            print_types = false,
        )
        return sprint() do io
            return MOI.Utilities._print_model(io, print_options, model)
        end
    elseif format_lower == "latex"
        # NOTE: there are options for latex_formulation, e.g. don't print MOI function/set types
        # https://jump.dev/MathOptInterface.jl/dev/submodules/Utilities/reference/#MathOptInterface.Utilities.latex_formulation
        return sprint(print, MOI.Utilities.latex_formulation(model))
    elseif format_lower == "mof"
        print_options = (; format = MOI.FileFormats.FORMAT_MOF, print_compact = true)
    elseif format_lower == "lp"
        print_options = (; format = MOI.FileFormats.FORMAT_LP)
    elseif format_lower == "mps"
        print_options = (; format = MOI.FileFormats.FORMAT_MPS)
    elseif format_lower == "nl"
        print_options = (; format = MOI.FileFormats.FORMAT_NL)
    else
        throw(Error(Unsupported, "File type \"$format\" not supported."))
    end
    dest = MOI.FileFormats.Model(; print_options...)
    MOI.copy_to(dest, model)
    return sprint(write, dest)
end
function print_model(request::Request; T = Float64)
    errors = validate(request)
    if length(errors) > 0
        throw(CompositeException(errors))
    end

    # Default to MOI format.
    format = get(request.options, :print_format, "MOI")

    # TODO cleanup/refactor solver_info logic.
    use_indicator = T == Float64
    solver_info = Dict{Symbol,Any}(:use_indicator => use_indicator)
    model = MOI.Utilities.Model{T}()

    load!(model, request, T, solver_info)
    return print_model(model, format)
end
print_model(request::Dict) = print_model(JSON3.read(JSON3.write(request)))

# Internal
# ========================================================================================

function validate(json::Request)#::Vector{Error}
    out = Error[]
    # Helper for adding errors
    _err(msg::String) = push!(out, Error(InvalidFormat, msg))
    valid_shape = true

    # Syntax.
    for k in [:version, :sense, :variables, :constraints, :objectives, :options]
        if !haskey(json, k)
            valid_shape = false
            _err("Missing required field `$(k)`.")
        end
    end

    # If the shape is not valid we can't continue.
    valid_shape || return out

    if !isa(json.version, String)
        _err("Invalid `version` field. Must be a string.")
    end

    if json.version != "0.1"
        _err("Invalid version `$(repr(json.version))`. Only `\"0.1\"` is supported.")
    end

    if !isa(json.options, JSON3.Object)
        _err("Invalid `options` field. Must be an object.")
    end

    for k in [:variables, :constraints, :objectives]
        if !isa(json[k], JSON3.Array)
            _err("Invalid `$(k)` field. Must be an array.")
        end
    end

    if any(si -> !isa(si, String), json.variables)
        _err("Variables must be of type `String`.")
    end

    # Semantics.
    if !in(json.sense, ["feas", "min", "max"])
        _err("Invalid `sense` field. Must be one of `feas`, `min`, or `max`.")
    end

    for (T, k) in [(String, :print_format), (Number, :time_limit_sec)]
        if haskey(json.options, k) && !isa(json.options[k], T)
            _err("Invalid `options.$(k)` field. Must be of type `$(T)`.")
        end
    end

    for k in [:silent, :print_only]
        if haskey(json.options, k)
            val = json.options[k]
            # We allow `0` and `1` for convenience.
            if !isa(val, Bool) && val isa Number && val != 0 && val != 1
                _err("Invalid `options.$(k)` field. Must be a boolean.")
            end
        end
    end

    if json.sense == "feas"
        if !isempty(json.objectives)
            _err("Objectives must be empty when `sense` is `feas`.")
        end
    else
        if length(json.objectives) != 1
            _err("Only a single objective is supported.")
        end
    end

    return out
end

# Set solver options.
function set_options!(model::MOI.ModelLike, options::JSON3.Object)#::Nothing
    if MOI.supports(model, MOI.TimeLimitSec())
        # Set time limit, defaulting to 5min.
        MOI.set(model, MOI.TimeLimitSec(), Float64(get(options, :time_limit_sec, 300.0)))
    end

    for (key, val) in options
        if key in [:solver, :print_format, :print_only, :time_limit_sec]
            # Skip - these are handled elsewhere.
            continue
        elseif key == :silent
            MOI.set(model, MOI.Silent(), Bool(val))
        else
            attr = MOI.RawOptimizerAttribute(string(key))
            if !MOI.supports(model, attr)
                throw(Error(Unsupported, "Unsupported attribute: `$key`:`$val`."))
            end
            MOI.set(model, attr, val)
        end
    end

    return nothing
end

function load!(model::MOI.ModelLike, json::Request, T::Type, solver_info::Dict{Symbol,Any})#::Nothing
    # handle variables
    vars_map = Dict{String,MOI.VariableIndex}()
    vars = MOI.add_variables(model, length(json.variables))

    for (vi, si) in zip(vars, json.variables)
        vars_map[si] = vi
        MOI.set(model, MOI.VariableName(), vi, si) # TODO make this optional
    end

    # handle variable domains and constraints
    for con in json.constraints
        add_cons!(T, model, con, vars_map, solver_info)
    end

    # handle objective function
    if json.sense != "feas"
        add_obj!(T, model, json.sense, only(json.objectives), vars_map, solver_info)
    end

    return nothing
end

end # module SolverAPI
