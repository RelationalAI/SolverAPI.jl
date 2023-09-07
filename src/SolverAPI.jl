"""
    SolverAPI

This module provides an interface to solve optimization problems.

Exports `solve`, `print_model`, `response`, `serialize`, and
`deserialize`.

Our format is based on JSON. For examples, please see `tests/inputs`
and `tests/outputs`.
"""
module SolverAPI

import MathOptInterface as MOI
import JSON3

export serialize, deserialize, solve, print_model, response

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
    response([request], [model]; kw...)::Dict{String,Any}

Return a response

## Args
- `request::SolverAPI.Request`: [optionally] The request that was received.
- `model::MOI.ModelLike`: [optionally] The model that was solved.

## Keyword Args
- `version::String`: The version of the API.
- `termination_status::MOI.TerminationStatus`: The termination status of the solver.
- `errors::SolverAPI.Error`: A vector of `Error`s.
"""
function response(;
    version="0.1",
    termination_status=MOI.OTHER_ERROR,
    errors::Vector{Error}=Error[],
)
    pairs = Pair{String,Any}["version"=>version, "termination_status"=>termination_status]
    if !isempty(errors)
        push!(pairs, "errors" => errors)
    end
    return Dict(pairs)
end
response(error::Error; kw...) = response(; errors=[error], kw...)
function response(json::Request, model::MOI.ModelLike; version="0.1", kw...)
    res = Dict{String,Any}()

    res["version"] = version

    status = MOI.get(model, MOI.TerminationStatus())
    res["termination_status"] = status

    options = get(() -> Dict{String,Any}(), json, :options)
    format = get(options, :print_format, nothing)
    if !isnothing(format)
        # print model
        res["model_string"] = print_model(model, format)
    end

    if Bool(get(options, :print_only, false))
        return res
    end

    result_count = MOI.get(model, MOI.ResultCount())

    results = [Dict{String,Any}() for _ = 1:result_count]

    for idx = 1:result_count
        results[idx]["primal_status"] = MOI.get(model, MOI.PrimalStatus(idx))

        if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            vars = MOI.get(model, MOI.ListOfVariableIndices())
            sol = MOI.get(model, MOI.VariablePrimal(), vars)
            results[idx]["names"] = [string('\"', v, '\"') for v in json.variables]
            results[idx]["values"] = sol

            if json.sense != "feas"
                results[idx]["objective_value"] = MOI.get(model, MOI.ObjectiveValue())
            end
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
function serialize(response::Response)
    return Vector{UInt8}(JSON3.write(response))
end
serialize(io::IO, response::Response) = JSON3.write(io, response)

"""
    deserialize(body::Vector{UInt8})::Request
    deserialize(body::String)::Request

Deserialize `body` to an `Request`. Returns an instances that can be
used with `solve` or `print_model`.
"""
function deserialize(body::Vector{UInt8})
    return JSON3.read(body)
end
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
    if length(errors) > 0
        return response(; errors)
    end

    try
        T, solver_info, model = initialize(json, solver)
        load!(json, T, solver_info, model)
        fn(model)
        options = get(() -> Dict{String,Any}(), json, :options)
        if Bool(get(options, :print_only, false))
            return response(json, model)
        end
        MOI.optimize!(model)
        return response(json, model)
    catch e
        if e isa MOI.UnsupportedError
            throw(Error(Unsupported, sprint(Base.showerror, e)))
        elseif e isa MOI.NotAllowedError
            throw(Error(NotAllowed, sprint(Base.showerror, e)))
        elseif e isa MOI.InvalidIndex ||
               e isa MOI.ResultIndexBoundsError ||
               e isa MOI.ScalarFunctionConstantNotZero ||
               e isa MOI.LowerBoundAlreadySet ||
               e isa MOI.UpperBoundAlreadySet ||
               e isa MOI.OptimizeInProgress ||
               e isa MOI.InvalidCallbackUsage
            throw(Error(Domain, sprint(Base.showerror, e)))
        elseif e isa ErrorException
            throw(Error(Other, e.msg))
        else
            rethrow()
        end
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
- `default`
- `LaTeX`
- `MOF`
- `LP`
- `MPS`
- `NL`
"""
function print_model(model::MOI.ModelLike, format::String)
    format = lowercase(format)
    if format == "default" || format == "latex"
        mime = MIME(format == "latex" ? "text/latex" : "text/plain")
        options =
            MOI.Utilities._PrintOptions(mime; simplify_coefficients=true, print_types=false)
        return sprint() do io
            MOI.Utilities._print_model(io, options, model)
        end
    elseif format == "latex"
        # NOTE: there are options for latex_formulation, e.g. don't print MOI function/set types
        # https://jump.dev/MathOptInterface.jl/dev/submodules/Utilities/reference/#MathOptInterface.Utilities.latex_formulation
        return sprint(print, MOI.Utilities.latex_formulation(model))
    elseif format == "mof"
        options = (; format=MOI.FileFormats.FORMAT_MOF, print_compact=true)
    elseif format == "lp"
        options = (; format=MOI.FileFormats.FORMAT_LP)
    elseif format == "mps"
        options = (; format=MOI.FileFormats.FORMAT_MPS)
    elseif format == "nl"
        options = (; format=MOI.FileFormats.FORMAT_NL)
    else
        throw(Error(Unsupported, "File type $format not supported."))
    end
    dest = MOI.FileFormats.Model(; options...)
    MOI.copy_to(dest, model)
    return sprint(write, dest)
end
function print_model(request::Request; T=Float64)
    errors = validate(request)

    if length(errors) > 0
        throw(CompositeException(errors))
    end

    options = get(() -> Dict{String,Any}(), request, :options)
    format = get(options, :print_format, "default")

    use_indicator = T == Float64
    solver_info = Dict{Symbol,Any}(:use_indicator => use_indicator)
    model = MOI.Utilities.Model{T}()

    load!(request, T, solver_info, model)
    return print_model(model, format)
end
print_model(request::Dict) = print_model(JSON3.read(JSON3.write(request)))


# Internal
# ========================================================================================

function validate(json::Request)#::Vector{Error}
    out = Error[]
    valid_shape = true

    # Syntax.
    for k in [:version, :sense, :variables, :constraints, :objectives]
        if !haskey(json, k)
            valid_shape = false
            push!(out, Error(InvalidFormat, "Missing required field `$(k)`."))
        end
    end

    # If the shape is not valid we can't continue.
    if !valid_shape
        return out
    end

    if !isa(json.version, String)
        push!(out, Error(InvalidFormat, "Invalid `version` field. Must be a string."))
    end

    if json.version != "0.1"
        push!(
            out,
            Error(
                InvalidFormat,
                """Invalid version `$(repr(json.version))`. Only `"0.1"` is supported.""",
            ),
        )
    end

    if haskey(json, :options) && !isa(json.options, JSON3.Object)
        push!(out, Error(InvalidFormat, "Invalid `options` field. Must be an object."))
    end

    for k in [:variables, :constraints, :objectives]
        if !isa(json[k], JSON3.Array)
            push!(out, Error(InvalidFormat, "Invalid `$(k)` field. Must be an array."))
        end
    end

    if any(si -> !isa(si, String), json.variables)
        push!(out, Error(InvalidFormat, "Variables must be of type `String`."))
    end

    # Semantics.
    if !in(json.sense, ["feas", "min", "max"])
        push!(
            out,
            Error(
                InvalidModel,
                "Invalid `sense` field. Must be one of `feas`, `min`, or `max`.",
            ),
        )
    end

    if haskey(json, :options)
        for (T, k) in [(String, :print_format), (Number, :time_limit_sec)]
            if haskey(json.options, k) && !isa(json.options[k], T)
                push!(
                    out,
                    Error(
                        InvalidFormat,
                        "Invalid `options.$(k)` field. Must be of type `$(T)`.",
                    ),
                )
            end
        end

        for k in [:silent, :print_only]
            if haskey(json.options, k)
                val = json.options[k]
                # We allow `0` and `1` for convenience.
                if !isa(val, Bool) && val isa Number && val != 0 && val != 1
                    push!(
                        out,
                        Error(
                            InvalidFormat,
                            "Invalid `options.$(k)` field. Must be a boolean.",
                        ),
                    )
                end
            end
        end
    end

    if json.sense == "feas"
        if !isempty(json.objectives)
            push!(
                out,
                Error(InvalidModel, "Objectives must be empty when `sense` is `feas`."),
            )
        end
    else
        if length(json.objectives) != 1
            push!(out, Error(InvalidModel, "Only a single objective is supported."))
        end
    end

    for con in json.constraints
        if first(con) == "range"
            if length(con) != 5
                push!(
                    out,
                    Error(InvalidModel, "The `range` constraint expects 4 arguments."),
                )
            elseif con[4] != 1
                push!(
                    out,
                    Error(InvalidModel, "The `range` constraint expects a step size of 1."),
                )
            end
        end
    end

    return out
end


function initialize(json::Request, solver::MOI.AbstractOptimizer)#::Tuple{Type, Dict{Symbol, Any}, MOI.ModelLike}
    solver_info = Dict{Symbol,Any}()

    # TODO (dba) `SolverAPI.jl` should be decoupled from any solver
    # specific code.
    options = get(() -> Dict{String,Any}(), json, :options)
    if lowercase(get(options, :solver, "highs")) == "minizinc"
        T = Int
        solver_info[:use_indicator] = false
    else
        T = Float64
        solver_info[:use_indicator] = true
    end

    model = MOI.instantiate(() -> solver; with_cache_type=T, with_bridge_type=T)

    for (key, val) in options
        if key in [:solver, :print_format, :print_only]
            continue
        elseif key == :time_limit_sec
            MOI.set(model, MOI.TimeLimitSec(), Float64(val))
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

    return (T, solver_info, model)
end

function load!(json::Request, T::Type, solver_info::Dict{Symbol,Any}, model::MOI.ModelLike)#::Nothing
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

# convert JSON array to MOI ScalarNonlinearFunction
function json_to_snf(a::JSON3.Array, vars_map::Dict)
    length(a) > 0 || throw(Error(InvalidModel, "The given JSON array `$a` is empty."))

    head = a[1]
    args = Any[json_to_snf(a[i], vars_map) for i in eachindex(a) if i != 1]

    head isa String || return args
    if head == "range"
        # TODO handle variables in different positions, etc
        # TODO handle as interval constraint?
        lb, ub, step, x = args
        step == 1 || throw(Error(NotAllowed, "Step size $step is not supported."))
        return MOI.ScalarNonlinearFunction(
            :∧,
            Any[
                MOI.ScalarNonlinearFunction(:<=, Any[lb, x]),
                MOI.ScalarNonlinearFunction(:<=, Any[x, ub]),
            ],
        )
    elseif head == "and"
        head = "forall"
        args = Any[args]
    elseif head == "or"
        head = "exists"
        args = Any[args]
    elseif head == "not"
        head = "!"
    elseif head == "implies"
        head = "=>"
    elseif head == "natural_exp"
        head = "exp"
    elseif head == "natural_log"
        head = "log"
    elseif head == "alldifferent"
        args = Any[args]
    elseif head == "count"
        args = Any[args]
    elseif head == "max"
        head = "maximum"
        args = Any[args]
    end
    return MOI.ScalarNonlinearFunction(Symbol(head), args)
end

json_to_snf(a::String, vars_map::Dict) = vars_map[a]
json_to_snf(a::Real, ::Dict) = a

# convert SNF to SAF/SQF{T}
function nl_to_aff_or_quad(::Type{T}, f::MOI.ScalarNonlinearFunction) where {T<:Real}
    args = nl_to_aff_or_quad.(T, f.args)
    if !any(Base.Fix2(isa, MOI.ScalarNonlinearFunction), args)
        if f.head == :^
            if length(args) == 2 && args[2] == 2
                return MOI.Utilities.operate(*, T, args[1], args[1])
            end
        else
            h = get(_quad_ops, f.head, nothing)
            isnothing(h) || return MOI.Utilities.operate(h, T, args...)
        end
    end
    throw(Error(Domain, "Function $f cannot be converted to linear or quadratic form."))
end

nl_to_aff_or_quad(::Type{<:Real}, f::MOI.VariableIndex) = f
nl_to_aff_or_quad(::Type{T}, f::T) where {T<:Real} = f
nl_to_aff_or_quad(::Type{T}, f::Real) where {T<:Real} = convert(T, f)

_quad_ops = Dict(:+ => +, :- => -, :* => *, :/ => /)

function canonicalize_SNF(::Type{T}, f) where {T<:Real}
    try
        f = nl_to_aff_or_quad(T, f)
    catch
    end
    return f
end

# TODO(cdc) experiment more:
# function canonicalize_SNF(::Type{T}, f) where {T <: Real}
#     try
#         f = convert(MOI.ScalarQuadraticFunction{T}, f)
#         try
#             f = convert(MOI.ScalarAffineFunction{T}, f)
#         catch end
#     catch end
#     return f
# end

function add_obj!(
    ::Type{T},
    model::MOI.ModelLike,
    sense::String,
    a::Union{String,JSON3.Array},
    vars_map::Dict,
    ::Dict,
) where {T<:Real}
    if sense == "min"
        moi_sense = MOI.MIN_SENSE
    elseif sense == "max"
        moi_sense = MOI.MAX_SENSE
    end
    MOI.set(model, MOI.ObjectiveSense(), moi_sense)

    g = canonicalize_SNF(T, json_to_snf(a, vars_map))
    g_type = MOI.ObjectiveFunction{typeof(g)}()
    if !MOI.supports(model, g_type)
        throw(Error(Unsupported, "Objective function $g isn't supported by this solver."))
    end
    MOI.set(model, g_type, g)
    return nothing
end

function add_cons!(
    ::Type{T},
    model::MOI.ModelLike,
    a::JSON3.Array,
    vars_map::Dict,
    solver_info::Dict,
) where {T<:Real}
    head = a[1]

    function _check_v_type(v)
        if !(v isa MOI.VariableIndex)
            throw(
                Error(
                    InvalidModel,
                    "Variable $v must be of type MOI.VariableIndex, not $(typeof(v)).",
                ),
            )
        end
    end

    if head == "and"
        for i in eachindex(a)
            i == 1 && continue
            if a[i] isa Bool
                a[i] || push!(out, Error(InvalidModel, "Model is infeasible."))
            else
                add_cons!(T, model, a[i], vars_map, solver_info)
            end
        end
    elseif head == "Int"
        v = json_to_snf(a[2], vars_map)
        _check_v_type(v)
        MOI.add_constraint(model, v, MOI.Integer())
    elseif head == "Bin"
        v = json_to_snf(a[2], vars_map)
        _check_v_type(v)
        MOI.add_constraint(model, v, MOI.ZeroOne())
    elseif head == "Float"
    elseif head == "Nonneg"
        v = json_to_snf(a[2], vars_map)
        _check_v_type(v)
        MOI.add_constraint(model, v, MOI.GreaterThan(zero(T)))
    elseif head == "PosNegOne"
        v = json_to_snf(a[2], vars_map)
        _check_v_type(v)
        # TODO only for MiniZinc
        MOI.add_constraint(model, v, MOI.Integer())
        f = MOI.ScalarNonlinearFunction(:abs, Any[v])
        MOI.add_constraint(model, f, MOI.EqualTo(1))
    elseif head == "range"
        v = json_to_snf(a[5], vars_map)

        (a[2] isa Int && a[3] isa Int) ||
            throw(Error(InvalidModel, "The `range` constraint expects integer bounds."))
        _check_v_type(v)

        MOI.add_constraint(model, v, MOI.Integer())
        MOI.add_constraint(model, v, MOI.Interval{T}(a[2], a[3]))
    elseif head == "implies" && solver_info[:use_indicator]
        # TODO maybe just check if model supports indicator constraints
        # use an MOI indicator constraint
        length(a) == 3 ||
            throw(Error(InvalidModel, "The `implies` constraint expects 2 arguments."))
        f = json_to_snf(a[2], vars_map)
        g = json_to_snf(a[3], vars_map)
        (f.head == :(==) && length(f.args) == 2) || throw(
            Error(
                InvalidModel,
                "The first argument of the `implies` constraint expects to be converted to an equality SNF with 2 arguments.",
            ),
        )

        v, b = f.args
        _check_v_type(v)
        (b == 1 || b == 0) || throw(
            Error(
                InvalidModel,
                "The second argument of the derived equality SNF from the `implies` constraint expects a binary variable.",
            ),
        )

        A = (b == 1) ? MOI.ACTIVATE_ON_ONE : MOI.ACTIVATE_ON_ZERO
        S1 = get(ineq_to_moi, g.head, nothing)
        (!isnothing(S1) && length(g.args) == 2) || throw(
            Error(
                InvalidModel,
                "The second argument of the `implies` constraint expects to be converted to an (in)equality SNF with 2 arguments.",
            ),
        )

        h = shift_terms(T, g.args)
        vaf = MOI.Utilities.operate(vcat, T, v, h)
        MOI.add_constraint(model, vaf, MOI.Indicator{A}(S1(zero(T))))
    else
        f = json_to_snf(a, vars_map)
        S = get(ineq_to_moi, f.head, nothing)
        if isnothing(S)
            # CSP constraint
            ci = MOI.add_constraint(model, f, MOI.EqualTo(1))
        else
            # (in)equality constraint
            g = shift_terms(T, f.args)
            s = S(zero(T))
            if !MOI.supports_constraint(model, typeof(g), typeof(s))
                throw(
                    Error(
                        Unsupported,
                        "Constraint $g in $s isn't supported by this solver.",
                    ),
                )
            end
            ci = MOI.Utilities.normalize_and_add_constraint(model, g, s)
        end
    end
    return nothing
end

ineq_to_moi = Dict(:<= => MOI.LessThan, :>= => MOI.GreaterThan, :(==) => MOI.EqualTo)

function shift_terms(::Type{T}, args::Vector) where {T<:Real}
    @assert length(args) == 2 # This should never happen. 
    g1 = canonicalize_SNF(T, args[1])
    g2 = canonicalize_SNF(T, args[2])
    return MOI.Utilities.operate(-, T, g1, g2)
end

end # module SolverAPI
