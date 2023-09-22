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
    if lowercase(get(json.options, :solver, "highs")) == "minizinc"
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

# Convert JSON array to MOI ScalarNonlinearFunction.
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

_is_snf(::Any) = false
_is_snf(::MOI.ScalarNonlinearFunction) = true

# Convert SNF to SAF/SQF{T} in place.
function nl_to_aff_or_quad!(::Type{T}, f::MOI.ScalarNonlinearFunction) where {T<:Real}
    stack = Tuple{MOI.ScalarNonlinearFunction,Int,MOI.ScalarNonlinearFunction}[]

    # Push the arguments in reverse order, s.t. we process them in the correct order.
    for i in length(f.args):-1:1
        if _is_snf(f.args[i])
            push!(stack, (f, i, f.args[i]))
        end
    end

    while !isempty(stack)
        parent, i, arg = pop!(stack)
        if any(_is_snf, arg.args)
            push!(stack, (parent, i, arg))
            for j in length(arg.args):-1:1
                if _is_snf(arg.args[j])
                    push!(stack, (arg, j, arg.args[j]))
                end
            end
        else
            # Leaf - all of `parent`'s arguments have been converted.
            parent.args[i] = _construct_aff_or_quad(T, arg)
        end
    end

    return _construct_aff_or_quad(T, f)
end

# Construct a new SAF or SQF from a SNF. Assume all arguments are already converted.
function _construct_aff_or_quad(::Type{T}, f::MOI.ScalarNonlinearFunction) where {T<:Real}
    for i in eachindex(f.args)
        f.args[i] = convert_if_needed(T, f.args[i])
    end

    if f.head == :^
        if length(f.args) == 2 && f.args[2] == 2
            return MOI.Utilities.operate(*, T, f.args[1], f.args[1])
        end
    else
        if f.head == :+
            # NOTE (dba) this is a workaround to avoid a
            # `StackOverflowError` coming from `MOI.Utilities.operate`
            # for large `args`. It is recursively defined:
            # https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Utilities/operate.jl#L323-L327
            # But this may change in
            # https://github.com/jump-dev/MathOptInterface.jl/pull/2285
            # NOTE (dba) It's important we use the in-place version to reduce allocations!
            plus_op(accum, x) = MOI.Utilities.operate!(+, T, accum, x)
            return reduce(plus_op, f.args)
        else
            h = get(_quad_ops, f.head, nothing)
            # All other operators do not take varargs. See
            # https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Utilities/operate.jl#L329
            if !isnothing(h)
                # TODO (dba) convert this assertion into a validation.
                @assert length(f.args) == 2
                return MOI.Utilities.operate(h, T, f.args[1], f.args[2])
            end
        end
    end
    return error() # Gets caught by canonicalize_SNF.
end

_quad_ops = Dict(:+ => +, :- => -, :* => *, :/ => /)

convert_if_needed(::Type{T}, f) where {T<:Real} = f
convert_if_needed(::Type{T}, f::Real) where {T<:Real} = convert(T, f)

# Convert SNF to SAF/SQF{T} if possible.
canonicalize_SNF(::Type{T}, f) where {T<:Real} = convert_if_needed(T, f)
function canonicalize_SNF(::Type{T}, f::MOI.ScalarNonlinearFunction) where {T<:Real}
    try
        f = nl_to_aff_or_quad!(T, f)
    catch
    end
    return f
end

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
        msg = "Objective function $(trunc_str(g)) isn't supported by this solver."
        throw(Error(Unsupported, msg))
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
    if head == "and"
        for i in eachindex(a)
            i == 1 && continue
            if a[i] isa Bool
                if !a[i]
                    throw(Error(InvalidModel, "Model is infeasible."))
                end
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
        if length(a) != 5
            throw(Error(InvalidModel, "The `range` constraint expects 4 arguments."))
        end
        v = json_to_snf(a[5], vars_map)
        _check_v_type(v)
        if !(a[2] isa Int && a[3] isa Int)
            throw(Error(InvalidModel, "The `range` constraint expects integer bounds."))
        end
        if a[4] != 1
            throw(Error(InvalidModel, "The `range` constraint expects a step size of 1."))
        end
        MOI.add_constraint(model, v, MOI.Integer())
        MOI.add_constraint(model, v, MOI.Interval{T}(a[2], a[3]))
    elseif head == "implies" && solver_info[:use_indicator]
        # TODO maybe just check if model supports indicator constraints
        # use an MOI indicator constraint
        if length(a) != 3
            throw(Error(InvalidModel, "The `implies` constraint expects 2 arguments."))
        end
        f = json_to_snf(a[2], vars_map)
        g = json_to_snf(a[3], vars_map)
        if !(f.head == :(==) && length(f.args) == 2)
            msg = "The first argument of the `implies` constraint expects to be converted to an equality SNF with 2 arguments."
            throw(Error(InvalidModel, msg))
        end

        v, b = f.args
        _check_v_type(v)
        if b != 1 && b != 0
            msg = "The second argument of the derived equality SNF from the `implies` constraint expects a binary variable."
            throw(Error(InvalidModel, msg))
        end

        A = (b == 1) ? MOI.ACTIVATE_ON_ONE : MOI.ACTIVATE_ON_ZERO
        S1 = get(ineq_to_moi, g.head, nothing)
        if isnothing(S1) || length(g.args) != 2
            msg = "The second argument of the `implies` constraint expects to be converted to an (in)equality SNF with 2 arguments."
            throw(Error(InvalidModel, msg))
        end

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
                msg = "Constraint $(trunc_str(g)) in $(trunc_str(s)) isn't supported by this solver."
                throw(Error(Unsupported, msg))
            end
            ci = MOI.Utilities.normalize_and_add_constraint(model, g, s)
        end
    end
    return nothing
end

_check_v_type(::MOI.VariableIndex) = nothing
_check_v_type(_) =
    throw(Error(InvalidModel, "$v must be a `MOI.VariableIndex`, not $(typeof(v))."))

ineq_to_moi = Dict(:<= => MOI.LessThan, :>= => MOI.GreaterThan, :(==) => MOI.EqualTo)

function shift_terms(::Type{T}, args::Vector) where {T<:Real}
    @assert length(args) == 2 # This should never happen.
    g1 = canonicalize_SNF(T, args[1])
    g2 = canonicalize_SNF(T, args[2])
    return MOI.Utilities.operate(-, T, g1, g2)
end

# Convert object to string and truncate string length if too long.
function trunc_str(f::Union{MOI.AbstractScalarFunction,MOI.AbstractScalarSet})
    f_str = string(f)
    if length(f_str) > 256
        f_str = f_str[1:256] * " ... (truncated)"
    end
    return f_str
end

end # module SolverAPI
