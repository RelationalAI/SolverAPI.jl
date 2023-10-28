
# Add objective function to model from objective JSON.
function add_obj!(
    ::Type{T},
    model::MOI.ModelLike,
    sense::String,
    a::Any,
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
    if !(g isa MOI.AbstractScalarFunction)
        g = convert(MOI.ScalarAffineFunction{T}, g)
    end
    g_type = MOI.ObjectiveFunction{typeof(g)}()
    if !MOI.supports(model, g_type)
        msg = "Objective function $(trunc_str(g)) isn't supported by this solver."
        throw(Error(Unsupported, msg))
    end
    MOI.set(model, g_type, g)
    return nothing
end

# Add constraints to model from constraint JSON.
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
    elseif head == "interval"
        if length(a) != 4
            throw(Error(InvalidModel, "The `interval` constraint expects 3 arguments."))
        end
        v = json_to_snf(a[4], vars_map)
        _check_v_type(v)
        if !(a[2] isa Number && a[3] isa Number)
            throw(Error(InvalidModel, "The `interval` constraint expects number bounds."))
        end
        MOI.add_constraint(model, v, MOI.Interval{T}(T(a[2]), T(a[3])))
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
    elseif head == "in"
        if length(a) < 3
            msg = "The relational application constraint expects at least two arguments."
            throw(Error(InvalidModel, msg))
        end
        exprs = [json_to_snf(a[i], vars_map) for i in 2:length(a)-1]
        vecs = a[end]
        if !(vecs isa JSON3.Array) || !all(length(row) == length(exprs) for row in vecs)
            msg = "The relational application constraint is malformed."
            throw(Error(InvalidModel, msg))
        end
        mat = convert(Matrix{T}, stack(vecs, dims = 1))
        vaf = MOI.Utilities.operate(vcat, T, exprs...)
        MOI.add_constraint(model, vaf, MOI.Table(mat))
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
    g = MOI.Utilities.operate(-, T, g1, g2)
    if !(g isa MOI.AbstractScalarFunction)
        g = convert(MOI.ScalarAffineFunction{T}, g)
    end
    return g
end

# Convert object to string and truncate string length if too long.
function trunc_str(f::Union{MOI.AbstractScalarFunction,MOI.AbstractScalarSet})
    f_str = string(f)
    if length(f_str) > 256
        f_str = f_str[1:256] * " ... (truncated)"
    end
    return f_str
end

# Convert JSON array to MOI ScalarNonlinearFunction.
function json_to_snf(a::JSON3.Array, vars_map::Dict)
    length(a) > 0 || throw(Error(InvalidModel, "The given JSON array `$a` is empty."))

    head = a[1]
    args = Any[json_to_snf(a[i], vars_map) for i in eachindex(a) if i != 1]

    head isa String || return args
    if head == "and"
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

# Convert SNF to SAF/SQF{T} if possible.
canonicalize_SNF(::Type{T}, f) where {T<:Real} = convert_if_needed(T, f)
function canonicalize_SNF(::Type{T}, f::MOI.ScalarNonlinearFunction) where {T<:Real}
    try
        f = nl_to_aff_or_quad!(T, f)
    catch
    end
    return f
end

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

_is_snf(::Any) = false
_is_snf(::MOI.ScalarNonlinearFunction) = true

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
