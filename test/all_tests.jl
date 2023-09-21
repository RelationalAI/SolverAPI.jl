@testsetup module SolverSetup
using SolverAPI

import HiGHS
import MiniZinc
import JSON3
import MathOptInterface as MOI

export run_solve, read_json

function _get_solver(solver_name::String)
    solver_name_lower = lowercase(solver_name)
    if solver_name_lower == "minizinc"
        return MiniZinc.Optimizer{Int}("chuffed")
    elseif solver_name_lower == "highs"
        return HiGHS.Optimizer()
    else
        error("Solver $solver_name not supported.")
    end
end

function run_solve(input::String)
    json = deserialize(input)
    solver = _get_solver(json.options.solver)
    solution = solve(json, solver)
    return String(serialize(solution))
end

read_json(in_out::String, name::String) =
    read(joinpath(@__DIR__, in_out, name * ".json"), String)

end # end of setup module.

@testitem "solve" setup = [SolverSetup] begin
    using SolverAPI
    import JSON3

    # names of JSON files in inputs/ and outputs/ folders
    json_names = [
        "feas_range",
        "min_range",
        "tiny_min",
        "tiny_feas",
        "tiny_infeas",
        "simple_lp",
        "n_queens",
    ]

    @testset "$j" for j in json_names
        result = JSON3.read(run_solve(read_json("inputs", j)))
        @test result.solver_version isa String
        @test result.solve_time_sec isa Float64

        expect = JSON3.read(read_json("outputs", j))
        for (key, expect_value) in pairs(expect)
            @test result[key] == expect_value
        end
    end
end

@testitem "print" setup = [SolverSetup] begin
    using SolverAPI

    tiny_min = Dict(
        :version => "0.1",
        :sense => "min",
        :variables => ["x"],
        :constraints => [["==", "x", 1], ["Int", "x"]],
        :objectives => ["x"],
    )

    # test each format
    for format in ["moi", "latex", "mof", "lp", "mps", "nl"]
        options = Dict(:print_format => format)
        @test print_model(Dict(tiny_min..., :options => options)) isa String
    end
end

@testitem "validate" setup = [SolverSetup] begin
    using SolverAPI: deserialize, validate
    import JSON3

    # scenarios with incorrect format
    format_err_json_names = [
        # TODO fix: error not thrown for "unsupported_print_format"
        # "unsupported_print_format",     # print format not supported
        "feas_with_obj",                # objective provided for a feasibility problem
        "min_no_obj",                   # no objective function specified for a minimization problem
        "unsupported_sense",            # unsupported sense such as 'feasiblity'
        "obj_len_greater_than_1",       # length of objective greater than 1
        "incorrect_range_num_params",   # number of parameters not equal to 4
        "incorrect_range_step_not_1",   # step not one in range definition
        "vars_is_not_str",              # field variables is not a string
        "vars_is_not_arr",              # field variables is not an array
        "objs_is_not_arr",              # field objectives is not an array
        "cons_is_not_arr",              # field constraints is not an array
        "missing_vars",                 # missing field variables
        "missing_cons",                 # missing field constraints
        "missing_objs",                 # missing field objectives
        "missing_sense",                # missing field sense
        "missing_version",              # missing field version
    ]

    @testset "$j" for j in format_err_json_names
        input = deserialize(read_json("inputs", j))
        errors = validate(input)
        @test errors isa Vector{SolverAPI.Error}
        @test length(errors) >= 1
    end
end

@testitem "stress-test" setup = [SolverSetup] begin
    using SolverAPI
    import JSON3

    # names of JSON files in inputs/ and outputs/ folders
    json_names = [
        "nl_to_aff_or_quad_overflow",
    ]

    @testset "$j" for j in json_names
        input = read_json("inputs", j)
        output = JSON3.read(run_solve(input))
        @test output isa JSON3.Object
    end
end
