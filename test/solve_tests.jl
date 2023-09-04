@testsetup module SolverSetup
using SolverAPI

import HiGHS
import MiniZinc
import JSON3
import MathOptInterface as MOI

export run_solve, read_json

function _get_solver(solver_name::String)
    if solver_name == "MiniZinc"
        solver = MiniZinc.Optimizer{Int}("chuffed")
    else
        solver = HiGHS.Optimizer()
    end
    return solver
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
    json_names = ["feas_range", "min_range", "tiny_min", "tiny_feas", "tiny_infeas"]

    @testset "$j" for j in json_names
        input = read_json("inputs", j)
        output = run_solve(input)
        @test JSON3.read(output) == JSON3.read(read_json("outputs", j))
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

    for format in ["default", "latex", "mof", "lp", "mps", "nl"]
        options = Dict(:print_format => format)
        @test print_model(Dict(tiny_min..., :options => options)) isa String
    end

end

@testitem "validate" setup = [SolverSetup] begin
    using SolverAPI: deserialize, validate
    import JSON3

    # scenarios with incorrect format 
    format_err_json_names = [
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
        @test length(errors) >= 1
        @test all(errors[i] isa SolverAPI.Error for i in eachindex(errors))
    end
end
