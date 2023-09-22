@testsetup module SolverSetup
using SolverAPI

import HiGHS
import MiniZinc
import JSON3
import MathOptInterface as MOI

export get_solver, run_solve, read_json

function get_solver(solver_name::String)
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
    solver = get_solver(json.options.solver)
    output = solve(json, solver)
    return String(serialize(output))
end

read_json(in_out::String, name::String) =
    read(joinpath(@__DIR__, in_out, name * ".json"), String)

end # end of setup module.

@testitem "print" setup = [SolverSetup] begin
    using SolverAPI: print_model

    model = Dict(
        :version => "0.1",
        :sense => "min",
        :variables => ["x"],
        :constraints => [["==", "x", 1], ["Int", "x"]],
        :objectives => ["x"],
    )

    # check MOI model printing for each format
    @testset "$f" for f in ["moi", "latex", "mof", "lp", "mps", "nl"]
        request = Dict(model..., :options => Dict(:print_format => f))
        @test print_model(request) isa String
    end
end

@testitem "solve" setup = [SolverSetup] begin
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

    # solve and check output is expected for each input json file
    @testset "$j" for j in json_names
        output = JSON3.read(run_solve(read_json("inputs", j)))
        @test output.solver_version isa String
        @test output.solve_time_sec isa Float64
        expect = JSON3.read(read_json("outputs", j))
        for (key, expect_value) in pairs(expect)
            @test output[key] == expect_value
        end
    end
end

@testitem "errors" setup = [SolverSetup] begin
    using SolverAPI
    import JSON3

    # names of JSON files in inputs/ folder and expected error types
    json_names_and_errors = [
        # missing field variables
        ("missing_vars", "InvalidFormat"),
        # missing field constraints
        ("missing_cons", "InvalidFormat"),
        # missing field objectives
        ("missing_objs", "InvalidFormat"),
        # missing field sense
        ("missing_sense", "InvalidFormat"),
        # missing field version
        ("missing_version", "InvalidFormat"),
        # field variables is not a string
        ("vars_is_not_str", "InvalidFormat"),
        # field variables is not an array
        ("vars_is_not_arr", "InvalidFormat"),
        # field objectives is not an array
        ("objs_is_not_arr", "InvalidFormat"),
        # field constraints is not an array
        ("cons_is_not_arr", "InvalidFormat"),
        # length of objective greater than 1
        ("obj_len_greater_than_1", "InvalidFormat"),
        # objective provided for a feasibility problem
        ("feas_with_obj", "InvalidFormat"),
        # no objective function specified for a minimization problem
        ("min_no_obj", "InvalidFormat"),
        # unsupported sense such as 'feasibility'
        ("unsupported_sense", "InvalidFormat"),
        # range: wrong number of args
        ("incorrect_range_num_params", "InvalidModel"),
        # range: step not one
        ("incorrect_range_step_not_1", "InvalidModel"),
        # unsupported objective function type
        ("unsupported_obj_type", "Unsupported"),
        # unsupported constraint function type
        ("unsupported_con_type", "Unsupported"),
        # unsupported constraint sign
        ("unsupported_con_sign", "Unsupported"),
        # unsupported operator
        ("unsupported_operator", "Unsupported"),
        # unsupported solver option
        ("unsupported_solver_option", "Unsupported"),
        # print format not supported
        ("unsupported_print_format", "Unsupported"),
    ]

    @testset "$j" for (j, es...) in json_names_and_errors
        result = JSON3.read(run_solve(read_json("inputs", j)))
        @test haskey(result, :errors) && length(result.errors) >= 1
        @test Set(e.type for e in result.errors) == Set(es)
    end
end

@testitem "large-model" setup = [SolverSetup] begin
    using SolverAPI: solve

    # setup linear objective model with n variables
    n = 1000
    vars = ["x$i" for i in 1:n]
    model = Dict(
        :version => "0.1",
        :sense => "max",
        :variables => vars,
        :constraints => [["Bin", v] for v in vars],
        :objectives => [vcat("+", 1, [["*", i, v] for (i, v) in enumerate(vars)])],
    )

    # check that model is solved correctly without errors
    output = solve(model, get_solver("HiGHS"))
    @test output isa Dict{String,Any}
    @test !haskey(output, "errors")
    @test output["termination_status"] == "OPTIMAL"
    @test output["results"][1]["objective_value"] ≈ 1 + div(n * (n + 1), 2)
end
