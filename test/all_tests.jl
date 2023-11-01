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
    solver_name = try
        json.options.solver
    catch
        "highs"
    end
    solver = get_solver(solver_name)
    (numerical_type, use_indicator) =
        solver isa MiniZinc.Optimizer ? (Int, false) : (Float64, true)
    output = solve(json, solver; numerical_type, use_indicator)
    return String(serialize(output))
end

read_json(in_out::String, name::String) =
    read(joinpath(@__DIR__, in_out, name * ".json"), String)

end # end of setup module.

@testitem "print" setup = [SolverSetup] begin
    using SolverAPI: print_model

    json = Dict(
        :version => "0.1",
        :sense => "min",
        :variables => ["x"],
        :constraints => [["==", "x", 1], ["Int", "x"]],
        :objectives => ["x"],
        :options => Dict(:print_format => "none"),
    )

    # check MOI model printing for each format
    @testset "$f" for f in ["moi", "latex", "mof", "lp", "mps", "nl"]
        json[:options][:print_format] = f
        @test print_model(json) isa String
    end
end

@testitem "solve" setup = [SolverSetup] begin
    import JSON3

    # names of JSON files in inputs/ and outputs/ folders
    json_names = [
        "feas_range",
        "min_interval",
        "tiny_min",
        "tiny_feas",
        "tiny_infeas",
        "simple_lp",
        "n_queens",
        "min_constant",
        "cons_true",
        "cons_false",
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
        # missing field options
        ("missing_options", "InvalidFormat"),
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
        # absolute_gap_tolerance out of range, e.g., -0.1
        ("abs_gap_out_of_range", "NotAllowed"),
        # relative_gap_tolerance must be within [0,1] 
        ("rel_gap_out_of_range", "NotAllowed"),
        # absolute_gap_tolerance set for non-mip solvers such as MiniZinc or CSP2SAT
        ("abs_gap_non_mip_solvers", "NotAllowed"),
        # relative_gap_tolerance set for non-mip solvers such as MiniZinc or CSP2SAT
        ("rel_gap_non_mip_solvers", "NotAllowed"),
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

    # check that expected error types are returned for each json input
    @testset "$j" for (j, es...) in json_names_and_errors
        result = JSON3.read(run_solve(read_json("inputs", j)))
        @test haskey(result, :errors) && length(result.errors) >= 1
        @test Set(e.type for e in result.errors) == Set(es)
    end
end

@testitem "large-model" setup = [SolverSetup] begin
    using SolverAPI: solve

    # setup linear objective model with n variables
    n = 20000
    vars = ["x$i" for i in 1:n]
    json = Dict(
        :version => "0.1",
        :sense => "max",
        :variables => vars,
        :constraints => [["Bin", v] for v in vars],
        :objectives => [vcat("+", 1, [["*", i, v] for (i, v) in enumerate(vars)])],
        :options => Dict(:solver => "HiGHS"),
    )

    # check that model is solved correctly without errors (particularly stack overflow)
    @testset "solve n=$n" begin
        output = solve(json, get_solver("HiGHS"))
        @test output isa Dict{String,Any}
        @test !haskey(output, "errors")
        @test output["termination_status"] == "OPTIMAL"
        @test output["results"][1]["objective_value"] ≈ 1 + div(n * (n + 1), 2)
    end
end
