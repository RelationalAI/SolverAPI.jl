# SolverAPI.jl

This Julia package takes serialized Rel SolverAPI library models,
translates these to enable access to external solvers, and returns
serialized solution information.

Currently, JSON is used for serialization, and only optimization
solvers and constraint programming solvers are supported.

## API
The main API is `solve` which takes a JSON request or Julia dictionary
and a solver and returns a response.

```julia
using SolverAPI

import MathOptInterface as MOI
import HiGHS

tiny_min = Dict(
        "version" => "0.1",
        "sense" => "min",
        "variables" => ["x"],
        "constraints" => [["==", "x", 1], ["Int", "x"]],
        "objectives" => ["x"],
    )

solve(tiny_min, HiGHS.Optimizer())
```

## Format

### Request
Request format example:
```json
{
  "version": "0.1",
  "sense": "min",
  "variables": [
    "x"
  ],
  "constraints": [
    [
      "==",
      "x",
      1
    ],
    [
      "Int",
      "x"
    ]
  ],
  "objectives": [
    "x"
  ],
  "options": {
    "silent": 0,
    "print_format": "LP"
  }
}
```
Required fields:
- `version`: [String] The version of the API that is used.
- `options`: [Array] Options, such as the time limit, if the
  model should be printed, or general solver attributes. For a
  complete list, please refer to the documentation of the solver.
- `sense`: [String] One of `feas`, `min`, or `max`
- `variables`: [Array] A list of variables that are used in the model.
- `constraints`: [Array] A list of constraints. Each constraint
  contains an operation and a set of arguments, such as `["==", "x", 1]`.
- `objectives`: [Array] The objective. 

### Response
Response format examples:
```json
{
  "version": "0.1",
  "results": [
    {
      "objective_value": 0.0,
      "primal_status": "FEASIBLE_POINT",
      "sol_names": [
        "\"x\""
      ],
      "sol_values": [
        1
      ]
    }
  ],
  "termination_status": "OPTIMAL"
}
```
Example with a model error:
```json
{
  "version": "0.1",
  "errors": [
    {
      "type": "InvalidModel",
      "message": "Objectives must be empty when `sense` is `feas`."
    }
  ],
  "termination_status": "OTHER_ERROR"
}
```

Example with printing and no solving:
```json
{
  "model_string": "{\"name\":\"MathOptFormat Model\",\"version\":{\"major\":1,\"minor\":4},\"variables\":[{\"name\":\"x\"}],\"objective\":{\"sense\":\"min\",\"function\":{\"type\":\"Variable\",\"name\":\"x\"}},\"constraints\":[{\"name\":\"c1\",\"function\":{\"type\":\"ScalarAffineFunction\",\"terms\":[{\"coefficient\":1.0,\"variable\":\"x\"}],\"constant\":0.0},\"set\":{\"type\":\"EqualTo\",\"value\":1.0}},{\"function\":{\"type\":\"Variable\",\"name\":\"x\"},\"set\":{\"type\":\"Integer\"}}]}",
  "termination_status": "OPTIMIZE_NOT_CALLED",
  "version": "0.1"
}
```

Required fields:
- `version`: [String] The version of the API that is used.
- `termination_status`: [String] The MOI termination status.

Optional fields:
- `results`: [Array] The results array. Depending on
  the optimization none, one, or multiple results will be
  present. Each result will contain multiple fields describing the
  solution, such as `objective_value`, `primal_status`,
  `sol_names`, and `sol_values`.
- `errors`: [Array] None, one, or multiple errors that were present.
- `model_string`: [String] If requested via `print_format` the model
as a string. 

