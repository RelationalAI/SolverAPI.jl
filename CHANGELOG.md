# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3]

  - Make `options` field optional [#22](https://github.com/RelationalAI/SolverAPI.jl/pull/22)
  - Remove `names` field in results [#20](https://github.com/RelationalAI/SolverAPI.jl/pull/20)
  - Update format to use JSON vector for relational appl constraint
  args [#21](https://github.com/RelationalAI/SolverAPI.jl/pull/21)

## [0.2.2]

  - Allow empty array (for no constraints) [#18](https://github.com/RelationalAI/SolverAPI.jl/pull/18)
  - Adding relative & absolute mip gaps [#17](https://github.com/RelationalAI/SolverAPI.jl/pull/17)
  - Support in / table / relational application constraints [#16](https://github.com/RelationalAI/SolverAPI.jl/pull/16)

## [0.2.1]

  - Support `interval` constraint [#14](https://github.com/RelationalAI/SolverAPI.jl/pull/14)
  - Handle constant objectives and constraints containing no
    variables
    [#13](https://github.com/RelationalAI/SolverAPI.jl/pull/13)
  - Improve error messages for empty objective [#12](https://github.com/RelationalAI/SolverAPI.jl/pull/12)

## [0.2.0]

  - Improve test coverage, error messages, and others [#1](https://github.com/RelationalAI/SolverAPI.jl/pull/1), [#7](https://github.com/RelationalAI/SolverAPI.jl/pull/7),
    [#8](https://github.com/RelationalAI/SolverAPI.jl/pull/8), [#10](https://github.com/RelationalAI/SolverAPI.jl/pull/10)
  - Fix `StackOverflowError` and improve performance [#6](https://github.com/RelationalAI/SolverAPI.jl/pull/6)
  - Return solve time and solver version [#5](https://github.com/RelationalAI/SolverAPI.jl/pull/5)
  - Add default time limit [#2](https://github.com/RelationalAI/SolverAPI.jl/pull/2)

## [0.1.0]

  - Initial release
