using LinearSolverBenchmarks
using Aqua

# stale_deps = false, since scripts depend on packages which are not used in src
Aqua.test_all(LinearSolverBenchmarks, stale_deps = false, persistent_tasks = false)

# How to test the package?
# 1) julia package mode "]"
# 2) enter "test"
