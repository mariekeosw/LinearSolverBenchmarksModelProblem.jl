# Archived results: Krylov.jl stopping criterion

Results of the iterative solvers from the first runs, in which preconditioned CG stopped with
Krylov.jl's default criterion ‖r_k‖_{M⁻¹} ≤ atol + rtol ‖r_0‖_{M⁻¹} (atol = rtol = √ε_mach).
Since this norm depends on the preconditioner, the methods stopped at different Euclidean
residuals (up to 1e-3 for AMG in 2D). The current results in `results/` and `results_scaled/`
use the same criterion ‖b - Au_k‖₂ ≤ 1e-8 ‖b‖₂ for all iterative solvers.

- `unscaled/`: matrix L + δI (including PyAMG)
- `scaled/`: matrix L + δh²I
- `pdf-files/`: figures of these runs (the direct solvers are unchanged and remain in `results/`)
