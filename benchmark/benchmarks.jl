using FunctionProperties, BenchmarkTools

const SUITE = BenchmarkGroup()

# Representative functions with sample inputs
f_branching(x) = x > 0 ? x : -x
f_linear(x) = 2.0 * x
f_quad(x) = x^2 + 3.0x
f_smooth(x) = sin(x)
f_time(u, p, t) = p .* u .+ t
f_auto(u, p) = p .* u

x = 1.5
u = [1.0, 2.0]
p = [0.5, 0.5]

# =============================================================================
# Property queries (reflection-based analysis)
# =============================================================================

SUITE["queries"] = BenchmarkGroup()

SUITE["queries"]["hasbranching_true"] = @benchmarkable hasbranching($f_branching, $x)
SUITE["queries"]["hasbranching_false"] = @benchmarkable hasbranching($f_linear, $x)
SUITE["queries"]["islinear"] = @benchmarkable islinear($f_linear, $x)
SUITE["queries"]["isquadratic"] = @benchmarkable isquadratic($f_quad, $x)
SUITE["queries"]["issmooth"] = @benchmarkable issmooth($f_smooth, $x)
SUITE["queries"]["isautonomous_true"] = @benchmarkable isautonomous(
    $f_auto, $u, $p
)
SUITE["queries"]["isautonomous_false"] = @benchmarkable isautonomous(
    $f_time, $u, $p, 0.5
)
SUITE["queries"]["is_leaf"] = @benchmarkable is_leaf($f_linear)
