module ManifoldML
import ManifoldsBase:
    Manifold
import Manopt: Options, Problem, solve, StoppingCriterion, StopAfterIteration
import Manifolds: mean

include("kmeans.jl")

export kmeans
end # module
