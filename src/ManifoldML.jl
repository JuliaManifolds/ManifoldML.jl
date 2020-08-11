module ManifoldML
using ManifoldsBase
using Manopt
import Manopt: initialize_solver!, step_solver!
using Manifolds: mean

include("kmeans.jl")

export initialize_solver!, step_solver!
export kmeans
end # module
