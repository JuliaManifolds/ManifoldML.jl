module ManifoldML
using ManifoldsBase
using Manopt
import Manopt: initialize_solver!, step_solver!
using Manifolds: mean
using MLJBase
using MLJModelInterface
using MLJScientificTypes

using Requires

using Tables # needed for MLJ

include("kmeans.jl")
include("tangent_transformer.jl")

function __init__()
    @require Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7" begin
        using .Distances
        include("distances.jl")
    end

    return nothing
end

export initialize_solver!, step_solver!
export kmeans
end # module
