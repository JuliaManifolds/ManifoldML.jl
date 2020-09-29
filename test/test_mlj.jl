include("utils.jl")
using MLJBase
using MLJModelInterface
using DataFrames

@testset "MLJ interoperability" begin
    M = Sphere(2)
    p1 = [1.0, 0.0, 0.0]
    p2 = [0.0, 1.0, 0.0]
    p3 = [0.0, sqrt(2), -sqrt(2)]

    X = DataFrame(pm = [p1, p2, p3], y = [1, 2, 1])
end
