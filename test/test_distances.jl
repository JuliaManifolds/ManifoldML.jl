include("utils.jl")
using Distances

Random.seed!(42)

@testset "Distances" begin
    M = SymmetricPositiveDefinite(3)
    A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
    ptsF = [#
        [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1],
        [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1],
        A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6)),
    ]
    dist = ManifoldML.RiemannianDistance(M)
    @test Distances.evaluate(dist, reshape(ptsF[1], 9), reshape(ptsF[3], 9)) ≈
          distance(M, ptsF[1], ptsF[3])

    point_matrix = reduce(hcat, map(a -> reshape(a, 9), ptsF))
    dists = pairwise(dist, point_matrix)
    for i in 1:3, j in 1:3
        @test dists[i, j] ≈ distance(M, ptsF[i], ptsF[j]) atol = 1e-15
    end
end
