include("utils.jl")

Random.seed!(42)

@testset "K-Means" begin
    M = Sphere(2)
    p1 = 1 / sqrt(3) .* [1.0, 1.0, 1.0]
    p2 = 1 / sqrt(3) .* [-1.0, -1.0, 1.0]
    c1 = [exp(M, p1, project(M, p1, 0.7 .* (rand(3) .- 0.5))) for i in 1:20]
    c2 = [exp(M, p2, project(M, p2, 0.7 .* (rand(3) .- 0.5))) for i in 1:20]

    pts = [p1, p2, c1..., c2...]
    o = kmeans(M, pts; num_centers = 2)
    @test distance(M, p1, o.centers[1]) ≈ 0
    @test distance(M, p2, o.centers[2]) ≈ 0
    @test sum(o.assignment .== 1) == 21
    @test sum(o.assignment .== 2) == 21
end
