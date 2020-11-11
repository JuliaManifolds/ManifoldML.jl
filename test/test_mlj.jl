include("utils.jl")
using MLJBase
using MLJModelInterface
using DataFrames

@testset "MLJ interoperability" begin
    M = Sphere(2)
    p1 = [1.0, 0.0, 0.0]
    p2 = [0.0, 1.0, 0.0]
    p3 = [0.0, sqrt(2) / 2, -sqrt(2) / 2]

    X = DataFrame(pm = [(p1, M), (p2, M), (p3, M)], y = [1, 2, 1])

    tst_model = ManifoldML.TangentSpaceTransformer(
        :mean,
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthogonalBasis(),
        [:pm],
    )
    pm = mean(M, [p1, p2, p3])

    logs = map(
        y -> get_coordinates(M, pm, log(M, pm, y), DefaultOrthogonalBasis()),
        [p1, p2, p3],
    )

    @testset "univariate" begin
        fitted_univariate = fit!(machine(tst_model, X[:pm]))
        transformed_univariate = MLJBase.transform(fitted_univariate, X[:pm])
        for i in 1:2
            @test transformed_univariate[i] == map(y -> y[i], logs)
        end
    end

    @testset "zero argument constructor" begin
        default_tst_model = ManifoldML.TangentSpaceTransformer()
        @test default_tst_model.p === :mean
    end

    fitted_model = fit!(machine(tst_model, X))
    transformed = MLJBase.transform(fitted_model, X)

    @testset "forward transform" begin
        @test schema(transformed).names == (:pm_1, :pm_2, :y)
        @test schema(transformed).types == (Float64, Float64, Int64)
        @test schema(transformed).scitypes == (Continuous, Continuous, Count)

        for i in 1:2
            @test transformed[Symbol("pm_$i")] == map(y -> y[i], logs)
        end
    end

    inv_transformed = MLJBase.inverse_transform(fitted_model, transformed)
    @testset "inverse transform" begin
        @test schema(inv_transformed) == schema(X)
        @test isapprox(map(p -> p[1], X[:pm]), map(p -> p[1], inv_transformed[:pm]))
    end
end
