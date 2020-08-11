# ManifoldML.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/ManifoldML.jl/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/ManifoldML.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/ManifoldML.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov.io](http://codecov.io/github/JuliaManifolds/ManifoldML.jl/coverage.svg?branch=master)](https://codecov.io/gh/JuliaManifolds/ManifoldML.jl)

The Package `ManifoldML.jl` provides methods to do Machine Learning methods on Riemannian
manifolds. It is based on [`ManifoldsBase`](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html)
in order to work with all manifolds from [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/index.html).

Related packages are
* [`Manopt.jl`](https://manoptjl.org) for general optimization on manifolds – this might be employed for certain optimization tasks
* [`PosDefManifoldML.jl`](https://marco-congedo.github.io/PosDefManifoldML.jl/dev/) for the specific case on symmetric positive definite matrices.