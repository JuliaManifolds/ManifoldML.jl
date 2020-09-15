# Tangent space models

One of the more popular methods of statistical modeling on manifolds is selecting a point $p$ on the manifold $\mathcal M$ (for example Riemannian center of mass of data points), transforming all point to the tangent space $T_p\mathcal M$ at that point, and using traditional linear models in that tangent space.
Tangent space Principal Component Analysis is a popular example of this kind of modeling.
The example below demonstrates how to build and use such model.

```@example
using Manifolds, MultivariateStats
M = Sphere(2)
pts = [project(M, randn(3)) for _ in 1:100]
m = mean(M, pts)
logs = log.(Ref(M), Ref(m), pts)
basis = DefaultOrthonormalBasis()
coords = map(X -> get_coordinates(M, m, X, basis), logs)
coords_red = reduce(hcat, coords)
z = zeros(manifold_dimension(M)))
model = fit(PCA, coords_red; maxoutdim=1, mean=z)
X = get_vector(M, m, reconstruct(model, [1.0]), basis)
geodesic(M, m, X, range(-1.0, 1.0, length=101))
```

The code performs the following steps:

  1. Creating random data `pts`.
  2. Computing Riemannian center of mass `m` of `pts`.
  3. Transforming data to the tangent space at `m` using the logarithmic map.
  4. Selecting a `basis` of the tangent space at `m` in which the points will be represented.
  5. Computing coordinates of tangent vectors in said basis.
  6. Transforming a vector of vectors of coordinates into a matrix `coords_red`.
  7. Fitting an ordinary PCA model to the matrix `coords_red`. Zero mean chosen, and the first principal direction is to be computed (the `maxoutdim` argument).
  8. The first principal component is converted into a tangent vector `X`.
  9. A geodesic on `M` along that direction is computed.

The same general procedure can be applied to other tangent space models by replacing steps 7 and 8.
