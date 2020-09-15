# K Nearest Neighbors

K Nearest Neighbors (kNN) is a popular classification algorithm and a member of the family of instance-based learning methods.

```@example
using Manifolds, ManifoldML, NearestNeighbors, StatsBase, Statistics
M = SymmetricPositiveDefinite(3)
N = 100
pts = [cov(randn(10, 3)) for _ in 1:N]
ys = [rand([1, 2]) for _ in 1:N]
dist = ManifoldML.RiemannianDistance(M)
point_matrix = reduce(hcat, map(a -> reshape(a, 9), pts))
balltree = BallTree(point_matrix, dist)
function classify(tree, p, ys, k)
    num_coords = prod(representation_size(balltree.metric.manifold))
    idxs, dists = knn(tree, reshape(p, num_coords), k)
    return mode(ys[idxs])
end
classify(balltree, pts[2], ys, 3)
```

The code performs the following steps:

  1. Creating random data `pts` on the manifold of symmetric positive definite matrices (i.e. covariance matrices).
  2. Creating random class labels `ys`.
  3. Selecting the distance `dist` to use in kNN.
  4. Collecting coordinates of points in a matrix `point_matrix`.
  5. Creating nearest neighbor search tree using the `NearestNeighbors` package.
  6. Writing a simple kNN classifier that reshapes the point `p`, performs a kNN search for `k` nearest neighbors and returns the most common label using the function `mode` from `StatsBase`.

The same general procedure can be followed to build other distance-based classifiers.
