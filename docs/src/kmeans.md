# K-Means

The [k means](https://en.wikipedia.org/wiki/K-means_clustering) using [Lloyd's algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) can be generalized to manifolds, since its first step
looks for the closest center of each data point, which can be done in terms of the geodesic distance.
The second step of computing the mean within each cluster is generalized to computing the Riemannian center of mass[^Karcher1977].

```@autodocs
Modules = [ManifoldML]
Pages = ["kmeans.jl"]
Order = [:type, :function]
```

## Literature

[^Karcher1977]:
    > Karcher, H.: Riemannian center of mass and mollifier smoothing,
    > Communications on Pure and Applied Mathematics 30(5), 1977, pp. 509–541.
    > doi: [10.1002/cpa.3160300502](https://doi.org/10.1002/cpa.3160300502)