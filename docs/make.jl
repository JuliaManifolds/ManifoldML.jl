using ManifoldML, Documenter

makedocs(
    format = Documenter.HTML(prettyurls = false, assets = ["assets/favicon.ico"]),
    modules = [ManifoldML],
    authors = "Ronny Bergmann.",
    sitename = "ManifoldML.jl",
    pages = [
        "Home" => "index.md",
        "K means" => "kmeans.md",
        "K Nearest Neighbors" => "knn.md",
        "Tangent space models" => "tangent-models.md",
    ],
)

deploydocs(repo = "github.com/JuliaManifolds/ManifoldML.jl.git", push_preview = true)
