import Pkg

# List of required external packages
dependencies = [
    "CSV",
    "DataFrames",
    "MLJ",
    "Plots",
    "CairoMakie",
    "BenchmarkTools",
    "StatsPlots",
    "StatsBase",
    "Clustering",
    "NearestNeighbors",
    "Flux",
    "Random",
    "Distributions",
    "LinearAlgebra"
]

# Install all packages
println("Installing dependencies...")
Pkg.add(dependencies)
println("All dependencies have been successfully installed!")
