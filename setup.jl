using Pkg

# Use the environment belonging to this repository.
Pkg.activate(@__DIR__)

# Install and record the required packages.
Pkg.add([
    "Quiqbox",
    "Paulimorphic",
    "ITensors",
    "ITensorMPS",
    "ITensorMPOConstruction",
    "Plots",
])