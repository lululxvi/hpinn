module PINN

using Gmsh
using Gridap
#using GridapGmsh
using SparseArrays
#using Gridap.Geometry
using ChainRules
using Zygote
using LinearAlgebra
import ChainRules: rrule
import Gmsh: gmsh

export MeshGenerator
export MatrixA
export MatrixB
export pf_p
export u_pf
export g_u
export g_p
export Threshold
export p_vec
export extract_design
export Filter
export a_f

include("FilterAndThreshold.jl")
include("Helper.jl")
include("MeshGenerator.jl")
include("PML.jl")
include("Objective.jl")
include("Model.jl")

end