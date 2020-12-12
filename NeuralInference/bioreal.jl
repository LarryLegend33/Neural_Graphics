using PyCall
using AbstractPlotting 
using Makie
using AbstractPlotting
using MakieLayout
using Gen
using LinearAlgebra
using Random
using Images
using ShiftedArrays
using Statistics
using StatsBase


b2 = pyimport("brian2")
# note this will be different than python tutorials. multi line string is """ in juia, ''' in python. 
eqs = """
dv/dt = (1-v)/tau : 1
"""
G = b2.NeuronGroup(1, eqs)
