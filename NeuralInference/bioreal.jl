using GLMakie
using Gen
using LinearAlgebra
using Random
using Statistics
using StatsBase
using SpikingCircuits
using SpikingCircuits.SpiketrainViz


# b2 = pyimport("brian2")
# # note this will be different than python tutorials. multi line string is """ in juia, ''' in python. 
# eqs = """
# dv/dt = (1-v)/tau : 1
# """
# G = b2.NeuronGroup(1, eqs)



events = SpikingSimulator.simulate_for_time_and_get_events(
    OnOffPoissonNeuron(1.0),
    100.0;
    initial_inputs=(:on,)
)

