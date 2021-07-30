A computational model of metaplastic synapses for reinforcement learning derived from
the paper by Farashahi, S. et al. (2017). Used networkx to implement each population of
neurons encoding the alternatives in the forced choice task as a directed graph.
Vertices represent the various metastates and encode how many synapses are in each one.
Edges represent the possible transitions between states following potentiation or depression
events, with edge weights encoding the probability of each transition.
