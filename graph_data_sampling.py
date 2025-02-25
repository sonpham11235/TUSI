import numpy as np
import networkx as nx
import random

print('Constructing Bernoulli Graph from adjacency matrix ')
ber_graph = nx.read_adjlist('./data/graph/bernoulli.adjlist')
print(f' Number of nodes: {ber_graph.number_of_nodes()}')
print(f' Number of edges: {ber_graph.number_of_edges()}')
print('Graph constructed')


print('Constructing Hierarchical Graph from adjacency matrix ')
hier_graph = nx.read_adjlist('./data/graph/hierarchical.adjlist')
print(f' Number of nodes: {hier_graph.number_of_nodes()}')
print(f' Number of edges: {hier_graph.number_of_edges()}')
print('Graph constructed')

print('Sampling valid simple paths of Bernoulli graph')
positive_set = []
negative_set = []
for i in range(195):
        for j in range(i+1,200):
            source = 'X'+str(i)
            target = 'X'+str(j)
            if nx.has_path(ber_graph, source, target):
                paths = nx.all_simple_paths(ber_graph, source, target, 5)
                for path in paths:
                    if len(path) > 2:
                        sample = target + ' '.join(path) + 'p1'
                        positive_set.append(sample)
            else:
                negative_set.append(f'{target} {source} p0')
with open('./data/graph/bernoulli_positive.txt', 'w') as file:
    for sample in positive_set:
        file.write(sample)
        
with open('./data/graph/bernoulli_negative.txt', 'w') as file:
    for sample in negative_set:
        file.write(sample)