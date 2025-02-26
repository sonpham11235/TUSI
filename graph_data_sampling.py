import numpy as np
import networkx as nx
import random

print('Constructing Bernoulli Graph from adjacency matrix ')
ber_graph = nx.read_adjlist('./data/graph/bernoulli.adjlist', create_using=nx.DiGraph)
print(f' Number of nodes: {ber_graph.number_of_nodes()}')
print(f' Number of edges: {ber_graph.number_of_edges()}')
print('Graph constructed')


print('Constructing Hierarchical Graph from adjacency matrix ')
hier_graph = nx.read_adjlist('./data/graph/hierarchical.adjlist', create_using=nx.DiGraph)
print(f' Number of nodes: {hier_graph.number_of_nodes()}')
print(f' Number of edges: {hier_graph.number_of_edges()}')
print('Graph constructed')

# print('Sampling valid simple paths of Bernoulli graph')
# positive_set = []
# negative_set = []
# for i in range(195):
#     source = 'X'+str(i)
#     for j in range(i+2,200):
#         target = 'X'+str(j)
#         paths = list(nx.all_simple_paths(ber_graph, source, target, 6))
#         if len(paths) > 0:
#             print(f'Positive pair [{source},{target}]')
#             for path in paths:
#                 sample = target + ' ' + ' '.join(path) + ' p1'
#                 positive_set.append(sample)
#         else:
#             print(f'Negative pair [{source},{target}]')
#             negative_set.append(f'{target} {source} p0')

# print(f'Total positive pair: {len(positive_set)}')
# print(f'Total negative pair: {len(negative_set)}')

# with open('./data/graph/bernoulli_positive.txt', 'w') as file:
#     for sample in positive_set:
#         file.write(f'{sample}\n')
        
# with open('./data/graph/bernoulli_negative.txt', 'w') as file:
#     for sample in negative_set:
#         file.write(f'{sample}\n')

print('Sampling valid simple paths of Bernoulli graph')
positive_set = []
negative_set = []
for i in range(195):
    source = 'X'+str(i)
    for j in range(i+2,200):
        target = 'X'+str(j)
        paths = list(nx.all_simple_paths(hier_graph, source, target))
        if len(paths) > 0:
            print(f'Positive pair [{source},{target}]')
            for path in paths:
                sample = target + ' ' + ' '.join(path) + ' p1'
                positive_set.append(sample)
        else:
            print(f'Negative pair [{source},{target}]')
            negative_set.append(f'{target} {source} p0')

print(f'Total positive pair: {len(positive_set)}')
print(f'Total negative pair: {len(negative_set)}')

with open('./data/graph/hierarchical_positive.txt', 'w') as file:
    for sample in positive_set:
        file.write(f'{sample}\n')
        
with open('./data/graph/hierarchical_negative.txt', 'w') as file:
    for sample in negative_set:
        file.write(f'{sample}\n')