import numpy as np
import networkx as nx
import random

print('Constructing Bernoulli Graph from adjacency matrix ')
ber_graph = nx.read_adjlist('./data/graph/bernoulli/bernoulli.adjlist', create_using=nx.DiGraph)
print(f' Number of nodes: {ber_graph.number_of_nodes()}')
print(f' Number of edges: {ber_graph.number_of_edges()}')
print('Graph constructed')


print('Constructing Hierarchical Graph from adjacency matrix ')
hier_graph = nx.read_adjlist('./data/graph/hierarchical/hierarchical.adjlist', create_using=nx.DiGraph)
print(f' Number of nodes: {hier_graph.number_of_nodes()}')
print(f' Number of edges: {hier_graph.number_of_edges()}')
print('Graph constructed')

print('Sampling valid simple paths of Bernoulli graph')
ber_edge = []
ber_negative_set = []
ber_positive_set = []
hier_edge = []
hier_positive_set = []
hier_negative_set = []

for i in range(1,200):
    source = 'X'+str(i)
    for j in range(i,i+1):
        target = 'X'+str(j)
        paths = list(nx.all_simple_paths(ber_graph, source, target, 6))
        if len(paths) > 0:
            print(f'Positive pair [{source},{target}]')
            for path in paths:
                sample = target + ' ' + ' '.join(path) + ' p1'
                ber_edge.append(sample)
        else:
            print(f'Negative pair [{source},{target}]')
            ber_negative_set.append(f'{target} {source} p0')

for i in range(1,199):
    source = 'X'+str(i)
    for j in range(i+2,201):
        target = 'X'+str(j)
        paths = list(nx.all_simple_paths(ber_graph, source, target, 6))
        if len(paths) > 0:
            print(f'Positive pair [{source},{target}]')
            for path in paths:
                sample = target + ' ' + ' '.join(path) + ' p1'
                ber_positive_set.append(sample)
        else:
            print(f'Negative pair [{source},{target}]')
            ber_negative_set.append(f'{target} {source} p0')

print('Sampling valid simple paths of Bernoulli graph')
layer = 0
for i in range(9):
    for j in range(1,21):
        source = 'X'+str(i*20+j)
        for k in range(1,21):
            target = 'X'+str((i+1)*20+k)
            paths = list(nx.all_simple_paths(hier_graph, source, target))
            if len(paths) > 0:
                print(f'Positive pair [{source},{target}]')
                for path in paths:
                    sample = target + ' ' + ' '.join(path) + ' p1'
                    hier_edge.append(sample)
            else:
                print(f'Negative pair [{source},{target}]')
                hier_negative_set.append(f'{target} {source} p0')

for i in range(7):
    for j in range(1,21):
        source = 'X'+str(i*20+j)
        for t in range(i+2,9):
            for k in range(1,21):
                target = 'X'+str(t*20+k)
                paths = list(nx.all_simple_paths(hier_graph, source, target))
                if len(paths) > 0:
                    print(f'Positive pair [{source},{target}]')
                    for path in paths:
                        sample = target + ' ' + ' '.join(path) + ' p1'
                        hier_positive_set.append(sample)
                else:
                    print(f'Negative pair [{source},{target}]')
                    hier_negative_set.append(f'{target} {source} p0')

with open('./data/graph/bernoulli/edges.txt', 'w') as file:
    for sample in ber_edge:
        file.write(f'{sample}\n')

with open('./data/graph/bernoulli/positive.txt', 'w') as file:
    for sample in ber_positive_set:
        file.write(f'{sample}\n')
        
with open('./data/graph/bernoulli/negative.txt', 'w') as file:
    for sample in ber_negative_set:
        file.write(f'{sample}\n')

with open('./data/graph/hierarchical/edges.txt', 'w') as file:
    for sample in hier_edge:
        file.write(f'{sample}\n')

with open('./data/graph/hierarchical/positive.txt', 'w') as file:
    for sample in hier_positive_set:
        file.write(f'{sample}\n')
        
with open('./data/graph/hierarchical/negative.txt', 'w') as file:
    for sample in hier_negative_set:
        file.write(f'{sample}\n')

print(f'Total Bernoulli edges: {len(ber_edge)}')
print(f'Total Bernoulli positive pairs: {len(ber_positive_set)}')
print(f'Total Bernoulli negative pairs: {len(ber_negative_set)}')

print(f'Total Hierarchical edges: {len(hier_edge)}')
print(f'Total Hierarchical positive pairs: {len(hier_positive_set)}')
print(f'Total Hierarchical negative pairs: {len(hier_negative_set)}')