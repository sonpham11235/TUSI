import numpy as np
import networkx as nx
import random

class BernoulliBuilder:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.node_names = ['X' + str(i) for i in range(n)]
        self.digraph = nx.DiGraph()
    
    def CreateUpperTriangularMask(self):
        mask = np.random.choice(a=[0,1], size=(self.n,self.n), p=[1-self.p, self.p])
        return np.triu(mask,1)

    def CreateSparseGraph(self):
        digraph = nx.DiGraph()
        digraph.add_nodes_from(self.node_names)
        return digraph

    def GenerateBernoulliGraph(self):
        self.digraph = self.CreateSparseGraph()
        while not nx.is_weakly_connected(self.digraph):
            self.digraph = self.CreateSparseGraph()
            mask = self.CreateUpperTriangularMask()
            self.digraph.add_edges_from((('X' + str(int(e[0])), 'X' + str(int(e[1]))) for e in zip(*mask.nonzero())))
        return self.digraph
    
class HierarchicalBuilder:
    def __init__(self, num_nodes_per_layer, num_layers, p):
        self.num_nodes_per_layer = num_nodes_per_layer
        self.num_layers = num_layers
        self.num_nodes = num_nodes_per_layer * num_layers
        self.node_names = ['X' + str(i) for i in range(self.num_nodes)]
        self.digraph = nx.DiGraph()
        self.p = p

    def CreateSparseGraph(self):
        digraph = nx.DiGraph()
        digraph.add_nodes_from(self.node_names)
        return digraph

    def GenerateHierarchicalDAG(self):
        self.digraph = self.CreateSparseGraph()
        while nx.number_of_isolates(self.digraph) > 0:
            self.digraph = self.CreateSparseGraph()
            for i in range(self.num_layers-1):
                for j in range(self.num_nodes_per_layer):
                    for k in range(self.num_nodes_per_layer):
                        if random.uniform(0.0, 1.0) <= self.p:
                            num1 = (i * self.num_nodes_per_layer) + j
                            num2 = ((i+1) * self.num_nodes_per_layer) + k
                            self.digraph.add_edge('X'+str(num1), 'X'+str(num2))
        return self.digraph

bernoulli_builder = BernoulliBuilder(200, 0.05)
print('Bernoulli builder prepared, 200 nodes with edge probability of 0.05')
hierarchical_builder = HierarchicalBuilder(20, 10, 0.1)
print('Hierarchical builder prepared, 10 layers with 20 nodes per layer with edge probability of 0.1')

print('-- Building bernoulli graph')
ber_graph = bernoulli_builder.GenerateBernoulliGraph()
print('-- Done building bernoulli graph')

print('-- Building Hierarchical graph')
hier_graph = hierarchical_builder.GenerateHierarchicalDAG()
print('-- Done building hierarchical graph')

print('-- Saving')
nx.write_adjlist(ber_graph, './data/graph/bernoulli_adj')
print('-- Saving')
nx.write_adjlist(hier_graph, './data/graph/hierarchical_adj')