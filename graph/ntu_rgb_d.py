import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        #print(args,kwargs)
        if (kwargs['labeling_mode'] == 'dual_graph'):
            self.A_binary = tools.get_dual_graph_adj_matrix(self.num_nodes)
            print('use dual graph')
        elif (kwargs['labeling_mode'] == 'spatial'):
            self.A_binary = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
            print('use normal graph')
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        # self.A = tools.normalize_adjacency_matrix(self.A_binary)


