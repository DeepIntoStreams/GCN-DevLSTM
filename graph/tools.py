import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_dual_graph_adj_matrix(num_nodes):
    ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                    (19, 18), (20, 19), (21, 21), (22, 23),  (23, 8), (24, 25),(25, 12)
                    )
    adj_matrix = np.zeros((num_nodes,num_nodes))
    for i in range(len(ntu_pairs)):
        cur_joint = ntu_pairs[i]
        for j in range(i+1,len(ntu_pairs)):
            if cur_joint[0] in ntu_pairs[j] or cur_joint[1] in ntu_pairs[j]:
                adj_matrix[i,j] = 1
                adj_matrix[j,i] = 1
    return adj_matrix

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A