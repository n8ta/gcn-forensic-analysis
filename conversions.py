from graphviz import Source
from pydot import graph_from_dot_file
import collections
import numpy as np
from scipy import sparse


def create_adjaceny_matrix(paths):
    adjacency_matrix = collections.defaultdict()
    count = 0
    nodes_to_id = {}
    for path in paths:
        dot_graph = graph_from_dot_file(path)
        for node in dot_graph[0].obj_dict['nodes']:
            nodes_to_id[node] = count
            adjacency_matrix[count] = list()
            count = count + 1

    num_nodes = len(adjacency_matrix)
    feature_matrix = np.zeros((2,num_nodes),int)
    for path in paths:
        dot_graph = graph_from_dot_file(path)
        for edge in dot_graph[0].obj_dict['edges']:
            from_node = nodes_to_id[edge[0]]
            to_node = nodes_to_id[edge[1]]

            feature_matrix[1][from_node] += 1
            feature_matrix[0][to_node] += 1


            if to_node not in adjacency_matrix[from_node]:
                adjacency_matrix[from_node].append(to_node)

    feature_matrix = sparse.csr_matrix(feature_matrix)

x = create_adjaceny_matrix(['our_data/output_.gv'])
