import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

CUTOFF = 0.2


def condense_and_plot(adj, class_labels, class_names):
    # generate_graph(adj)
    num_classes = len(class_labels[0])
    nodes_in_class = [list() for x in range(num_classes+1)]  # class_index => list of node index

    G = nx.DiGraph()
    add_edges_to_graph(adj, G)

    # Find node we are certain of the class of
    for node_index in range(0, len(adj)):
        max_value = max(class_labels[node_index])
        if max_value > CUTOFF:
            class_index = class_labels[node_index].index(max_value)
            nodes_in_class[num_classes].append(node_index)

    # Remove these nodes from the graph and add their edges to the super node
    for indx,nodes in enumerate(nodes_in_class):
        print(indx)
        print(class_names)
        super_node_name = class_names[indx]
        G.add_node(super_node_name)
        for node_index in nodes:
            for out_edge in G.edges_iter(node_index):
                G.add_edge((super_node_name, node_index))
                G.remove_edge(out_edge)

def generate_graph(adjacency_matrix, labels={}):
    G = nx.DiGraph()
    add_edges_to_graph(adjacency_matrix, G)
    nx.draw(G, cmap=plt.get_cmap('jet'))
    plt.show()


def add_edges_to_graph(adjacency_matrix, G):
    edges = []
    for i, row in enumerate(adjacency_matrix):
        for j, value in enumerate(row):
            if adjacency_matrix[i][j] == 1:
                edges.append((i, j))
            elif adjacency_matrix[i][j] != 0:
                raise Exception("YO! adjacency matrix should only have 0s and 1s")
    G.add_edges_from(edges)
    return G

# generate_graph([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1]])
