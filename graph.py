import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

CUTOFF = 0.9

def condense_and_plot(adj, class_labels, class_names):
    generate_graph(adj)
    num_classes = len(class_labels[0])
    nodes_in_class = [list() for x in range(num_classes)]  # class_index => list of node index
    labels = {}
    for class_name in class_names:
        labels[class_name] = class_name

    G = nx.DiGraph()
    add_edges_to_graph(adj, G)
    nx.write_gexf(G, "uncompress-winrar.gexf")
    # Find node we are certain of the class of
    for node_index in range(0, len(adj)):
        max_value = max(class_labels[node_index])
        if max_value > CUTOFF:
            class_index = class_labels[node_index].index(max_value)
            nodes_in_class[class_index].append(node_index)

    # Remove these nodes from the graph and add their edges to the super node
    for indx, nodes in enumerate(nodes_in_class):
        super_node_name = class_names[indx]
        G.add_node(super_node_name)
        for node_index in nodes:
            for successor in G.successors(node_index):
                G.add_edges_from([(super_node_name, successor)])
            for predecessor in G.predecessors(node_index):
                G.add_edges_from([(predecessor, super_node_name)])
            G.remove_node(node_index)
    print(len(G.nodes))
    # nx.draw_networkx_labels(G, {'winrar': (.5, 0)}, {'winrar': 'winrar'})
    nx.draw(G, cmap=plt.get_cmap('jet'))
    nx.write_gexf(G, "winrar.gexf")
    plt.show()


def generate_graph(adjacency_matrix, labels={}):
    G = nx.DiGraph()
    add_edges_to_graph(adjacency_matrix, G)
    print(len(G.nodes))
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


training_classes = ['utorrent', 'winrar', 'skype_file',
                    'winscp', 'spotify_online', 'filezilla',
                    'spotify_offline', 'skype_video']


# generate_graph([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1]])
