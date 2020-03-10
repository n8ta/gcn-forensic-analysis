import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Take a graph, draw it, merge nodes whose labels we are more than $CUTOFOF$ certain of and


def condense_and_plot(trace_name, adj, class_labels, class_names, cutoffs, class_names_to_ids):
    false_positive_rate = {}
    true_positive_rate = {}
    # generate_graph(adj)
    for cutoff in cutoffs:
        print("calculating for cutoff: {}".format(cutoff))
        num_classes = len(class_labels[0])
        nodes_in_class = [list() for x in range(num_classes)]  # class_index => list of node index for class_name in class_names:
        G = nx.DiGraph()
        add_edges_to_graph(adj, G)
        nx.write_gexf(G, "graphs_out/uncompress-{}-nodes-{}.gexf".format(trace_name,len(G.nodes)))
        # Find node we are certain of the class of
        for node_index in range(0, len(adj)):
            max_value = max(class_labels[node_index])
            if max_value > cutoff:
                class_index = class_labels[node_index].index(max_value)
                nodes_in_class[class_index].append(node_index)

        # Remove these nodes from the graph and add their edges to the super node
        for indx, nodes in enumerate(nodes_in_class):
            if len(nodes) < 5:
                continue
            adsorbed_count = 0
            super_node_name = class_names[indx]
            G.add_node(super_node_name)
            for node_index in nodes:
                for successor in G.successors(node_index):
                    G.add_edges_from([(super_node_name, successor)])
                for predecessor in G.predecessors(node_index):
                    G.add_edges_from([(predecessor, super_node_name)])
                adsorbed_count += 1
                G.remove_node(node_index)
            G.add_weighted_edges_from([(super_node_name, super_node_name, adsorbed_count)])
        nx.draw(G, cmap=plt.get_cmap('jet'))
        nx.write_gexf(G, "graphs_out/compressed-{}-nodes-{}-{}.gexf"
                      .format(trace_name,len(G.nodes), str(cutoff)))
        print("out!")
        plt.show()

        correct_nodes = len(nodes_in_class[class_names_to_ids[trace_name]])
        identified_nodes = sum([len(x) for x in nodes_in_class])
        total_nodes = len(adj)
        false_positive_rate[cutoff] = (identified_nodes-correct_nodes)/total_nodes
        true_positive_rate[cutoff] = correct_nodes/total_nodes
    print("FPR:")
    for key,value in false_positive_rate.items():
        print(value)
    print("TPR:")
    for key, value in true_positive_rate.items():
        print(value)


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