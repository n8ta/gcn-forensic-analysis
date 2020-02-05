from graphviz import Source
from pydot import graph_from_dot_file
import collections
import numpy as np
from scipy import sparse


def prepare_data(training_paths, test_paths):
    # Training and test paths are dictionaries whose keys are the features
    # they must have the same keys
    # The values in the dictionary are lists of paths to dot (.gv) format graph files
    adjacency_matrix = collections.defaultdict()
    training_count = 0
    testing_count = 0
    count = 0
    nodes_to_id = {}

    class_count = 0  # number of classes
    class_dict = {}  # index -> name
    class_dict_rev = {}  # name -> index
    class_nodes_training = {}  # index -> list(node indices)
    class_nodes_test = {}  # ^

    for class_name in training_paths.keys():
        class_dict[class_count] = class_name
        class_dict_rev[class_name] = class_count
        class_nodes_training[class_count] = list()
        class_nodes_test[class_count] = list()
        class_count += 1

    def input_path(path, count, secondary_count, class_id, class_nodes_dict):
        dot_graph = graph_from_dot_file(path)
        for node in dot_graph[0].obj_dict['nodes']:
            nodes_to_id[node + path] = count  # Add this node to the dictionary mapping names to ids
            if count not in class_nodes_dict[class_id]:
                class_nodes_dict[class_id].append(
                    count)  # Keep track of what nodes are in what class in the class_nodes_dict
            adjacency_matrix[count] = list()  # Init the adjacency matrix column for this node to the empty list
            count = count + 1
            secondary_count += 1
        return count, secondary_count

    # Build the adjacency matrix keeping count of how many nodes we have in 'count'
    for class_name in training_paths.keys():
        for path in training_paths[class_name]:
            count, training_count = input_path(path, count, training_count, class_dict_rev[class_name], class_nodes_training)
    for class_name in test_paths.keys():
        for path in test_paths[class_name]:
            count, testing_count = input_path(path, count, testing_count, class_dict_rev[class_name], class_nodes_test)

    # Start looping through the class_nodes_training/test dicts
    # add their values to the appropriate index in the label arrays
    # we only used the dictionaries b/c we couldn't initialize the arrays
    # until we knew how big they would be.
    training_labels = np.zeros((count, training_count), int)
    testing_labels = np.zeros((count, testing_count), int)

    for class_index in class_nodes_training.keys():
        for x in range(training_count):
            training_labels[x][class_index] = 1

    for class_index in class_nodes_test.keys():
        for x in range(testing_count):
            testing_labels[x][class_index] = 1
    # End Loop

    # Testing / Training feature vectors, start as zero
    training_feature_vec = np.zeros((2, count), int)
    test_feature_vec = np.zeros((2, count), int)

    def add_to_feature_vec(paths, feat_vec):
        for path in paths:
            dot_graph = graph_from_dot_file(path)
            for edge in dot_graph[0].obj_dict['edges']:
                from_node = nodes_to_id[edge[0] + path]
                to_node = nodes_to_id[edge[1] + path]

                feat_vec[1][from_node] += 1
                feat_vec[0][to_node] += 1

                if to_node not in adjacency_matrix[from_node]:
                    adjacency_matrix[from_node].append(to_node)

    for class_name in training_paths.keys():
        add_to_feature_vec(training_paths[class_name], training_feature_vec)
        add_to_feature_vec(test_paths[class_name], test_feature_vec)

    training_feature_vec = sparse.csr_matrix(training_feature_vec)  # cast to correct type for gcn library
    test_feature_vec = sparse.csr_matrix(test_feature_vec)  # cast to correct type for gcn library

    return adjacency_matrix, training_feature_vec, test_feature_vec, training_labels, testing_labels, class_dict


training_paths = {'filezilla': ["our_data/FileZilla10MB_joker.XML.dot",
                                "our_data/FileZilla10MB_madhatter.XML.dot",
                                "our_data/FileZilla20MB_joker.XML.dot",
                                "our_data/FileZilla20MB_madhatter.XML.dot",
                                "our_data/FileZilla50MB_joker.XML.dot"],
                  'winscp': ["our_data/WinSCP10MB_madhatter.XML.dot",
                             "our_data/WinSCP20MB_joker.XML.dot",
                             "our_data/WinSCP50MB_madhatter.XML.dot",
                             "our_data/WinSCP100MB_joker.XML.dot"]}
test_paths = {'filezilla': ["our_data/FileZilla50MB_madhatter.XML.dot",
                            "our_data/FileZilla100MB_joker.XML.dot",
                            "our_data/FileZilla100MB_madhatter.XML.dot"],
              'winscp': ["our_data/WinSCP10MB_joker.XML.dot",
                         "our_data/WinSCP20MB_madhatter.XML.dot",
                         "our_data/WinSCP50MB_joker.XML.dot",
                         "our_data/WinSCP100MB_madhatter.XML.dot"]}

prepare_data(training_paths, test_paths)
