from graphviz import Source
from pydot import graph_from_dot_file
import collections
import numpy as np
from scipy import sparse
from os.path import join
import pickle


def prepare_data(training_paths, test_paths, dataset_name, output_path):
    # Training and test paths are dictionaries whose keys are the features
    # they must have the same keys
    # The values in the dictionary are lists of paths to dot (.gv) format graph files
    adjacency_matrix = collections.defaultdict()
    training_count = 0  # count for number of training nodes
    testing_count = 0  # ^
    count = 0  # sum of the above
    training_nodes_to_id = {}  # names of training nodes --> index of training nodes
    testing_nodes_to_id = {}  # ^

    class_count = 0  # number of classes
    class_dict = {}  # index of class -> name of class
    class_dict_rev = {}  # name of class -> index of class
    class_nodes_training = {}  # index of class -> list(node indices of that class)
    class_nodes_test = {}  # ^

    for class_name in training_paths.keys():
        # Generate # of class, plus init lists of nodes
        class_dict[class_count] = class_name
        class_dict_rev[class_name] = class_count
        class_nodes_training[class_count] = list()
        class_nodes_test[class_count] = list()
        class_count += 1

    def input_path(path, count, secondary_count, class_id, class_nodes_dict, nodes_to_id):
        # Insert a path (file) into the adjacency matrix and add its nodes to the relevent dict.
        # Increment counters for number of nodes & number of test or training nodes
        dot_graph = graph_from_dot_file(path)
        for node in dot_graph[0].obj_dict['nodes']:
            nodes_to_id[node + path] = secondary_count  # Add this node to the dictionary mapping names to ids
            if count not in class_nodes_dict[class_id]:
                class_nodes_dict[class_id].append(count)  # Keep track of node class in this dict
            adjacency_matrix[count] = list()  # Init the adjacency matrix column for this node to the empty list
            count = count + 1
            secondary_count += 1
        return count, secondary_count  # all node counter, training/test counter

    # Build the adjacency matrix keeping count of how many nodes we have in 'count'
    for class_name in training_paths.keys():
        for path in training_paths[class_name]:
            count, training_count = input_path(path, count, training_count, class_dict_rev[class_name],
                                               class_nodes_training, training_nodes_to_id)
    for class_name in test_paths.keys():
        for path in test_paths[class_name]:
            count, testing_count = input_path(path, count, testing_count, class_dict_rev[class_name], class_nodes_test,
                                              testing_nodes_to_id)

    # Start looping through the class_nodes_training/test dicts
    # add their values to the appropriate index in the label arrays
    # we only used the dictionaries b/c we couldn't initialize the arrays
    # until we knew how big they would be.
    training_labels = np.zeros((training_count, class_count), int)
    testing_labels = np.zeros((testing_count, class_count), int)

    training_node_indices = []
    j = 0
    for class_index in class_nodes_training.keys():
        for training_node in class_nodes_training[class_index]:
            training_labels[j][class_index] = 1
            training_node_indices.append(training_node)
            j += 1

    test_node_indices = []  # Indices of all testing nodes
    j = 0
    for class_index in class_nodes_test.keys():
        for test_node in class_nodes_test[class_index]:
            testing_labels[j][class_index] = 1
            test_node_indices.append(test_node)
            j += 1
    # End Loop

    # Testing / Training feature vectors, start as zero
    training_feature_vec = np.zeros((training_count, 2), int)
    test_feature_vec = np.zeros((testing_count, 2), int)

    def add_to_feature_vec(paths, feat_vec, nodes_to_id):
        # Take a path (dot file - a graph) and loop through its edges adding them to the feature vec
        # require the dictionary converting names+path -> index; it's different for test/training
        for path in paths:
            dot_graph = graph_from_dot_file(path)
            for edge in dot_graph[0].obj_dict['edges']:
                from_node = nodes_to_id[edge[0] + path]
                to_node = nodes_to_id[edge[1] + path]
                # Increment feature for in/out degree
                feat_vec[from_node][1] += 1
                feat_vec[to_node][0] += 1
                # Add to adjacency matrix
                if to_node not in adjacency_matrix[from_node]:
                    adjacency_matrix[from_node].append(to_node)

    for class_name in training_paths.keys():
        add_to_feature_vec(training_paths[class_name], training_feature_vec, training_nodes_to_id)
        add_to_feature_vec(test_paths[class_name], test_feature_vec, testing_nodes_to_id)

    training_feature_vec = sparse.csr_matrix(training_feature_vec)  # cast to correct type for gcn library
    test_feature_vec = sparse.csr_matrix(test_feature_vec)  # cast to correct type for gcn library

    # Dump in pickle format
    pickle.dump(adjacency_matrix, open(join(output_path, "ind.{}.graph".format(dataset_name)), 'wb'))
    pickle.dump(training_feature_vec, open(join(output_path, "ind.{}.x".format(dataset_name)), 'wb'))
    pickle.dump(training_feature_vec, open(join(output_path, "ind.{}.allx".format(dataset_name)), 'wb'))
    pickle.dump(test_feature_vec, open(join(output_path, "ind.{}.tx".format(dataset_name)), 'wb'))
    pickle.dump(training_labels, open(join(output_path, "ind.{}.y".format(dataset_name)), 'wb'))
    pickle.dump(training_labels, open(join(output_path, "ind.{}.ally".format(dataset_name)), 'wb'))
    pickle.dump(testing_labels, open(join(output_path, "ind.{}.ty".format(dataset_name)), 'wb'))
    # Dump as text the raw test indices (index in the adjacency matrix not the index in the feature matrix)
    with open(join(output_path, "ind.{}.test.index".format(dataset_name)), 'w') as test_index_file:
        for node in training_node_indices:
            test_index_file.write(str(node) + "\r\n")

    return 0


training_paths = {'filezilla': [join("our_data", "FileZilla10MB_joker.XML.dot"),
                                join("our_data", "FileZilla10MB_madhatter.XML.dot"),
                                join("our_data", "FileZilla20MB_joker.XML.dot"),
                                join("our_data", "FileZilla20MB_madhatter.XML.dot"),
                                join("our_data", "FileZilla50MB_joker.XML.dot")],
                  'winscp': [join("our_data", "WinSCP10MB_madhatter.XML.dot"),
                             join("our_data", "WinSCP20MB_joker.XML.dot"),
                             join("our_data", "WinSCP50MB_madhatter.XML.dot"),
                             join("our_data", "WinSCP100MB_joker.XML.dot")]}
test_paths = {'filezilla': [join("our_data", "FileZilla50MB_madhatter.XML.dot"),
                            join("our_data", "FileZilla100MB_joker.XML.dot"),
                            join("our_data", "FileZilla100MB_madhatter.XML.dot")],
              'winscp': [join("our_data", "WinSCP10MB_joker.XML.dot"),
                         join("our_data", "WinSCP20MB_madhatter.XML.dot"),
                         join("our_data", "WinSCP50MB_joker.XML.dot"),
                         join("our_data", "WinSCP100MB_madhatter.XML.dot")]}

"our_data/"
prepare_data(training_paths, test_paths, "n8ta", "data")
