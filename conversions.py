from graphviz import Source
from pydot import graph_from_dot_file
import collections
import numpy as np
from scipy import sparse
from os.path import join
import pickle
import json


class Node:
    count = 0
    sub_counts = {'training': 0, 'testing': 0}

    def __init__(self, str, class_id, type, path):
        obj = json.loads(str)
        self.fileName = obj['filePath']
        self.eventName = obj['eventName']
        self.callStack = obj['callstack']
        self.in_degree = 0
        self.out_degree = 0
        self.id = Node.count
        self.children = set()
        self.parents = set()
        self.path = path
        self.class_id = class_id
        if type not in ['training', 'testing']:
            raise Exception('Bad node type')
        self.type = type
        self.sub_id = Node.sub_counts[self.type]
        Node.count += 1
        Node.sub_counts[self.type] += 1


def prepare_data(training_paths, testing_paths, dataset_name, output_path):
    nodes = {}  # name -> node
    class_count = 0
    class_dict = {}
    adjacency_matrix = {}
    for class_name in training_paths.keys():
        class_dict[class_name] = class_count
        class_count = class_count + 1

    # Generate nodes in the nodes{} dict for each node on each line in each file in each path
    # Don't generated duplicates
    def prepare(type, class_id, paths):
        for path in paths:
            with open(path, 'r') as file:
                last_node = None
                for line in file:
                    current_node = None
                    filePath = json.loads(line)['filePath']
                    eventName = json.loads(line)['eventName']
                    if filePath + path + eventName in nodes.keys():
                        current_node = nodes[filePath + path + eventName]
                    else:
                        current_node = Node(line, class_id, type, path)
                        nodes[filePath + path + eventName] = current_node
                    if last_node:
                        last_node.out_degree += + 1
                        current_node.in_degree += + 1
                        current_node.parents.add(last_node)
                        last_node.children.add(current_node)
                    last_node = current_node

    for class_name in training_paths.keys():
        prepare("training", class_dict[class_name], training_paths[class_name])
    for class_name in testing_paths.keys():
        prepare("testing", class_dict[class_name], testing_paths[class_name])

    training_count = len(list(filter(lambda node: node.type == "training", nodes.values())))
    testing_count = len(list(filter(lambda node: node.type == "testing", nodes.values())))

    training_feature_vec = np.zeros((training_count, 4), float)
    testing_feature_vec = np.zeros((testing_count, 4), float)

    training_labels = np.zeros((training_count, class_count), int)
    testing_labels = np.zeros((testing_count, class_count), int)
    testing_node_indices = list()
    training_node_indices = list()

    def build_feat_and_class_vec(nodes, labels, feat_vec, indices_list):
        for node in nodes:
            if node.id not in adjacency_matrix.keys():
                adjacency_matrix[node.id] = list()
            labels[node.sub_id][node.class_id] = 1
            indices_list.append(node.id)
            feat_vec[node.sub_id][2] = hash(node.callStack)
            feat_vec[node.sub_id][3] = hash(node.eventName)
            for child in node.children:
                feat_vec[node.sub_id][1] += 1
                feat_vec[child.sub_id][0] += 1
                if child.id not in adjacency_matrix[node.id]:
                    adjacency_matrix[node.id].append(child.id)

    build_feat_and_class_vec(filter(lambda node: node.type == "training", nodes.values()), training_labels,
                             training_feature_vec,
                             training_node_indices)
    build_feat_and_class_vec(filter(lambda node: node.type == "testing", nodes.values()), testing_labels,
                             testing_feature_vec,
                             testing_node_indices)

    # Dump in pickle format
    pickle.dump(adjacency_matrix, open(join(output_path, "ind.{}.graph".format(dataset_name)), 'wb'))
    pickle.dump(training_feature_vec, open(join(output_path, "ind.{}.x".format(dataset_name)), 'wb'))
    pickle.dump(training_feature_vec, open(join(output_path, "ind.{}.allx".format(dataset_name)), 'wb'))
    pickle.dump(testing_feature_vec, open(join(output_path, "ind.{}.tx".format(dataset_name)), 'wb'))
    pickle.dump(training_labels, open(join(output_path, "ind.{}.y".format(dataset_name)), 'wb'))
    pickle.dump(training_labels, open(join(output_path, "ind.{}.ally".format(dataset_name)), 'wb'))
    pickle.dump(testing_labels, open(join(output_path, "ind.{}.ty".format(dataset_name)), 'wb'))
    # Dump as text the raw test indices (index in the adjacency matrix not the index in the feature matrix)
    with open(join(output_path, "ind.{}.test.index".format(dataset_name)), 'w') as test_index_file:
        for node in testing_node_indices:
            test_index_file.write(str(node) + "\r\n")

    return 0


def jn(text):
    return join("our_data_txt", text)


training_paths = {
    'filezilla': [jn("FileZilla10MB_joker.txt"),
                  jn("FileZilla20MB_madhatter.txt"),
                  jn("FileZilla50MB_joker.txt"),
                  jn("FileZilla100MB_madhatter.txt")],
    'winrar': [jn("winrar1.txt"),
               jn("winrar2.txt")],
    'skype_transfer':
        [jn("SkypeFileTransfer10MB.txt"),
         jn("SkypeFileTransfer50MB.txt")],
    'skype_video': [jn("SkypeVideoCall10min.txt"),
                    jn("SkypeVideoCall20mins.txt")],
    'spotify_music': [jn("Spotify10min.txt"),
                      jn("Spotify1min.txt")],
    'spotify_online': [jn("Spotify1min_online_singlesong.txt"), jn("Spotify5min_online_singlesong.txt")],
    'spotify_offline': [jn("Spotify10min_offline_singlesong.txt"), jn("Spotify1min_offline_singlesong.txt")]
}

testing_paths = {
    'winrar': [jn("winrar3.txt"),
               jn("winrar4.txt")],
    'filezilla': [jn("FileZilla10MB_madhatter.txt"),
                  jn("FileZilla20MB_joker.txt"),
                  jn("FileZilla50MB_madhatter.txt"),
                  jn("FileZilla100MB_joker.txt"), ],
    'skype_transfer': [jn("SkypeFileTransfer20MB.txt")],
    'skype_video': [jn("SkypeVideoCall15min.txt"),
                    jn("SkypeVideoCall1min.txt")],
    'spotify_music': [jn("Spotify5min.txt")],
    'spotify_online': [jn("Spotify10min_online_singlesong.txt")],
    'spotify_offline': [jn("Spotify5min_offline_singlesong.txt")],
}

prepare_data(training_paths, testing_paths, "n8ta", "data")
