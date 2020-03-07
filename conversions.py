import os
import numpy as np
from os.path import join
import pickle
import json


class Node:
    count = 0
    sub_counts = {'training': 0, 'testing': 0}
    callstack_count = 0
    event_count = 0
    event_to_index = {}
    callstack_to_index = {}
    nodes = []

    def number_of_nodes_in_class_id(id):
        return len(list(filter(lambda node: node.class_id == id, Node.nodes)))


    def __init__(self, str, class_id, type, path):
        obj = json.loads(str)
        self.fileName = obj['filePath']
        self.eventName = obj['eventName']
        self.callStack = obj['callstack']

        if self.callStack not in Node.callstack_to_index.keys():
            Node.callstack_to_index[self.callStack] = Node.callstack_count
            Node.callstack_count += 1

        if self.eventName not in Node.event_to_index.keys():
            Node.event_to_index[self.eventName] = Node.event_count
            Node.event_count += 1

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
        Node.nodes.append(self)

    def one_hot_event_name(self):
        arr = [0 for x in range(size())]
        arr[Node.event_to_index[self.eventName]] = 1
        return np.array(arr, dtype=int)

    def one_hot_callstack(self):
        arr = [0 for x in range(size())]
        arr[Node.callstack_to_index[self.callStack]] = 1
        return np.array(arr, dtype=int)


def size():
    return max(Node.event_count, Node.callstack_count)


def prepare_data(training_paths, dataset_name, output_path):
    nodes = {}  # name -> node
    class_count = 0  # how many classes have we seen
    class_dict = {}  # class_name -> class id
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
                    try:
                        filePath = json.loads(line)['filePath']
                    except Exception:
                        x = 1
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
        print("Training class {}".format(class_name))
        prepare("training", class_dict[class_name], training_paths[class_name])
    weights = np.zeros(0)
    for class_name in training_paths.keys():
        count_in_class = Node.number_of_nodes_in_class_id(class_dict[class_name])
        weight_per_node = float(Node.count)/float(class_count*count_in_class)
        print("class is: {}, \t# nodes in class: {}, \ttotal nodes: {}\t weight per item in class: {}".
              format(class_name,count_in_class, Node.count, weight_per_node))
        class_weight = np.ones(count_in_class)*weight_per_node  # Weight all classes equally
        weights = np.hstack((weights,  class_weight))

    training_count = len(list(filter(lambda node: node.type == "training", nodes.values())))

    adjacency_matrix = np.zeros((training_count, training_count))

    training_feature_vec = np.zeros((training_count, 2,), float)
    # training_feature_vec = np.zeros((training_count, 4), float).tolist()

    training_labels = np.zeros((training_count, class_count), float)
    testing_node_indices = list()
    training_node_indices = list()

    def build_feat_and_class_vec(nodes, labels, feat_vec, indices_list):
        for node in nodes:
            labels[node.sub_id][node.class_id] = 1
            indices_list.append(node.id)
            # callstack one hot feature
            # feat_vec[node.sub_id][2] = Node.callstack_to_index[node.callStack]
            # feat_vec[node.sub_id][3] = Node.event_to_index[node.eventName]
            for child in node.children:
                feat_vec[node.sub_id][1] += 1
                feat_vec[child.sub_id][0] += 1
                adjacency_matrix[node.id][child.id] = 1

    build_feat_and_class_vec(filter(lambda node: node.type == "training", nodes.values()), training_labels,
                             training_feature_vec,
                             training_node_indices)

    # Dump in pickle formatodel
    pickle.dump(weights, open(join(output_path, "{}.weights".format(dataset_name)), 'wb'))
    pickle.dump(adjacency_matrix, open(join(output_path, "{}.graph".format(dataset_name)), 'wb'))
    pickle.dump(training_feature_vec, open(join(output_path, "{}.x.features".format(dataset_name)), 'wb'))
    pickle.dump(training_labels, open(join(output_path, "{}.y.labels".format(dataset_name)), 'wb'))
    # pickle.dump([Node.callstack_count, Node.callstack_to_index], open(join(output_path, "{}.callstack_dict".format(dataset_name)), 'wb'))
    # pickle.dump([Node.event_count, Node.event_to_index], open(join(output_path, "{}.callstack_dict".format(dataset_name)), 'wb'))
    # Dump as text the raw test indices (index in the adjacency matrix not the index in the feature matrix)
    with open(join(output_path, "ind.{}.test.index".format(dataset_name)), 'w') as test_index_file:
        for node in testing_node_indices:
            test_index_file.write(str(node) + "\r\n")

    return 0


def jn(text):
    return join("our_data_txt", text)


paths = {}
for dir in os.listdir("our_data_txt"):
    if dir == ".DS_Store" or dir == "attack":
        continue
    paths[dir] = [join("our_data_txt", dir, x) for i, x in enumerate(os.listdir(join("our_data_txt", dir))) if
                  x != ".DS_Store"]
print(paths.keys())
type = "skype"
paths = {type: [join("our_data_txt", type, x) for x in os.listdir(join("our_data_txt", type)) if x != ".DS_Store"]}
print(paths)

#
# source_data = "full"  # Use feature labels from this dataset for the subset
#
# f = open("data/{}.callstack_dict".format(source_data), 'rb')
# callstack_count, callstack_dict = pickle.load(f)
# f.close()
# f = open("data/{}.full_".format(source_data), 'rb')
# event_count, event_dict = pickle.load(f)
# f.close()
#
# prepare_data(paths, "skype", "data", callstack_count=callstack_count, callstack_dict=callstack_dict,
#              event_dict=event_dict, event_count=event_count)
#
#
prepare_data(paths, "skype", "data")
