from scipy import sparse
import pickle as pkl
import numpy as np
import pickle as pkl
import sys

def prep(name):
    names = ["{}.graph".format(name), "{}.x.features".format(name), "{}.y.labels".format(name)]
    objects = []
    for i in range(len(names)):
        with open("data/{}".format(names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                print(names[i])
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    A, X, y = tuple(objects)

    A = sparse.csr_matrix(A, dtype=int)
    X = np.array(X)


    num_nodes = A.shape[0]

    x_training_feats = np.array([list(x) for i, x in enumerate(X) if i % 4 != 0 and i % 4 != 1])
    x_validation_feats = np.array([list(x) for i, x in enumerate(X) if i % 4 == 0])
    x_testing_feats = np.array([list(x) for i, x in enumerate(X) if i % 4 == 1])

    x_training_mask = np.array([(i % 4 != 0) and (i % 4 != 1) for i in
                                range(
                                    num_nodes)])  # every [False,False,True,True] repeated until we hit # training nodes
    x_validation_mask = np.array([(i % 4 == 0) for i in range(num_nodes)])
    x_testing_mask = np.array([(i % 4 == 1) for i in range(num_nodes)])

    y_training_labels = np.array([x for i, x in enumerate(y) if (i % 4 != 0) and (i % 4 != 1)])
    y_validation_labels = np.array([x for i, x in enumerate(y) if i % 4 == 0])
    y_testing_labels = np.array([x for i, x in enumerate(y) if i % 4 == 1])

    number_of_features = x_training_feats.shape[1]
    number_of_categories = y_training_labels.shape[1]
    return A, X, y, x_training_mask, x_validation_mask, x_testing_mask