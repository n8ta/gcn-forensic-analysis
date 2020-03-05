import numpy as np
from scipy import sparse
import pickle as pkl
import sys
from spektral.layers import GraphConv
from spektral import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
import tensorflow
import matplotlib.pyplot as plt
from graph import condense_and_plot


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
    X = sparse.csr_matrix(X, dtype=float)

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
    return A, X.toarray(), y, x_training_mask, x_validation_mask, x_testing_mask


A, X, y, train_mask, val_mask, test_mask = prep("n8ta")
A_raw = A.toarray()

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]

# Model definition
X_in = Input(shape=(F,))  # Input layer for X
A_in = Input((None,), sparse=True)  # Input layer for A

graph_conv_1 = GraphConv(A.shape[0], activation='relu')([X_in, A_in])
graph_conv_2 = GraphConv(A.shape[0], activation='relu')([graph_conv_1, A_in])
graph_conv_3 = GraphConv(A.shape[0], activation='relu')([graph_conv_2, A_in])
graph_conv_4 = GraphConv(A.shape[0], activation='relu')([graph_conv_3, A_in])
graph_conv_5 = GraphConv(A.shape[0], activation='relu')([graph_conv_4, A_in])
# dropout = Dropout(0.5)(graph_conv_5)
graph_conv_6 = GraphConv(n_classes, activation='softmax')([graph_conv_5, A_in])
graph_conv_7 = GraphConv(n_classes, activation='softmax')([graph_conv_6, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_conv_7)

A = utils.localpooling_filter(A).astype('f4')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', weighted_metrics=['acc'])

model.summary()
validation_data = ([X, A], y, val_mask)
history = model.fit([X, A],
                    y,
                    sample_weight=train_mask,
                    epochs=1,
                    batch_size=N,
                    validation_data=validation_data,
                    shuffle=False)

# Evaluate model
eval_results = model.evaluate([X, A],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.legend()
# plt.show()

print('Done.\nTest loss: {}\nTest accuracy: {}'.format(*eval_results))

Anew, Xnew, _, _, _, _ = prep("testing")
Anew_raw = Anew.toarray()  # array from of adj
Anew = utils.localpooling_filter(Anew).astype('f4')
out = model.predict([Xnew, Anew], batch_size=Anew.shape[0])


trace_types = ['filezilla',
'winrar',
'skype_transfer',
'skype_video',
'spotify_online',
'spotify_offline',
'other']

condense_and_plot(Anew_raw, out.tolist(), trace_types)