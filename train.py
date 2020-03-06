from spektral.layers import GraphConv
from spektral import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from util import prep
import tensorflow
from tensorflow.keras.optimizers import Adam

A, X, y, train_mask, val_mask, test_mask = prep("n8ta_full")
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
graph_conv_6 = GraphConv(A.shape[0], activation='relu')([graph_conv_5, A_in])
graph_conv_7 = GraphConv(n_classes, activation='softmax')([graph_conv_6, A_in])
graph_conv_8 = GraphConv(n_classes, activation='softmax')([graph_conv_7, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_conv_8)

A = utils.localpooling_filter(A).astype('f4')
optimizer = Adam(lr=1/100)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])
model.summary()

validation_data = ([X, A], y, val_mask)
history = model.fit([X, A],
                    y,
                    sample_weight=train_mask,
                    epochs=800,
                    batch_size=N,
                    validation_data=validation_data,
                    shuffle=False,
                    workers=8)

# Evaluate model
eval_results = model.evaluate([X, A],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
tensorflow.saved_model.save(model, "trained")

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
