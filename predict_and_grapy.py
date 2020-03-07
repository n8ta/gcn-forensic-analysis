from spektral import utils
from util import prep
from graph import condense_and_plot
import sys
import tensorflow
from keras.models import load_model
import os
import pickle as pkl

input_data = "skype"


trace_types = ['filezilla', 'spotifyOnline', 'VLC', 'skype', 'spotifyOffline', 'winscp', 'winrar']

print("started loading saved model")
model = tensorflow.saved_model.load('trained')
print("finished loading model")

Anew, Xnew, _, _, _, _ = prep(input_data)
Anew_raw = Anew.toarray()  # array from of adj
Anew = utils.localpooling_filter(Anew).astype('f4')
Xnew = tensorflow.convert_to_tensor(Xnew, float)
Anew = tensorflow.convert_to_tensor(Anew.toarray(), float)
out = model([Xnew, Anew])

np = out.numpy()
pkl.dump(np, open("prediction.pkl",'wb'))

condense_and_plot(Anew_raw, out.numpy().tolist(), trace_types)
