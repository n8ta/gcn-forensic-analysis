from spektral import utils
from util import prep
from graph import condense_and_plot
import sys
import tensorflow
from keras.models import load_model
import os

input_data = "skype_file"

print("started loading saved model")
model = tensorflow.saved_model.load('trained')

print("finished loading model")

Anew, Xnew, _, _, _, _ = prep(input_data)
Anew_raw = Anew.toarray()  # array from of adj
Anew = utils.localpooling_filter(Anew).astype('f4')
Xnew = tensorflow.convert_to_tensor(Xnew, float)
Anew = tensorflow.convert_to_tensor(Anew.toarray(), float)
out = model([Xnew, Anew])

trace_types = ['filezilla',
               'spotify_offline',
               'utorrent',
               'skype_file',
               'skype_video',
               'winscp',
               'spotify_online',
               'winrar']

np = out.numpy()
condense_and_plot(Anew_raw, out.numpy().tolist(), trace_types)
