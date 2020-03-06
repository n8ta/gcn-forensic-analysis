from spektral import utils
from util import prep
from graph import condense_and_plot
import sys
import tensorflow
from keras.models import load_model
import os

print("started loading saved model")
model = tensorflow.saved_model.load('trained')
print("finished loading model")

Anew, Xnew, _, _, _, _ = prep("filezilla")
Anew_raw = Anew.toarray()  # array from of adj
Anew = utils.localpooling_filter(Anew).astype('f4')
out = model.predict([Xnew, Anew], batch_size=Anew.shape[0])

trace_types = ['utorrent', 'winrar', 'skype_file', 'winscp', 'spotify_online', 'filezilla', 'spotify_offline',
               'skype_video']

condense_and_plot(Anew_raw, out.tolist(), trace_types)