import numpy as np
import pickle as pkl

with open("data/ind.cora.y", 'rb') as fyle:
    x = pkl.load(fyle, encoding='latin1')
    y = 2
