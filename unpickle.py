import numpy as np
import pickle as pkl

with open("data/ind.citeseer.x", 'rb') as fyle:
    x = pkl.load(fyle, encoding='latin1')
    y = 2
