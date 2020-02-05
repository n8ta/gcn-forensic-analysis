import numpy as np
import pickle as pkl

for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']:
    with open("data/ind.cora.{}".format(name), 'rb') as fyle:
        x = pkl.load(fyle, encoding='latin1')
        if name == "graph":
            print("{} : ({},{})".format(name, len(x),len(x)))
        else:
            print("{} : {}".format(name, x.shape))
