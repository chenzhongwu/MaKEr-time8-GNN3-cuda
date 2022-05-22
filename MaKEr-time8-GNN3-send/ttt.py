import pickle
import numpy as np
from utils import get_g, serialize
import torch
import lmdb
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import random
from scipy import sparse
import multiprocessing as mp
import numpy as np
# from scipy.sparse import coo_matrix
# row  = np.array([0, 1, 1, 0])
# col  = np.array([0, 1, 3, 2])
# data = np.array([4, 5, 7, 9])
# a = coo_matrix((data, (row, col)))
# # array([[4, 0, 9, 0],
# #        [0, 7, 0, 0],
# #        [0, 0, 0, 0],
# #        [0, 0, 0, 5]])
# print(a)
# print(a.row)
# print(a.col)




data = pickle.load(open('./data/fb_ext.pkl', 'rb'))
a = data['test']['support']
rel = dict()
for i , tri in enumerate(a):
    if (tri[0],tri[2])  in rel:
        rel[(tri[0],tri[2])]+=1
    else:
        rel[(tri[0], tri[2])] = 1
print(rel)



































