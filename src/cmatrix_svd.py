from utils import *
import numpy as np
import json


# global variables
window_size = 3
k_value_svd = 100


# reading the json file
data = []
with open('./../dataset/Electronics_5.json') as f:
    count = 0
    for line in f:
        count += 1
        data.append(tokenize((json.loads(line))['reviewText']))
        if(count > 30000):
            break

# data = ['i enjoy flying .', 'i like deep learning .', 'i like nlp .']

vocab, vocab_list = make_vocab(data)

cmatrix = make_cmatrix(data, vocab, vocab_list, window_size)

svd = make_svd(cmatrix, k_value_svd)
print(svd.shape)

# save the svd
np.savetxt('model1.csv', svd)

# save the vocab 
with open('./vocab.txt', 'w') as f:
    json.dump(vocab_list, f)

