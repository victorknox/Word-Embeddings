from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import json
from  utils import *
from scipy import spatial

def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)
    
# read a file 
with open('./../models/vocab1.txt', 'r') as f:
    vocab_list = json.load(f)

new_model = keras.models.load_model('./../models/model2')

# Check its architecture

embedding_layer = new_model.get_layer('embedding')

embeds = embedding_layer.get_weights()[0]

def get_top10(word):
    word_index = vocab_list.index(word)

    similarities = {}

    for i in range(len(vocab_list)):
        similarities[vocab_list[i]] = cosine_similarity(embeds[word_index], embeds[i])

    # sort the similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # return the top 10
    top10 = sorted_similarities[1:11]
    return top10 

def plot_similarity(words):

    # plot utils
    # plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(words)))

    for word, c in zip(words, colors):
        # print the top 10
        print(word)
        top10 = get_top10(word)

        embeddings = np.array([embeds[vocab_list.index(word)]])
        for x in top10:
            print(x)
            embeddings = np.append(embeddings, [embeds[vocab_list.index(x[0])]], axis = 0)
        
        # transform the embeddings to 2D and plot them
        embeddings = TSNE(n_components=2).fit_transform(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], color = c)
        plt.annotate(word, xy=(embeddings[0, 0], embeddings[0, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    # save the plot
    plt.savefig('tsne2.png')
    # plt.show()

words = ['husband', 'jump', 'brown', 'yay', 'gym']
plot_similarity(words)

x = get_top10('camera')
for word in x:
    print(word)
