# load the model
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

svd = np.loadtxt('./../models/model1.csv')

def get_top10(word):
    word_index = vocab_list.index(word)

    similarities = {}

    for i in range(svd.shape[0]):
        similarities[vocab_list[i]] = cosine_similarity(svd[word_index], svd[i])

    # sort the similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # return the top 10
    top10 = sorted_similarities[1:11]
    return top10    

def plot_similarity(words):

    # plot utils
    plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(words)))

    for word, c in zip(words, colors):
        # print the top 10
        print(word)
        top10 = get_top10(word)

        embeds = np.array([svd[vocab_list.index(word)]])
        for x in top10:
            print(x)
            embeds = np.append(embeds, [svd[vocab_list.index(x[0])]], axis = 0)
        
        # transform the embeddings to 2D and plot them
        embeds = TSNE(n_components=2).fit_transform(embeds)
        plt.scatter(embeds[:, 0], embeds[:, 1], color = c)
        plt.annotate(word, xy=(embeds[0, 0], embeds[0, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    # save the plot
    plt.savefig('tsne1.png')
    # plt.show()

words = ['husband', 'jump', 'brown', 'yay', 'gym']
plot_similarity(words)

camera_sims = get_top10('camera')
print("camera")
for word in camera_sims:
    print(word)