from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from utils import *
import json
import numpy as np


# reading the json file
data = []
with open('./../dataset/Electronics_5.json') as f:
    count = 0
    for line in f:
        count += 1
        data.append(tokenize((json.loads(line))['reviewText']))
        if(count > 30000):
            break


vocab, vocab_list = make_vocab(data)

# calculating unigram probabilities
totalWords = sum([freq for freq in vocab.values()])
wordProb = {word:(freq/totalWords) for word, freq in vocab.items()}

# function to generate negative samples
def generate_negative_sample(wordProb):
    """
    This function takes as input a dict with keys as the 
    words in the vocab and values as the probabilities.
    Probabilities must sum to 1.
    """    
    word, context = (np.random.choice(list(wordProb.keys()), 
                     p=list(wordProb.values())) for _ in range(2))
    return word, context

word, context = generate_negative_sample(wordProb)

# add positive examples
posTrainSet = []

for i in range(1, len(vocab_list)-1):
    word = i
    context_words = [i-1, i+1]
    for context in context_words:
        posTrainSet.append((word, context))

n_pos_examples = len(posTrainSet)

# add the same number of negative examples
n_neg_examples = 0
negTrainSet = []

while n_neg_examples < n_pos_examples:
    (word, context) = generate_negative_sample(wordProb)
    # convert to indicies
    word, context = vocab_list.index(word), vocab_list.index(context)
    if (word, context) not in posTrainSet:
        negTrainSet.append((word, context))
        n_neg_examples += 1

X = np.concatenate([np.array(posTrainSet), np.array(negTrainSet)], axis=0)
y = np.concatenate([[1]*n_pos_examples, [0]*n_neg_examples])

N_WORDS = len(vocab.keys())
EMBEDDING_DIM = 300
embedding_layer = layers.Embedding(N_WORDS, EMBEDDING_DIM, 
                                   embeddings_initializer="RandomNormal",
                                   input_shape=(2,))
model = keras.Sequential([
  embedding_layer,
  layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X,y, batch_size=X.shape[0])

# save the model
model.save('./../models/model2')
