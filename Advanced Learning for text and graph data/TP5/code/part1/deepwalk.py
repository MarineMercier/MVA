"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = list(G.neighbors(walk[i]))
        walk.append(neighbors[randint(0, len(neighbors)-1)])

    walk = list([str(node) for node in walk])
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    for i in range(num_walks):
        nodes = G.nodes()
        nodes = np.random.permutation(nodes)
        for j in range(nodes.shape[0]):
            walk = random_walk(G, nodes[j], walk_length)
            walks.append(walk)
    return walks

# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model