from __future__ import print_function
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

# G = json_graph.node_link_graph(json.load(open("ppi-G.json")))
# labels = json.load(open("ppi-class_map.json"))
# labels = {int(i):l for i, l in labels.items()}
# train_ids = [n for n in G.nodes if not G.nodes[n]['val'] and not G.nodes[n]['test']]
# test_ids = [n for n in G.nodes if G.nodes[n]['test']]
# train_labels = np.array([labels[i] for i in train_ids])
# if train_labels.ndim == 1:
#     train_labels = np.expand_dims(train_labels, 1)
# test_labels = np.array([labels[i] for i in test_ids])

edge_list = pd.read_csv("BioGrid_edgelist", sep = ' ', header = None)
edge_list = edge_list.drop(2, axis=1)
edge_list.rename(columns = {0:'source', 1: 'target'}, inplace = True)
#construct Graph from the edge list:
# create undirected graph from the edgelist
G=nx.from_pandas_edgelist(edge_list, source='source', target='target', create_using=nx.Graph())
# check the basic properties of the graph
# nx.info(G)
# G.nodes
#Create Random walks:
# function to generate random walk sequences of nodes for a particular node
def get_random_walk(node, walk_length):
    # initialization
    random_walk_length = [node]
    
    #loop over to get the nodes visited in a random walk
    for i in range(walk_length-1):
        # list of neighbors
        neighbors = list(G.neighbors(node))
        # if the same neighbors are present in ranom_walk_length list, then donot add them as new neighbors
        neighbors = list(set(neighbors) - set(random_walk_length))    
        if len(neighbors) == 0:
            break
        # pick any one neighbor randomly from the neighbors list
        random_neighbor = random.choice(neighbors)
        # append that random_neighbor to the random_walk_length list
        random_walk_length.append(random_neighbor)
        node = random_neighbor
        
    return random_walk_length

all_nodes = list(G.nodes())
number_of_random_walks = 1#5
random_walks = []

for node in tqdm(all_nodes):
    # number of random walks
    for i in range(number_of_random_walks):
        # append the random walk sequence of a node from a specified length
        random_walks.append(get_random_walk(node, 100))

type(random_walks)
#some nodes have random walks of length 3! most 100 to have 2d array all should be the same lenght
rw=random_walks
for i in range(len(rw)):
   if len(rw[i])< 100:
      a=len(rw[i])
      for q in range(100-len(rw[i])):
          xx=rw[i][len(rw[i])-1]
          # print(rw[i],'xx: ',xx)
          rw[i].append(xx)
          # print('after',rw[i],'xx: ',xx)

savetxt("biogrid_randomwalks.csv", rw, delimiter=",") #for server
rw = loadtxt("/content/drive/MyDrive/glove/biogrid_randomwalks.csv", delimiter=",")
rwX = rw[:,:]
from sklearn.manifold import Isomap
rwX=np.array(rwX)
print(rwX.shape, rw.shape)
from scipy.sparse.linalg import isolve
isolve = Isomap(n_components=100)
Isomapped_randomwalks = isolve.fit_transform(rwX)#### Caveat!! here the list indexes are the id of each vector in KeyedVectors its like a dictionary
savetxt("Isomapped_randomwalks.csv",Isomapped_randomwalks)

