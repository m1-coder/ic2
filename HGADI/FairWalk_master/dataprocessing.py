import numpy as np
import networkx as nx
from fairwalk import FairWalk
import torch
import pickle
import torch
import networkx as nx


def process_train_data_to_graph(train_data, original_to_homogeneous_map):

    graph = nx.Graph()
    for edge_type in train_data.edge_types:
        src, rel, dst = edge_type
        print(src, rel, dst)
        edge_index = train_data[src, rel, dst].edge_index

        for i in range(edge_index.shape[1]):
            node_1 = original_to_homogeneous_map[(src, int(edge_index[0, i]))]
            node_2 = original_to_homogeneous_map[(dst, int(edge_index[1, i]))]
            graph.add_edge(node_1, node_2)
            graph.nodes[node_1]['group'] = src
            graph.nodes[node_2]['group'] = dst

    return graph

def load_train_data_from_pickle(file_path):

    with open(file_path, 'rb') as f:
        train_data = pickle.load(f)
    return train_data


def get_edge_embeddings(graph):

    model = FairWalk(graph, dimensions=128, walk_length=10, num_walks=10, workers=1)
    model = model.fit(window=10, min_count=1, batch_words=1)

    EMBEDDING_FILENAME = './embeddings.emb'
    EMBEDDING_MODEL_FILENAME = './embeddings.model'

    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    model.save(EMBEDDING_MODEL_FILENAME)

    node_embeddings = model.wv
    return node_embeddings
