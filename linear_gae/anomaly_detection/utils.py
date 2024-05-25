from typing import Counter
import torch
from torch_geometric.datasets import TUDataset

import numpy as np
from sklearn.model_selection import train_test_split
import random


''' Funzione per scaricare i dataset dei grafi. Restituisce un array con
    una matrice di adiacenza per ogni grafo, un array di etichette e un
    array con lefeatures dei nodi se presenti.'''

def load_graph_dataset(dataset_name):

    dataset_path = f'./data/{dataset_name}'

    dataset = TUDataset(root=dataset_path, name=dataset_name)

    adjacency_matrices = []
    graph_labels = []
    node_features = []

    for data in dataset:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        adj_matrix[edge_index[0], edge_index[1]] = 1

        adjacency_matrices.append(adj_matrix)
        graph_labels.append(data.y.item())  
        if data.x is not None:
            node_features.append(data.x)
        else:
            node_features.append(None)

    return adjacency_matrices, graph_labels, node_features


''' Funzione per eseguire il padding dei grafi in modo che tutti 
    i grafi del dataset abbiano la stessa dimensione'''

def sample_and_pad_graphs(adj_matrices, node_features):
    
    processed_adj_matrices = []
    processed_node_features = []

    target_num_nodes = np.max([matrix.shape[0] for matrix in adj_matrices])

    # Controllo necessario per problemi di memoria
    if target_num_nodes > 2000:
        target_num_nodes = 2000
    
    for adj, features in zip(adj_matrices, node_features):
        num_nodes = adj.shape[0]
     
        if num_nodes > target_num_nodes:
            # Sampling: seleziona target_num_nodes nodi e crea il grafo indotto
            nodes = random.sample(range(num_nodes), target_num_nodes)
            nodes.sort() 
            adj = adj[nodes, :][:, nodes]
            if features is not None:
                features = features[nodes, :]
        
        elif num_nodes < target_num_nodes:
            # Padding: aggiungi nodi fittizi
            pad_matrix = torch.zeros((target_num_nodes, target_num_nodes))
            pad_matrix[:num_nodes, :num_nodes] = adj
            adj = pad_matrix

            if features is not None:
                pad_features = torch.zeros((target_num_nodes, features.shape[1]))
                pad_features[:num_nodes, :] = features
                features = pad_features

        processed_adj_matrices.append(adj)
        processed_node_features.append(features)

    return processed_adj_matrices, processed_node_features, target_num_nodes



''' Funzione che costruisce traing test e test set. Prende in input normal_class
    che è la classe considerata normale, le altre classi sono ritenute anomale.
    Divide i grafi della classe normale in training set (90%) e test set (10%), 
    e aggiunge grafi anomali al test set in modo proporzionale alla quantita di
    anomalie nel dataset originale.'''

def create_train_test_sets(adj_matrices, node_features, labels, normal_class):

    normal_indices = [i for i, label in enumerate(labels) if label == normal_class]
    anomalous_indices = [i for i, label in enumerate(labels) if label != normal_class]

    normal_adj_matrices = [adj_matrices[i] for i in normal_indices]
    anomalous_adj_matrices = [adj_matrices[i] for i in anomalous_indices]

    if node_features is not None:
        normal_node_features = [node_features[i] for i in normal_indices]
    else:
        normal_node_features = None

    # Suddivisione dei grafi normali in training set e test set (90% training, 10% test)
    (normal_train_adj_matrices, normal_test_adj_matrices) = train_test_split(
        normal_adj_matrices, test_size=0.1, random_state=42)
    
    if normal_node_features is not None:
        normal_train_node_features, normal_test_node_features = train_test_split(
            normal_node_features, test_size=0.1, random_state=42)
    else:
        normal_train_node_features = None
        normal_test_node_features = None

    # Selezione di grafi anomali casuali per il test set
    num_anomalous_test = len(normal_test_adj_matrices) * len(anomalous_adj_matrices) // len(normal_adj_matrices)
    selected_anomalous_indices = random.sample(anomalous_indices, num_anomalous_test)
    
    selected_anomalous_adj_matrices = [adj_matrices[i] for i in selected_anomalous_indices]
    if node_features is not None:
        selected_anomalous_node_features = [node_features[i] for i in selected_anomalous_indices]
    else:
        selected_anomalous_node_features = None

    # Costruzione del test set
    test_adj_matrices = normal_test_adj_matrices + selected_anomalous_adj_matrices
    if normal_test_node_features is not None and selected_anomalous_node_features is not None:
        test_node_features = normal_test_node_features + selected_anomalous_node_features
    else:
        test_node_features = None
        
    test_labels = [1] * len(normal_test_adj_matrices) + [0] * len(selected_anomalous_adj_matrices)

    # Costruzione del training set
    train_adj_matrices = normal_train_adj_matrices
    train_node_features = normal_train_node_features
    train_labels = [1] * len(normal_train_adj_matrices)

    return train_adj_matrices, train_node_features, train_labels, test_adj_matrices, test_node_features, test_labels


'''
# Test 
adj_matrices, labels, node_features = load_graph_dataset("IMDB-BINARY")

if node_features[0] == None:
    print("No features")

print("Numero di grafi nel dataset:", len(adj_matrices))
print("Numero di etichette di classe:", len(labels))
print(f"Tipe node_features {type(node_features)}")
if node_features[0] != None:
    print("Numero di feature dei nodi per il primo grafo:", node_features[0].shape[1])

print(adj_matrices[3])
print(labels[3])
print("\n")


adj_matrices_2, node_features_2, num_nodes = sample_and_pad_graphs(adj_matrices, node_features)
for a,f in zip(adj_matrices_2,node_features_2) :
    if a.shape[0] != num_nodes :
       print(f" diverso   {a.shape[0]} ")
    if f != None and a.shape[0] != f.shape[0]:
       print(f" diverso   {a.shape[0]}  - {f.shape[0]}") 


train_adj_matrices, train_node_features, train_labels, test_adj_matrices, test_node_features, test_labels = create_train_test_sets(adj_matrices_2, node_features_2, labels, normal_class=0)

print("Numero di grafi nel training set:", len(train_adj_matrices))
print("Numero di grafi nel test set:", len(test_adj_matrices))
print("Numero di etichette nel training set:", len(train_labels))
print("Numero di etichette nel test set:", len(test_labels))

'''