import os
import scipy.sparse as sp
import torch
from torch_geometric.datasets import TUDataset

import numpy as np
from sklearn.model_selection import train_test_split
import random


''' Funzione per scaricare i dataset dei grafi. Restituisce un array con
    una matrice di adiacenza per ogni grafo e un array di etichette '''

def load_graph_dataset(dataset_name):

    dataset_path = f'./data'
    dataset = TUDataset(root=dataset_path, name=dataset_name)

    adjacency_matrices = []
    graph_labels = []

    for data in dataset:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        adj_matrix[edge_index[0], edge_index[1]] = 1

        adjacency_matrices.append(adj_matrix)
        graph_labels.append(data.y.item())  

    return adjacency_matrices, graph_labels



''' Funzione per caricare le features dei nodi. Restituisce un array a tale che
    a[i] = matrice di dimensione (numero nodi grafo i) x (numero di features dei 
    nodi) contenente le features di ogni nodo del grafo i'''

def load_features(dataset_name):
    dataset_path = f"./data/{dataset_name}/raw"

    if not os.path.exists(f'{dataset_path}/{dataset_name}_node_attributes.txt'):
        return None

    with open(f'{dataset_path}/{dataset_name}_node_attributes.txt', 'r') as f:
        node_attributes = [list(map(float, line.strip().split(','))) for line in f]

    with open(f'{dataset_path}/{dataset_name}_graph_indicator.txt', 'r') as f:
        graph_indicators = [int(line.strip()) for line in f]

    node_attributes = np.array(node_attributes)
    graph_indicators = np.array(graph_indicators)

    num_graphs = np.max(graph_indicators)

    graph_feature_matrices = [[] for _ in range(num_graphs)]

    for idx, graph_id in enumerate(graph_indicators):
        graph_feature_matrices[graph_id - 1].append(node_attributes[idx])

    for idx in range(num_graphs):
        graph_feature_matrices[idx] = torch.tensor(np.array(graph_feature_matrices[idx]), dtype=torch.float)

    return graph_feature_matrices



''' Funzione per eseguire il padding dei grafi in modo che tutti 
    i grafi del dataset abbiano la stessa dimensione'''

def pad_graphs(adj_matrices, node_features):
    
    processed_adj_matrices = []
    processed_node_features = None if node_features is None else []

    target_num_nodes = np.max([matrix.shape[0] for matrix in adj_matrices])

    # Controllo necessario per problemi di memoria
    if target_num_nodes > 1000:
        target_num_nodes = 1000
    
    for i, adj in enumerate(adj_matrices):
        num_nodes = adj.shape[0]
        features = node_features[i] if node_features is not None else None
     
        if num_nodes > target_num_nodes:
            # Sampling: seleziona target_num_nodes nodi e crea il grafo indotto
            print("SAMPLING")
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
        if processed_node_features != None:
            processed_node_features.append(features)

    return processed_adj_matrices, processed_node_features, target_num_nodes



''' Funzione chiamata nel caso in cui i grafi del dataset non hanno nodi con features
    In questo caso le matrici delle features sono matrici identità'''

def features_control(features, num_graph, num_nodes):
    if features is not None:
        for i in range(num_graph):
            features[i] = sp.coo_matrix(features[i])
        return features

    features = []
    for i in range(num_graph):
        features.append(sp.eye(num_nodes))
    
    return features



''' Funzione che costruisce traing test e test set. Prende in input normal_class
    che è la classe considerata normale, le altre classi sono ritenute anomale.
    Divide i grafi della classe normale in training set (80%), test set (10%) e
    validation set (10%). Aggiunge grafi anomali al test set in modo proporzionale
    alla quantita di anomalie nel dataset originale.'''

def create_train_test_validation_sets(adj_matrices, node_features, labels, normal_class):
    normal_indices = [i for i, label in enumerate(labels) if label == normal_class]
    anomalous_indices = [i for i, label in enumerate(labels) if label != normal_class]

    normal_adj_matrices = [adj_matrices[i] for i in normal_indices]
    anomalous_adj_matrices = [adj_matrices[i] for i in anomalous_indices]

    if node_features is not None:
        normal_node_features = [node_features[i] for i in normal_indices]
    else:
        normal_node_features = None

    # Suddivisione dei grafi normali in training set, validation set e test set (80% training, 10% validation, 10% test)
    normal_train_adj_matrices, normal_temp_adj_matrices = train_test_split(
        normal_adj_matrices, test_size=0.2, random_state=42)
    normal_val_adj_matrices, normal_test_adj_matrices = train_test_split(
        normal_temp_adj_matrices, test_size=0.5, random_state=42)
    
    if normal_node_features is not None:
        normal_train_node_features, normal_temp_node_features = train_test_split(
            normal_node_features, test_size=0.2, random_state=42)
        normal_val_node_features, normal_test_node_features = train_test_split(
            normal_temp_node_features, test_size=0.5, random_state=42)
    else:
        normal_train_node_features = None
        normal_val_node_features = None
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

    # Costruzione del validation set
    val_adj_matrices = normal_val_adj_matrices
    val_node_features = normal_val_node_features
    val_labels = [1] * len(normal_val_adj_matrices)

    return train_adj_matrices, train_node_features, train_labels, val_adj_matrices, val_node_features, val_labels, test_adj_matrices, test_node_features, test_labels




'''
# Test 

# Caricamento del dataset
adj_matrices, labels= load_graph_dataset("ENZYMES")
node_features = load_features("ENZYMES")

print("Numero di grafi nel dataset:", len(adj_matrices))
print("Numero di etichette di classe:", len(labels))
print("Numero di matrici di features:", len(node_features) if node_features != None else "NO features")
print("Numero di feature dei nodi dei grafi:", node_features[0].shape[1] if node_features != None else "NO features")


# Trovo una matrice per stamparla
idx = 599
adj = adj_matrices[599]
for i, m in enumerate(adj_matrices):
    if m.shape[0] == -1 :
        idx = i
        adj = m

print(" Matrice: ")
adj = adj.numpy()
print(f" Dimensione: {adj.shape}")
print(" Label: ", labels[idx])
print(adj)

# Controllo features
print(" Features:")
f = node_features[idx] if node_features != None else sp.eye(0)
print(f"Type: {type(f)}  -  Dim:{f.shape}")
print(f)


# padding dei grafi
adj_matrices_2, node_features_2, num_nodes = pad_graphs(adj_matrices, node_features)

# Controllo sulle features
node_features_2 = features_control(node_features_2,len(adj_matrices_2),num_nodes)


# Controllo padding
for a,f in zip(adj_matrices_2,node_features_2) :
    if a.shape[0] != num_nodes or f.shape[0] != num_nodes :
       print(f"Diverso numero di nodi trovato nel grafo {i}: {a.shape[0]} nodi e {f.shape[0]} nodi nella matrice di features")

# Controllo allineamento
for i, adj in enumerate(adj_matrices_2):
    features = node_features_2[i]
    n = adj.shape[0]
    if features.shape[0] != n:
        print(f"Disallineamento trovato nel grafo {i}: {features.shape[0]} features ma {n} nodi")


# Controllo sui tipi
print(f" {type(adj_matrices_2)} - {type(labels)} - {type(node_features_2)}")
print(f" {type(adj_matrices_2[0])}     {type(labels[0])}      {type(node_features_2[0])}")


# Costruzione training test e validation set
train_adj_matrices, train_node_features, train_labels, val_adj_matrices, val_node_features, val_labels, test_adj_matrices, test_node_features, test_labels = create_train_test_validation_sets(adj_matrices_2, node_features_2, labels, normal_class=0)

print("Numero di grafi nel training set:", len(train_adj_matrices))
print("Numero di grafi nel test set:", len(test_adj_matrices))
print("Numero di grafi nel validation set:", len(val_adj_matrices))
print("Numero di etichette nel training set:", len(train_labels))
print("Numero di etichette nel test set:", len(test_labels))
print("Numero di etichette nel validation set:", len(val_labels))
'''