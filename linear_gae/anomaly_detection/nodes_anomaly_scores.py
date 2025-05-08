import numpy as np
import os
import networkx as nx
from sklearn.metrics import roc_curve


def load_graphs(dataset,model):
    base_dir = f"results/top_anomalous_graphs_{dataset}_{model}/"
    original_files = [f for f in os.listdir(base_dir) if 'original' in f]
    reconstructed_files = [f for f in os.listdir(base_dir) if 'reconstructed' in f]
    features_files = [f for f in os.listdir(base_dir) if 'features' in f]
    
    graphs = []
    
    for orig_file, recon_file, feat_file in zip(sorted(original_files), sorted(reconstructed_files), sorted(features_files)):
        orig_path = os.path.join(base_dir, orig_file)
        recon_path = os.path.join(base_dir, recon_file)
        feat_path = os.path.join(base_dir, feat_file)
        
        original_adj = np.load(orig_path)
        reconstructed_adj = np.load(recon_path)
        features = np.load(feat_path)
        
        graphs.append((original_adj, reconstructed_adj, features))
    
    return graphs

def compute_node_error(original_adj, reconstructed_adj):
    errors = np.abs(original_adj - reconstructed_adj).sum(axis=1)
    return errors

def find_optimal_threshold(original_adj, reconstructed_adj):
    fpr, tpr, thresholds = roc_curve(original_adj.flatten(), reconstructed_adj.flatten())
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    return optimal_threshold

def analyze_node_characteristics(graph, features, high_error_nodes, low_error_nodes):
    degrees = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    
    high_error_characteristics = {'degree': [], 'betweenness': [], 'closeness': [], 'features': []}
    low_error_characteristics = {'degree': [], 'betweenness': [], 'closeness': [], 'features': []}
   
    for node in high_error_nodes:
        high_error_characteristics['degree'].append(degrees[node])
        high_error_characteristics['betweenness'].append(betweenness[node])
        high_error_characteristics['closeness'].append(closeness[node])
        #high_error_characteristics['features'].append(features[node].tolist())
    
    for node in low_error_nodes:
        low_error_characteristics['degree'].append(degrees[node])
        low_error_characteristics['betweenness'].append(betweenness[node])
        low_error_characteristics['closeness'].append(closeness[node])
        #low_error_characteristics['features'].append(features[node].tolist())
    
    return high_error_characteristics, low_error_characteristics




avaible_dataset = ["IMDB-BINARY"]
avaible_models= ["gcn_ae","gcn_vae","linear_ae","linear_vae"]


for dataset in avaible_dataset: 
    print(f"DATESET: {dataset}")

    for model in avaible_models:
        print(f"MODEL: {model}")
        
        # Load graph
        graphs = load_graphs(dataset, model)

        # Analize each graph
        for i, (original_adj, reconstructed_adj, features) in enumerate(graphs):

            # Binarizzo la matrice ricostruita
            threshold = find_optimal_threshold(original_adj, reconstructed_adj)
            binarized_adjacency = np.where(reconstructed_adj > threshold, 1, 0)
            m = nx.from_numpy_matrix(binarized_adjacency)

            # Calcolo gli errori
            errors = compute_node_error(original_adj,m)

            # Ottengo i nodi con il più alto e il più basso errore
            sorted_indices = np.argsort(errors)

            # elimino i  nodi fittizi
            num_added_rows = 0
            for k in range(original_adj.shape[0] - 1, -1, -1):
                if np.all(original_adj[k, :] == 0):
                    num_added_rows += 1
                else:
                    break
            num_original_rows = original_adj.shape[0] - num_added_rows

            sorted_indices = [node for node in sorted_indices if node<num_original_rows]    
            high_error_nodes = sorted_indices[-5:]
            low_error_nodes = sorted_indices[:5]
            high_error_values = [errors[node] for node in high_error_nodes]
            low_error_values = [errors[node] for node in low_error_nodes]

            # Calcolo le caratteristice dei nodi
            high_results, low_results =  analyze_node_characteristics(m,features,high_error_nodes,low_error_nodes)

            print(f"Graph {i+1} ({dataset}):")
            print("High error nodes:", high_error_nodes)
            print("Low error nodes:", low_error_nodes)
            print("High error values:", high_error_values)
            print("Low error values:", low_error_values)
            print("High error node characteristics:")
            for key, values in high_results.items():
                print(f"  {key}: {values}")
            print("Low error node characteristics:")
            for key, values in low_results.items():
                print(f"  {key}: {values}")
            print("\n")