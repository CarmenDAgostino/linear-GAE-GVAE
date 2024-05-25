import numpy as np
import os
import networkx as nx


def load_graphs(dataset):
    base_dir = f"results/top_anomalous_graphs_{dataset}/"
    original_files = [f for f in os.listdir(base_dir) if 'original' in f]
    reconstructed_files = [f for f in os.listdir(base_dir) if 'reconstructed' in f]
    
    graphs = []
    
    for orig_file, recon_file in zip(sorted(original_files), sorted(reconstructed_files)):
        orig_path = os.path.join(base_dir, orig_file)
        recon_path = os.path.join(base_dir, recon_file)
        
        original_adj = np.load(orig_path)
        reconstructed_adj = np.load(recon_path)
        
        graphs.append((original_adj, reconstructed_adj))
    
    return graphs

def compute_node_error(original_adj, reconstructed_adj):
    errors = np.abs(original_adj - reconstructed_adj).sum(axis=1)
    return errors

def analyze_node_characteristics(graph, high_error_nodes, low_error_nodes):
    degrees = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    
    high_error_characteristics = {'degree': [], 'betweenness': [], 'closeness': []}
    low_error_characteristics = {'degree': [], 'betweenness': [], 'closeness': []}
   
    for node in high_error_nodes:
        high_error_characteristics['degree'].append(degrees[node])
        high_error_characteristics['betweenness'].append(betweenness[node])
        high_error_characteristics['closeness'].append(closeness[node])
    
    for node in low_error_nodes:
        low_error_characteristics['degree'].append(degrees[node])
        low_error_characteristics['betweenness'].append(betweenness[node])
        low_error_characteristics['closeness'].append(closeness[node])
    
    return high_error_characteristics, low_error_characteristics




avaible_dataset = ['AIDS']


for dataset in avaible_dataset: 

    # Load graph
    graphs = load_graphs(dataset)

    ok=True
    # Analize each graph
    for i, (original_adj, reconstructed_adj) in enumerate(graphs):
        #m = (reconstructed_adj > 0.1).astype(int)
        m = reconstructed_adj

        print(ok)
        if ok:
            print(m)
            ok=False
        errors = compute_node_error(original_adj, reconstructed_adj)

        # Get the nodes with highest e lowest error
        sorted_indices = np.argsort(errors)
        high_error_nodes = sorted_indices[-5:]
        low_error_nodes = sorted_indices[:5]

        
        high_results, low_results =  analyze_node_characteristics(nx.from_numpy_matrix(m),high_error_nodes,low_error_nodes)

        print(f"Graph {i+1} ({dataset}):")
        print("High error nodes:", high_error_nodes)
        print("Low error nodes:", low_error_nodes)
        print("High error node characteristics:", high_results)
        print("Low error node characteristics:", low_results)
        print("\n")