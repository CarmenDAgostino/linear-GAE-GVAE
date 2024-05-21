import networkx as nx
from linear_gae.input_data import load_data, load_label

# codice che calcola una serie di statistiche e misure sui tre dataset

datasets = ["cora", "citeseer", "pubmed"]

output_file="results/dataset_info.txt"
file = open(output_file, "w")

for dataset in datasets:
    # Caricamento del grafo
    adj, features = load_data(dataset)  
    labels = load_label(dataset)
    G = nx.from_scipy_sparse_matrix(adj)

    # Numero di nodi
    num_nodes = G.number_of_nodes()

    # Numero di archi
    num_edges = G.number_of_edges()
    
    # Etichette dei nodi
    num_classes = len(set(labels)) 
    unique_labels = set(labels)  
    labels_info = ", ".join(map(str, unique_labels))
    
    # Diretto
    directed = nx.is_directed(G)
    
    # Grado medio dei nodi
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    
    # Densità della rete
    density = nx.density(G)


    # controllo se il grafo è connesso
    # alcune misure non possono essere calcolate su grafi non connessi
    # in questo caso le calcolo sulla più grande componente connessa
    connected = nx.is_connected(G)
    H = nx.Graph()
    if not connected:
        connected_components = nx.connected_components(G)
        largest_component = max(connected_components, key=len)
        H = G.subgraph(largest_component)


    # Average path length
    avg_path_length = nx.average_shortest_path_length(G) if connected else nx.average_shortest_path_length(H)

    # Raggio della rete
    radius = nx.radius(G) if connected else nx.radius(H)

    # Diametro della rete
    diameter = nx.diameter(G) if connected else nx.diameter(H)


    # Centralità di grado media
    avg_degree_centrality = sum(nx.degree_centrality(G).values()) / num_nodes

    # Centralità di vicinanza media
    avg_closeness_centrality = sum(nx.closeness_centrality(G).values()) / num_nodes

    # Centralità di betweenness media
    avg_betweenness_centrality = sum(nx.betweenness_centrality(G).values()) / num_nodes
    
    # Clustering coefficient globale
    global_clustering_coefficient = nx.average_clustering(G)
    
    print(f"FASE 3  dataset: {dataset}")

    # Scrittura dei risultati nel file 
    file.write(f"Dataset: {dataset}\n")
    file.write(f"Numero di nodi: {num_nodes}\n")
    file.write(f"Numero di archi: {num_edges}\n")
    file.write(f"Diretto: {directed}\n")
    file.write(f"Connesso: {connected}\n")
    file.write(f"Numero di classi di etichette: {num_classes}\n")  
    file.write(f"Nomi delle etichette: {labels_info}\n")
    file.write(f"Grado medio dei nodi: {avg_degree}\n")
    file.write(f"Average Path Length: {avg_path_length}\n")
    file.write(f"Densità della rete: {density}\n")
    file.write(f"Raggio della rete: {radius}\n")
    file.write(f"Diametro della rete: {diameter}\n")
    file.write(f"Centralità di degree media: {avg_degree_centrality}\n")
    file.write(f"Centralità di betweenness media: {avg_betweenness_centrality}\n")
    file.write(f"Centralità di closeness media: {avg_closeness_centrality}\n")
    file.write(f"Coefficiente di clustering globale: {global_clustering_coefficient}\n\n")

file.close() 

print("Calculation of dataset statistics completed. Result saved in ", output_file)