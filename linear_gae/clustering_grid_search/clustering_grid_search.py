import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import numpy as np


# Codice che effettua la grid search per i parametri di KMeans DBSCAN e AgglomerativeClustering
# l'analisi è stata eeffettuata basandosi sugli embeddings e le labels ottenute addestrando
# il modello linear_ae sul dataset Cora


# Caricamento degli embeddings e delle labels
emb_df = pd.read_csv("embeddings.csv")
labels_df = pd.read_csv("labels.csv")

emb = emb_df.values
labels = labels_df.values.flatten() 


# Grid Search per l'algoritmo KMeans
print("Grid search for KMeans ...")

# parametri
n = len(np.unique(labels))

n_clusters_values = [n-2, n-1, n, n+1, n+2]
init_values = ['k-means++', 'random']

best_ami = -1
best_s = -1
best_n_clusters_ami = None
best_init_ami = None
best_n_clusters_s = None
best_init_s = None

for n_c in n_clusters_values:
    for i in init_values:
        clustering_pred = KMeans(n_clusters = n_c, init = i).fit(emb).labels_

        mi_score = adjusted_mutual_info_score(labels, clustering_pred)
        s_score = silhouette_score(labels.reshape(-1, 1), clustering_pred)

        if mi_score > best_ami:
            best_ami = mi_score
            best_n_clusters_ami = n_c
            best_init_ami = i
        
        if s_score > best_s:
            best_s = s_score
            best_n_clusters_s = n_c
            best_init_s = i


# Salvataggio dei migliori parametri
file = file = open("grid_search_results.txt", "w")

file.write(f"Best parameter for Kmeans:\n")
file.write(f" AMI score: {best_ami} \n" )
file.write(f" n_cluster: { best_n_clusters_ami}\n")
file.write(f" init: { best_init_ami}\n\n")

file.write(f" Silhouette score: {best_s}\n" )
file.write(f" n_cluster: {best_n_clusters_s}\n")
file.write(f" init: {best_init_s}\n\n\n")

print("Grid search for KMeans complete. Best param: n_cluste: ", best_n_clusters_ami," init: ",best_init_ami,)



# Grid Search per l'algoritmo DBSCAN
print("Grid search for DBSCAN ...")

# parametri
eps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
min_samples_values = [3, 5, 7, 10, 12, 15]
metric_values = ['euclidean', 'manhattan', 'cosine']

best_ami_d = -1
best_s_d = -1
best_eps_ami = None
best_min_samples_ami = None
best_metric_ami = None
best_eps_s = None
best_min_samples_s = None
best_metric_s = None

for e in eps_values:
    for min in min_samples_values:
        for m in metric_values:

            clustering_pred = DBSCAN(eps=e,min_samples=min,metric=m).fit(emb).labels_

            mi_score = adjusted_mutual_info_score(labels, clustering_pred)
            if len(np.unique(clustering_pred)) > 1:  
                s_score = silhouette_score(labels.reshape(-1, 1), clustering_pred)

            if mi_score > best_ami_d:
                best_ami_d = mi_score
                best_eps_ami = e
                best_min_samples_ami = min
                best_metric_ami = m
            
            if len(np.unique(clustering_pred)) > 1 and s_score > best_s_d:
                best_s_d = s_score
                best_eps_s = e
                best_min_samples_s = min
                best_metric_s = m


# Salvataggio dei migliori parametri

file.write(f"Best parameter for DBSCAN:\n")
file.write(f" AMI score: { best_ami_d}\n" )
file.write(f" eps: { best_eps_ami}\n")
file.write(f" min_samples: { best_min_samples_ami}\n")
file.write(f" metric: { best_metric_ami}\n\n")

file.write(f" Silhouette score: { best_s_d}\n" )
file.write(f" eps: {best_eps_s}\n")
file.write(f" min_samples: { best_min_samples_s}\n")
file.write(f" metric: { best_metric_s}\n\n\n")

print("Grid search for DBSCAN complete. Best param: eps: ", best_eps_ami," min_samples: ",best_min_samples_ami, " metric: ", best_metric_ami)





# Grid Search per l'algoritmo AgglomerativeClustering
print("Grid search for AgglomerativeClustering ...")

# parametri
n_clusters_values = [n-2, n-1, n, n+1, n+2]
linkage_values =  ['average', 'ward', 'complete', 'single']
affinity_values = ['manhattan', 'euclidean', 'cosine']

best_ami_a = -1
best_s_a = -1
best_n_cluster_a_ami = None
best_linkage_ami = None
best_affinity_ami = None
best_n_cluster_s = None
best_linkage_s = None
best_affinity_s = None

for n_c in n_clusters_values:
    for l in linkage_values:
        for a in affinity_values:

            if l == 'ward' and a != 'euclidean':
                continue
            clustering_pred = AgglomerativeClustering(n_clusters=n_c,linkage=l, affinity=a).fit(emb).labels_
        
            mi_score = adjusted_mutual_info_score(labels, clustering_pred)
            s_score = silhouette_score(labels.reshape(-1, 1), clustering_pred)

            if mi_score > best_ami_a:
                best_ami_a = mi_score
                best_n_cluster_a_ami = n_c
                best_linkage_ami = l
                best_affinity_ami = a
            
            if s_score > best_s_a:
                best_s_a = s_score
                best_n_cluster_a_s = n_c
                best_linkage_s = l
                best_affinity_s = a


# Salvataggio dei migliori parametri

file.write(f"Best parameter for AgglomerativeClustering:\n")
file.write(f" AMI score: {best_ami_a}\n" )
file.write(f" n_cluster: { best_n_cluster_a_ami}\n")
file.write(f" linkage: { best_linkage_ami}\n")
file.write(f" affinity: {best_affinity_ami}\n\n")

file.write(f" Silhouette score: {best_s_a}\n" )
file.write(f" n_cluster: { best_n_cluster_a_s}\n")
file.write(f" linkage: {best_linkage_s}\n")
file.write(f" affinity: {best_affinity_s}")

print("Grid search for AgglomerativeClustering complete. Best param: n_cluster: ", best_n_cluster_a_ami," linkage: ",best_linkage_ami, " affinity: ", best_affinity_ami)

file.close()