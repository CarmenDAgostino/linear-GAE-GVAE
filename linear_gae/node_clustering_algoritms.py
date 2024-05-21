from __future__ import division
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def node_clustering(emb, label, algoritm):
    """ Node Clustering: computes Adjusted Mutual Information score from a
    clustering of nodes in latent embedding space
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :param label: ground-truth node labels
    :param algorith: clustering algorithm 
    :return: Adjusted Mutual Information (AMI) score
    """

    mi_score = 0
    s_score = 0 
    nb_clusters = len(np.unique(label))

    if algoritm == 'kmeans':
        # K-Means Clustering
        clustering_pred = KMeans(n_clusters = nb_clusters, init = 'k-means++').fit(emb).labels_
        # Compute metrics
        mi_score = adjusted_mutual_info_score(label, clustering_pred)
        s_score = silhouette_score(label.reshape(-1, 1), clustering_pred) 

    elif algoritm == 'dbscan':
        # DBSCAN Clustering
        clustering_pred = DBSCAN(eps=0.1,min_samples=3,metric='cosine').fit(emb).labels_
        # Compute metrics
        mi_score = adjusted_mutual_info_score(label, clustering_pred)
        s_score = silhouette_score(label.reshape(-1, 1), clustering_pred)
        
    elif algoritm == 'agglomerative':
        # Agglomerative Clustering
        
        affinity = 'cosine'
        if np.any(np.all(emb == 0, axis=1)):
            affinity = 'euclidean'
        
        clustering_pred = AgglomerativeClustering(n_clusters=nb_clusters,linkage='average', affinity=affinity).fit(emb).labels_
        # Compute metrics
        mi_score = adjusted_mutual_info_score(label, clustering_pred)
        s_score = silhouette_score(label.reshape(-1, 1), clustering_pred)
    else: 
        raise ValueError('Undefined clustering algorithm!')
    
    return mi_score, s_score

