from __future__ import division
from __future__ import print_function
from utils import *
from optimazerAD import *
from model_ad import *
from linear_gae.preprocessing import *
import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf 
import time
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, roc_curve


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


flags = tf.app.flags
FLAGS = flags.FLAGS


# Select graph dataset
flags.DEFINE_string('dataset', 'IMDB-BINARY', 'Name of the graphs dataset')
''' Available datasets:

- AIDS: molecules active or inactive against HIV, from TUDataset

- DD:  molecules enzime or non-enzimes, from TUDataset

- ENZYMES: enzime molecules belonging to six classes, from TUDataset

- IMDB-BINARY: movie collaboration ego-networks, from TUDataset

- REDDIT-MULTI-5K: Reddit discussion graphs across 5 categories, from TUDataset

Please check the TUDataset website for raw versions.
'''

# Model
flags.DEFINE_string('model', 'gcn_ae', 'Name of the model')
''' Available Models:

- gcn_ae: Graph Autoencoder from Kipf and Welling (2016), with 2-layer
          GCN encoder and inner product decoder

- gcn_vae: Graph Variational Autoencoder from Kipf and Welling (2016), with
           Gaussian priors, 2-layer GCN encoders for mu and sigma, and inner
           product decoder

- linear_ae: Linear Graph Autoencoder, as introduced in section 3 of NeurIPS
             workshop paper, with linear encoder, and inner product decoder

- linear_vae: Linear Graph Variational Autoencoder, as introduced in section 3
              of NeurIPS workshop paper, with Gaussian priors, linear encoders
              for mu and sigma, and inner product decoder
 
- deep_gcn_ae: Deeper version of Graph Autoencoder, as introduced in section 4
               of NeurIPS workshop paper, with 3-layer GCN encoder, and inner
               product decoder
 
- deep_gcn_vae: Deeper version of Graph Variational Autoencoder, as introduced
                in section 4 of NeurIPS workshop paper, with Gaussian priors,
                3-layer GCN encoders for mu and sigma, and inner product
                decoder
'''

# Model parameters
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('epochs', 50, 'Number of epochs in training.')
flags.DEFINE_boolean('features', False, 'Include node features or not in encoder')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 32, 'Number of units in GCN hidden layer(s).')
flags.DEFINE_integer('dimension', 16, 'Dimension of encoder output, i.e. \
                                       embedding dimension')
flags.DEFINE_integer('n_nodes',-1, 'Number of nodes in graphs')
flags.DEFINE_boolean('verbose', True, 'Whether to print comments details.')


# Lists to collect average results
mean_anomaly_score = []
mean_auc_score = []
mean_f1_score = []
mean_time = []


# Load graph dataset
if FLAGS.verbose:
    print("Loading data...")
adjs, labels, features = load_graph_dataset(FLAGS.dataset)

# Get the mean adj dim
avg_size = FLAGS.n_nodes
if FLAGS.n_nodes == -1 :
    avg_size = int( np.mean([matrix.shape[0] for matrix in adjs]) )

print(f" AVG  {avg_size}")
#  Ensure that all graphs in the dataset have the same size with sampling and padding
processed_adjs , processed_features = sample_and_pad_graphs(adjs, features,avg_size) 

# Check over features
if FLAGS.features == False or processed_features.__contains__(None):
    processed_features = []
    for i in range(len(processed_adjs)):
        processed_features.append(sp.eye(avg_size))    

# Number of classes
classes = np.unique(labels)

# The entire training+test process is repeated as many times as the number of classes
for i in range(classes.size):
    
    # Create training and test set
    train_adjs, train_features, train_labels, test_adjs, test_features, test_labels = \
        create_train_test_sets(processed_adjs , processed_features, labels, classes[i])

    # Start computation of running times
    t_start = time.time()

    # Model training
    if FLAGS.verbose:
            print("Training model...")
            print(f"Training set dimension: {len(train_adjs)}")

    # Define placeholders: servono come contenitori vuoti per i dati che verranno forniti in seguito, in fase di esecuzione
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32), # placeholder in TensorFlow che può contenere una matrice sparsa di tipo float32
        'adj': tf.sparse_placeholder(tf.float32),      # matrice di adiacenza normalizzata
        'adj_orig': tf.sparse_placeholder(tf.float32), # matrice di adiacenza non normalizzata
        'dropout': tf.placeholder_with_default(0., shape = ())
    }

    # Create model
    model = None
    if FLAGS.model == 'gcn_ae':
        # Standard Graph Autoencoder
        model = GCNModelAE(placeholders, avg_size, avg_size)
    elif FLAGS.model == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(placeholders, avg_size, avg_size, avg_size)
    elif FLAGS.model == 'linear_ae':
        # Linear Graph Autoencoder
        model = LinearModelAE(placeholders,avg_size, avg_size)
    elif FLAGS.model == 'linear_vae':
        # Linear Graph Variational Autoencoder
        model = LinearModelVAE(placeholders, avg_size, avg_size, avg_size)
    elif FLAGS.model == 'deep_gcn_ae':
        # Deep (3-layer GCN) Graph Autoencoder
        model = DeepGCNModelAE(placeholders, avg_size, avg_size)
    elif FLAGS.model == 'deep_gcn_vae':
        # Deep (3-layer GCN) Graph Variational Autoencoder
        model = DeepGCNModelVAE(placeholders, avg_size, avg_size, avg_size)
    else:
        raise ValueError('Undefined model!')        

    # Iterating over training set   
    for adj_index in range(1): #TODO len(train_adjs)
        
        if FLAGS.verbose: 
            print(f"Training on item: {adj_index}")

        adj_init = train_adjs[adj_index]   
        label = train_labels[adj_index] 
        features = train_features[adj_index] 

        # TODO
        adj_tri = sp.triu(adj_init)
        adj = adj_tri + adj_tri.T         

        # Preprocessing and initialization
        avg_size = adj.shape[0]
        features = sparse_to_tuple(features)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
       
        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)  # Tuple contenente: - Coords: coordinate dei valori non zero come una matrice di forma (n_nonzero, 2) dove n_nonzero è il numero di valori non zero nella matrice originale. q- Values: I valori non zero stessi. Shape: La forma della matrice sparsa originale.
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

        # Initialize TF session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Optimizer
        with tf.name_scope('optimizer'):
            opt = OptimizerAD(  adj_input = adj,
                                adj_output= model.reconstructions,
                                learning_rate = FLAGS.learning_rate)


        # Model training
        for epoch in range(FLAGS.epochs):
            # Flag to compute running time for each epoch
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Weights update
            outs = sess.run([opt.opt_op, opt.cost],feed_dict = feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            if FLAGS.verbose:
                # Display epoch information
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "time=", "{:.5f}".format(time.time() - t))
        
    # Compute embedding
    emb = sess.run(model.z_mean, feed_dict = feed_dict)

    if FLAGS.verbose:
        print("Training complete. Testing...")

    # Predicting labels 
    predicted_probabilities = []
    nodes_anomaly_scores = []

    for t in test_adjs:
        adj_tri = sp.triu(t)
        adj = adj_tri + adj_tri.T 
        
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
        
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        
        reconstruction = model.predict() 
        adj_output = sess.run(reconstruction, feed_dict=feed_dict)

        adj_input = sp.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])), shape=adj_norm[2])
        adj_input = adj_input.toarray().flatten()
        
        anomaly_score = np.sum(np.square(adj_input - adj_output))
        nodes_anomaly_scores.append( np.square(adj_input - adj_output) )

        predicted_probabilities.append(anomaly_score)
    

    # Compute optimal threshold
    fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities)
    y = tpr - fpr
    index = np.argmax(y)
    threshold = thresholds[index]

    # use the optimal threshold to compute labels
    predicted_labels = [1 if p < threshold else 0 for p in predicted_probabilities]
    
    auc = roc_auc_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    a_score = np.mean(predicted_probabilities)

    print(f"Run for class {classes[i]}")
    print(f"AUC score: {auc}")
    print(f"F1 score: {f1}")
    print(f"Mean anomaly score: {a_score}\n")
    

    # Find the top 5 nodes with the highest reconstruction error
    output_file = "results/top_5_anomalous_nodes.txt"
    file = open(output_file, "a")
    file.write(f"DATASET {FLAGS.dataset} MODEL: {FLAGS.model} Class: {classes[i]}")
    for j, node_scores in enumerate(nodes_anomaly_scores):
        top_5_nodes = np.argsort(node_scores)[-5:][::-1]  
        file.write(f"Top 5 nodes with highest reconstruction error in test graph {j}:")
        for node in top_5_nodes:
            file.write(f"Node {node} with anomaly score {node_scores[node]}")
    file.close()

    # Aggiunta degli score alle liste
    mean_auc_score.append(auc)
    mean_f1_score.append(f1)
    mean_anomaly_score.append(a_score)

    # Compute training time
    t_model = time.time()
    mean_time.append(time.time() - t_start)



###### Report Final Results ######

# Report final results
print("\nTest results for", FLAGS.model,
      "model on", FLAGS.dataset, "on anomaly detection" "\n",
      "___________________________________________________\n")

print("Mean AUC score: ", np.mean(mean_auc_score),
        "\nStd of AUC scores: ", np.std(mean_auc_score), "\n")

print("Mean F1 score: ", np.mean(mean_f1_score),
        "\nStd of AP scores: ", np.std(mean_f1_score), "\n")

print("Mean Anomale Detection score: ", np.mean(mean_anomaly_score),
        "\nStd of AP scores: ", np.std(mean_anomaly_score), "\n")

print("Total Running times\n", mean_time)
print("Mean total running time: ", np.mean(mean_time),
      "\nStd of total running time: ", np.std(mean_time), "\n \n")
