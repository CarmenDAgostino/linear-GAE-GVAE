from __future__ import division, print_function

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

- ENZYMES: enzime molecules belonging to six classes, from TUDataset

- IMDB-BINARY: movie collaboration ego-networks, from TUDataset

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
flags.DEFINE_integer('batch_size',32,'Dimension of batch')
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


# Ensure that all graphs in the dataset have the same size with sampling and padding
processed_adjs, processed_features, num_nodes = sample_and_pad_graphs(adjs, features) 

# Check over features
if FLAGS.features == False or processed_features.__contains__(None):
    processed_features = []
    for i in range(len(processed_adjs)):
        processed_features.append(sp.eye(num_nodes))    

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
        model = GCNModelAE(placeholders, num_nodes, num_nodes)
    elif FLAGS.model == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(placeholders, num_nodes, num_nodes, num_nodes)
    elif FLAGS.model == 'linear_ae':
        # Linear Graph Autoencoder
        model = LinearModelAE(placeholders,num_nodes, num_nodes)
    elif FLAGS.model == 'linear_vae':
        # Linear Graph Variational Autoencoder
        model = LinearModelVAE(placeholders, num_nodes, num_nodes, num_nodes)
    elif FLAGS.model == 'deep_gcn_ae':
        # Deep (3-layer GCN) Graph Autoencoder
        model = DeepGCNModelAE(placeholders, num_nodes, num_nodes)
    elif FLAGS.model == 'deep_gcn_vae':
        # Deep (3-layer GCN) Graph Variational Autoencoder
        model = DeepGCNModelVAE(placeholders, num_nodes, num_nodes, num_nodes)
    else:
        raise ValueError('Undefined model!')        
    

    # Initialize TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for epoch in range(FLAGS.epochs):
        if FLAGS.verbose:
            print(f"Epoch {epoch+1}/{FLAGS.epochs}")
    
        t = time.time()
        avg_cost = 0
    
        # Calculate batch index
        batch_indices = random.sample(range(len(train_adjs)), FLAGS.batch_size)

        # Iterating over the batch   
        iter = 0
        for idx in batch_indices:
            
            iter += 1
            if FLAGS.verbose: 
                print(f"Training on item: {idx} (iteration {iter})")

            adj_init = train_adjs[idx]   
            label = train_labels[idx]
            features = train_features[idx]

            adj_tri = sp.triu(adj_init)
            adj = adj_tri + adj_tri.T         

            # Preprocessing and initialization
            num_nodes = adj.shape[0]
            features = sparse_to_tuple(features)
            num_features = features[2][1]
            features_nonzero = features[1].shape[0]
            
            # Normalization and preprocessing on adjacency matrix
            adj_norm = preprocess_graph(adj)  # Tuple contenente: - Coords: coordinate dei valori non zero come una matrice di forma (n_nonzero, 2) dove n_nonzero è il numero di valori non zero nella matrice originale. q- Values: I valori non zero stessi. Shape: La forma della matrice sparsa originale.
            adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

            # Optimizer
            with tf.name_scope('optimizer'):
                opt = OptimizerAD(  adj_input = adj,
                                    adj_output= model.reconstructions,
                                    learning_rate = FLAGS.learning_rate)
            
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features,placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            
            # Weights update
            outs = sess.run([opt.opt_op, opt.cost],feed_dict = feed_dict)
            
            # Accumulate average loss
            avg_cost += outs[1]

        avg_cost /= FLAGS.batch_size

        if FLAGS.verbose:
                # Display epoch information
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "time=", "{:.5f}".format(time.time() - t))
                

    if FLAGS.verbose:
        print("Training complete. Testing...")
        print(f"Test set dimension: {len(test_adjs)}\n")

    # Predicting labels 
    predicted_probabilities = []
    reconstructed_adjs = []

    for t in test_adjs:
        adj_tri = sp.triu(t)
        adj = adj_tri + adj_tri.T 
        
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
        
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        
        reconstruction = model.predict() 
        adj_output = sess.run(reconstruction, feed_dict=feed_dict)
        reconstructed_adjs.append(adj_output.reshape((num_nodes, num_nodes)))

        adj_input = sp.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])), shape=adj_norm[2])
        adj_input = adj_input.toarray().flatten()
        
        anomaly_score = np.sum(np.square(adj_input - adj_output))

        predicted_probabilities.append(anomaly_score)
    

    # Compute optimal threshold
    fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities)
    y = tpr - fpr
    index = np.argmax(y)
    threshold = thresholds[index]

    # Use the optimal threshold to compute labels
    predicted_labels = [1 if p < threshold else 0 for p in predicted_probabilities]

    # TODO
    print(f" Threshold: {threshold}")
    for w in range(15,30):
        print(f" prob: {predicted_probabilities[w]} label:  {predicted_labels[w]}")
    
    # Compute score
    auc = roc_auc_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    a_score = np.mean(predicted_probabilities)

    print(f"Run for class {classes[i]}")
    print(f"AUC score: {auc}")
    print(f"F1 score: {f1}")
    print(f"Mean anomaly score: {a_score}\n")
    
    mean_auc_score.append(auc)
    mean_f1_score.append(f1)
    mean_anomaly_score.append(a_score)

    # Compute training + test time
    t_model = time.time()
    mean_time.append(time.time() - t_start)

    # Find the top 5 graphs with the highest reconstruction error
    if classes[i] == 0 :
        num_top_graphs = 5

        errors_and_adjs = list(zip(predicted_probabilities, test_adjs, reconstructed_adjs))
        errors_and_adjs.sort(key=lambda x: x[0], reverse=True)
        top_graphs = errors_and_adjs[:num_top_graphs]

        output_dir = f"results/top_anomalous_graphs_{FLAGS.dataset}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for j, (error, adj, recon) in enumerate(top_graphs):
            adj_dense = adj.toarray() if sp.issparse(adj) else adj
            recon_dense = recon.reshape((num_nodes, num_nodes))

            adj_filename = os.path.join(output_dir, f"graph_{j+1}_error_{error:.5f}_original.npy")
            recon_filename = os.path.join(output_dir, f"graph_{j+1}_error_{error:.5f}_reconstructed.npy")
            
            np.save(adj_filename, adj_dense)
            np.save(recon_filename, recon_dense)

            if FLAGS.verbose:
                print(f"Saved adjacency matrix of graph {j+1} with error {error:.5f} to {adj_filename}")

###### Report Final Results ######

# Report final results
print("\nTest results for", FLAGS.model,
      "model on", FLAGS.dataset, "on anomaly detection" "\n",
      "___________________________________________________\n")

print("Mean AUC score: ", np.mean(mean_auc_score),
        "\nStd of AUC scores: ", np.std(mean_auc_score), "\n")

print("Mean F1 score: ", np.mean(mean_f1_score),
        "\nStd of F1 scores: ", np.std(mean_f1_score), "\n")

print("Mean Anomaly Detection score: ", np.mean(mean_anomaly_score),
        "\nStd of AD scores: ", np.std(mean_anomaly_score), "\n")

print("Total Running times\n", mean_time)
print("Mean total running time: ", np.mean(mean_time),
      "\nStd of total running time: ", np.std(mean_time), "\n\n")
