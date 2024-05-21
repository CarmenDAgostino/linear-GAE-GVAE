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

- IMDB-BINARY: IMDB-BINARY movie collaboration ego-networks, from TUDataset

- REDDIT-BINARY: REDDIT-BINARY discussions network on Reddit, from TUDataset

- TODO

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
flags.DEFINE_integer('epochs', 200, 'Number of epochs in training.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 32, 'Number of units in GCN hidden layer(s).')
flags.DEFINE_integer('dimension', 16, 'Dimension of encoder output, i.e. \
                                       embedding dimension')

# Experimental setup parameters
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
target_node = 37 
processed_adjs , processed_features = sample_and_pad_graphs(adjs, features,target_node)

classes = np.unique(labels)

# The entire training+test process is repeated as many times as the number of classes
for i in range(classes.size):
    

    
    train_adjs, train_node_features, train_labels, test_adjs, test_node_features, test_labels = \
        create_train_test_sets(processed_adjs , processed_features, labels, classes[i])

    # Start computation of running times
    t_start = time.time()

    # Model training
    if FLAGS.verbose:
            print("Training model...")

    max = 0
    min = 100000000
    mean = 0
    for a in train_adjs:
        mean += a.shape[0]
        if a.shape[0] > max :
            max = a.shape[0]
        if a.shape[0] < min :
            min = a.shape[0]
    
    print(f"       max: {max} min: {min} mean: {mean/len(train_adjs)}")

    for adj_index in range(1):  #len(train_adjs) TODO
        
        adj_init = train_adjs[adj_index]   #Tensore
        label = train_labels[adj_index] 
        features = sp.identity(adj_init.shape[0])  #TODO

        adj_tri = sp.triu(adj_init)
        adj = adj_tri + adj_tri.T          # Tensore

        # Preprocessing and initialization
        if FLAGS.verbose:
            print("Preprocessing and Initializing...")
        
        # Compute number of nodes
        num_nodes = adj.shape[0]
        # Preprocessing on node features
        features = sparse_to_tuple(features)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

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
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif FLAGS.model == 'gcn_vae':
            # Standard Graph Variational Autoencoder
            model = GCNModelVAE(placeholders, num_features, num_nodes,
                                features_nonzero)
        elif FLAGS.model == 'linear_ae':
            # Linear Graph Autoencoder
            model = LinearModelAE(placeholders, num_features, features_nonzero)
        elif FLAGS.model == 'linear_vae':
            # Linear Graph Variational Autoencoder
            model = LinearModelVAE(placeholders, num_features, num_nodes,
                                features_nonzero)
        elif FLAGS.model == 'deep_gcn_ae':
            # Deep (3-layer GCN) Graph Autoencoder
            model = DeepGCNModelAE(placeholders, num_features, features_nonzero)
        elif FLAGS.model == 'deep_gcn_vae':
            # Deep (3-layer GCN) Graph Variational Autoencoder
            model = DeepGCNModelVAE(placeholders, num_features, num_nodes,
                                    features_nonzero)
        else:
            raise ValueError('Undefined model!')        

        # Optimizer
        with tf.name_scope('optimizer'):
            opt = OptimizerAD(  adj_input = adj,
                                adj_output= model.reconstructions,
                                learning_rate = FLAGS.learning_rate)
            
        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)  # Tuple contenente: - Coords: coordinate dei valori non zero come una matrice di forma (n_nonzero, 2) dove n_nonzero è il numero di valori non zero nella matrice originale. q- Values: I valori non zero stessi. Shape: La forma della matrice sparsa originale.
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

        # Initialize TF session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

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
                i=0  # TODO
                # Display epoch information
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "time=", "{:.5f}".format(time.time() - t))
        
    # Compute embedding
    emb = sess.run(model.z_mean, feed_dict = feed_dict)

    if FLAGS.verbose:
        print("Training complete. Testing...")


    # Predicting labels 
    predicted_probabilities = []

    for t in test_adjs:
        #if t.shape[0] == 19:
        adj_tri = sp.triu(test_adjs[1])
        adj = adj_tri + adj_tri.T 
        
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
        #print(f" adj_norm: {type(adj_norm)}  dim: {adj_norm[0].shape}")
        
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        
        reconstruction = model.predict() #SparseTensor
        #print(f" rec : {type(reconstruction)}dim: {reconstruction.shape}")

        adj_output = sess.run(reconstruction, feed_dict=feed_dict)
        #print(f" adj_output: {type(adj_output)}")

        adj_input = sp.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])), shape=adj_norm[2])
        #print(f" adj_output: {adj_output.shape}  adj_input: {adj_input.shape}")
        adj_input = adj_input.toarray().flatten()
        
        anomaly_score = np.sum(np.square(adj_input - adj_output))
        #print(f"anomaly_score: {anomaly_score}  type: {type(anomaly_score)}")

        predicted_probabilities.append(anomaly_score)
    


    #adj_norm = tf.sparse.reorder(adj_norm)
    #adj_input = tf.sparse.to_dense(adj_norm)
    #adj_output = tf.reshape(reconstruction, [adj_norm.shape[0], adj_norm.shape[0]])
    #adj_output = tf.reshape(reconstruction, [adj_norm.shape[0], adj_norm.shape[0]])
    
    # flattenizza adj_norm
    # matrice X con parametro
    # anomaly score singoli nodi

    # Compute optimal threshold
    print(f" l type: {type(test_labels)}")
    print(f" p type: {type(predicted_probabilities[0])}")

    fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities)
    y = tpr - fpr
    index = np.argmax(y)
    threshold = thresholds[index]

    # use the optimal threshold to compute labels
    predicted_labels = [1 if p < threshold else 0 for p in predicted_probabilities]
    
    # TODO
    print(f"threshold: {threshold}\n")
    for w in range(5):
        print(f" predicted probability: {predicted_probabilities[w]}  -  predicted_labels: {predicted_labels[w]}")
    
    # Test model
    if FLAGS.verbose:
        print("Testing model...")

    auc = roc_auc_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    print(f"Run for class {classes[i]}\n")
    print(f"AUC score: {auc}\n")
    print(f"F1 score: {f1}\n\n")
    
    # Aggiunta degli score alle liste
    mean_auc_score.append(auc)
    mean_f1_score.append(f1)


    # Compute training time
    t_model = time.time()
    mean_time.append(time.time() - t_start)

# Calcolo della media degli score
mean_auc = np.mean(mean_auc_score)
mean_f1 = np.mean(mean_f1_score)


###### Report Final Results ######

# Report final results
print("\nTest results for", FLAGS.model,
      "model on", FLAGS.dataset, "on anomaly detection" "\n",
      "___________________________________________________\n")

print("Mean AUC score: ", np.mean(mean_auc_score),
        "\nStd of AUC scores: ", np.std(mean_auc_score), "\n \n")

print("Mean F1 score: ", np.mean(mean_f1_score),
        "\nStd of AP scores: ", np.std(mean_f1_score), "\n \n")

print("Total Running times\n", mean_time)
print("Mean total running time: ", np.mean(mean_time),
      "\nStd of total running time: ", np.std(mean_time), "\n \n")
