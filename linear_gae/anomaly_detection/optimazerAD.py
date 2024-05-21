import tensorflow as tf
from scipy.sparse import csr_matrix


flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAD(object):
    """ Optimizer for anomaly detection task """

    def __init__(self, adj_input, adj_output, learning_rate):
        self.adj_input = tf.convert_to_tensor(adj_input.toarray(), dtype=tf.float32)
        self.adj_output = tf.reshape(adj_output, [adj_input.shape[0], adj_input.shape[0]]) 
        self.learning_rate = learning_rate

        # Calcola l'errore di ricostruzione come la differenza tra le due matrici
        self.cost = tf.reduce_mean(tf.square(self.adj_input - self.adj_output))
        
        # Definisci l'ottimizzatore e l'operazione di minimizzazione
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        