import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAD(object):
    """ Optimizer for anomaly detection task """

    def __init__(self, adj_input, adj_output, learning_rate, clip_norm=5.0):

        self.adj_input = tf.convert_to_tensor(adj_input.toarray().flatten(), dtype=tf.float32)#
        self.adj_output = adj_output #tf.reshape(adj_output, [adj_input.shape[0], adj_input.shape[0]]) #
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm

        # Calcolo l'errore di ricostruzione come la differenza tra le due matrici
        self.cost = tf.reduce_mean(tf.square(self.adj_input - self.adj_output))
        
        # Definisco l'ottimizzatore e l'operazione di minimizzazione
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.cost)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, self.clip_norm), var) 
                                  for grad, var in grads_and_vars if grad is not None]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
