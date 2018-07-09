#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range2

from util import *
import pdb
from time import gmtime, strftime

from config import Config

class SeparationModel():
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        TODO: Add these placeholders to self as the instance variables
            self.inputs_placeholder
            self.targets_placeholder
            self.seq_lens_placeholder

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op. 
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features]. 

        (Don't change the variable names)
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features), name='inputs')
        self.targets_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.output_size), name='targets')

    def create_feed_dict(self, inputs_batch, targets_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """        
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.targets_placeholder: targets_batch,
        }

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete 
        in this function: 

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden]. 
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This 
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to 
          "logits". 

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """

        cell = None
        cell_bw = None
        if Config.num_layers > 1:
            # multi layer
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(Config.output_size, 
            input_size=Config.num_final_features) for _ in range2(Config.num_layers)], state_is_tuple=False)
            cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(Config.output_size, 
            input_size=Config.num_final_features) for _ in range2(Config.num_layers)], state_is_tuple=False)
        else:
            cell = tf.contrib.rnn.GRUCell(Config.output_size, input_size=Config.num_final_features)
            cell_bw = tf.contrib.rnn.GRUCell(Config.output_size, input_size=Config.num_final_features)

        # output, state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, dtype=tf.float32)
        output, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, self.inputs_placeholder, dtype=tf.float32)
        output = output[0]

        # output_seq_length = tf.shape(output)[1]
        # last_output = output[:,output_seq_length - 1,:]
        
        self.output = output
        # self.output = tf.Print(self.output, [self.output, tf.shape(self.output)])


    def add_loss_op(self, freq_weighted):
        l2_cost = 0.0

        weighted_differences = self.output - self.targets_placeholder

        num_freq_bins = Config.num_final_features
        frequencies = np.array([2.0 * 180 * i / (num_freq_bins - 1) * 22050 / 360 for i in range(num_freq_bins)])
        frequencies[0] = 2.0 * 180 / (num_freq_bins - 1) / 2 * 22050 / 360  # 0th frequency threshold is computed at 3/4th of the frequency range2
        ath_val = 3.64 * np.power(1000 / frequencies, 0.8) - 6.5 * np.exp(-0.6 * np.power(frequencies / 1000 - 3.3, 2)) + \
                  np.power(0.1, 3) * np.power(frequencies / 1000, 4)

        ath_shifted = (1 - np.amin(ath_val)) + ath_val  # shift all ath vals so that min is 1
        weights = np.tile(1 / ath_shifted, 2)

        if freq_weighted:
            weighted_differences = weights * weighted_differences
        else:
            normalized = np.full(weights.shape, np.sqrt(np.sum(np.power(weights, 2)) / num_freq_bins))
            weighted_differences = normalized * weighted_differences

        squared_error = tf.norm(weighted_differences, ord=2)
        self.loss = Config.l2_lambda * l2_cost + squared_error

        tf.summary.scalar("squared_error", squared_error)
        tf.summary.scalar("loss", self.loss)

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model. Call optimizer.minimize() on self.loss. 

        """
        optimizer = None 

        ### YOUR CODE HERE (~1-2 lines)
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)
        ### END YOUR CODE
        
        self.optimizer = optimizer

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    # This actually builds the computational graph 
    def build(self, freq_weighted):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op(freq_weighted)
        self.add_training_op()
        self.add_summary_op()


    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        output, batch_cost, summary = session.run([self.output, self.loss, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return output, batch_cost, summary

    #def print_results(self, train_inputs_batch, train_targets_batch):
    #    train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
    #    train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
    #    compare_predicted_to_true(train_first_batch_preds, train_targets_batch)

    def __init__(self, freq_weighted=None):
        self.build(freq_weighted)

    

