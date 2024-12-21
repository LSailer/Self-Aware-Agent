from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import tensorflow as tf


class ConvNet(object):
    """Basic implementation of ConvNet class compatible with tfutils.
    """

    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.output = None
        self._params = OrderedDict()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def build_model(self, input_tensor, num_classes):
        """
        Build the convolutional network.

        Args:
            input_tensor: Input data (e.g., images or feature vectors).
            num_classes: Number of output classes.

        Returns:
            logits: Raw predictions from the network.
        """
        #TODO: Modifiziere build_model, um Zustände und Aktionen beider Agenten als Eingabe zu verwenden
        with tf.variable_scope('convnet', reuse=tf.AUTO_REUSE):
            net = tf.layers.conv2d(input_tensor, 32, (3, 3), activation=tf.nn.relu, name='conv1')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=2, name='pool1')

            net = tf.layers.conv2d(net, 64, (3, 3), activation=tf.nn.relu, name='conv2')
            net = tf.layers.max_pooling2d(net, (2, 2), strides=2, name='pool2')

            net = tf.layers.flatten(net, name='flatten')
            net = tf.layers.dense(net, 128, activation=tf.nn.relu, name='fc1')

            logits = tf.layers.dense(net, num_classes, name='logits')
            self.output = logits

        return logits

    def compute_loss(self, logits, labels):
        """
        Compute the loss for the given predictions and labels.

        Args:
            logits: Predictions from the model.
            labels: Ground truth labels.

        Returns:
            loss: Loss value.
        """
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def compute_accuracy(self, logits, labels):
        """
        Compute the accuracy for the given predictions and labels.

        Args:
            logits: Predictions from the model.
            labels: Ground truth labels.

        Returns:
            accuracy: Accuracy value.
        """
        with tf.name_scope('accuracy'):
            correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return accuracy
    
    def compute_cooperative_reward(self, state_agent1, state_agent2):
        #TODO: Integriere eine Funktion, die kooperative Aktionen belohnt
        distance = tf.norm(state_agent1 - state_agent2)
        reward = tf.exp(-distance)  # Belohnt Nähe
        return reward
# TODO: Ergänze eine Methode, die die Ausgaben des Welt- und Selbst-Modells mit den Aktionen des anderen Agenten abgleicht.
#TODO: Implementiere ein Training, bei dem beide Agenten dieselbe Instanz des Welt-Modells verwenden, aber ihre eigenen Selbst-Modelle optimieren.