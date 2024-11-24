import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers

def mnist_model(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0)
    )(x)
    x = layers.Dense(
        32,
        activation='relu',
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0)
    )(x)
    outputs = layers.Dense(
        10,
        activation=None,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0)
    )(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def alexnet_model(input_shape=(227, 227, 3), num_classes=1000, train=True, norm=True):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        96, 11, strides=4, padding='valid', activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer=initializers.Constant(0)
    )(inputs)
    if norm:
        x = layers.Lambda(lambda x: tf.nn.lrn(x, depth_radius=5, bias=1, alpha=1e-4, beta=0.75))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(
        256, 5, padding='same', activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer=initializers.Constant(0)
    )(x)
    if norm:
        x = layers.Lambda(lambda x: tf.nn.lrn(x, depth_radius=5, bias=1, alpha=1e-4, beta=0.75))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(
        384, 3, padding='same', activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer=initializers.Constant(0)
    )(x)
    x = layers.Conv2D(
        256, 3, padding='same', activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer=initializers.Constant(0)
    )(x)
    x = layers.Conv2D(
        256, 3, padding='same', activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer=initializers.Constant(0)
    )(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        4096, activation='relu',
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1)
    )(x)
    if train:
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        4096, activation='relu',
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1)
    )(x)
    if train:
        x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(
        num_classes, activation=None,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0)
    )(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def alexnet_3dworld_model(input_shape=(227, 227, 3), num_classes=55, train=True, norm=True):
    return alexnet_model(input_shape, num_classes, train, norm)
