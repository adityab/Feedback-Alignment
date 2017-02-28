import math
import numpy as np
import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def fc(input_, output_size, fn=None, stddev=0.02, bias_start=0.0, name=None):
  """Fully-Connected Layer
  """
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    # Define Weights and Biases in layer scope
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    act = tf.matmul(input_, matrix) + bias

    # Apply activation function fn
    if fn is not None:
      out = fn(act)
    else:
      out = act

    return out
