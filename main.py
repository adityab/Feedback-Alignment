import sys
import argparse
import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from mnist import Model_MNIST

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 25, 'Epochs to train for [25]')
flags.DEFINE_float('lr', 0.02, 'Learning rate [0.02]')
flags.DEFINE_integer('bs', 128, 'Batch Size [128]')
flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', 'Directory for storing dataset')
flags.DEFINE_string('output_dir', '.', 'Directory to store generated logs, checkpoints, and artifacts [.]')
flags.DEFINE_string('flow', 'BP', 'Gradient flow scheme (autodiff, BP, FA, DFA) [BP]')

def main(_):
  pp = pprint.PrettyPrinter()
  pp.pprint(flags.FLAGS.__flags)

  with tf.Session() as session:
    model = Model_MNIST(session)
    model.train(FLAGS)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  _, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
