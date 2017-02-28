import sys
import tensorflow as tf

from tensorflow import summary
from tensorflow.examples.tutorials.mnist import input_data

from ops import *
from utils import *

FLAGS = None

class Model_MNIST(object):
  def __init__(self, session):
    self.session = session

    self.x_dim = 784
    self.y_dim = 10
    
    self.build_model()

  def forward(self, x):
    with tf.variable_scope('forward') as scope:
      h0 = fc(x, 100, fn=lrelu, name='fc1')
      h1 = fc(h0, 100, fn=lrelu, name='fc2')
      h2 = fc(h1, self.y_dim, fn=None, name='fc3')

      return h2

  def build_model(self):
    # Placeholders for inputs and targets
    self.inputs = tf.placeholder(tf.float32, [None, 784])
    self.labels = tf.placeholder(tf.float32, [None, 10])

    # Create network
    preds = self.forward(self.inputs)

    # Define loss function
    self.loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=preds))

  def train(self, FLAGS):
    # Set up optimizer with fallback
    train_step = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr).minimize(self.loss)
    tf.global_variables_initializer().run()

    # Set up logging
    loss_summary = summary.scalar('loss', self.loss)
    writer = summary.FileWriter(FLAGS.output_dir + '/logs', self.session.graph)

    # Load MNIST dataset
    dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    n_batches = int(dataset.train.num_examples / FLAGS.bs)

    # Learn
    for epoch in range(FLAGS.epochs):
      train_loss = None
      val_loss = None

      # Iterate over batches
      for i in range(n_batches):
        inputs, labels = dataset.train.next_batch(FLAGS.bs)
        _, loss = self.session.run([train_step, self.loss],
          feed_dict={
            self.inputs: inputs,
            self.labels: labels 
          })
        train_loss = loss

        # Log validation loss at every 100th minibatch or end of epoch
        if (i % 150 == 0) or (i == n_batches - 1):
          _, summary_str, val_loss = self.session.run([train_step, loss_summary, self.loss],
          feed_dict={
            self.inputs: dataset.validation.images,
            self.labels: dataset.validation.labels
          })

          # Log progress
          writer.add_summary(summary_str, epoch * n_batches + i)
          frac = (i + 1)/ n_batches
          sys.stdout.write('\r')
          sys.stdout.write((\
            col(CYAN, 'Epoch (%d/%d):') + col(BOLD, '\t[%-10s] %d%% \t') + \
            col(YELLOW, 'Train Loss:') + col(None, ' %.8f ') + '\t' + \
            col(YELLOW, 'Val Loss:') + col(None, ' %.8f')) % \
            (epoch + 1, FLAGS.epochs, '='*int(frac*10), int(frac*100), train_loss, val_loss))
          sys.stdout.flush()

      sys.stdout.write('\n')
