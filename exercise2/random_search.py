import logging

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import argparse

from cnn_mnist import mnist, neural_network_model

logging.basicConfig(level=logging.WARNING)

#import hpbandster.visualization as hpvis
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.
        rgs:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        num_epochs = budget
        filter_size = config['filter_size']

        n_samples = self.x_train.shape[0]
        n_batches = n_samples // batch_size
        #Again define the session
        #sess = tf.InteractiveSession()
        # Placeholder for x and y
        x = tf.placeholder('float', [None, 784])
        y = tf.placeholder('float')
        
        # call the model and produce prediction

        prediction = neural_network_model(x, nf= num_filters, fs= filter_size)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
        optimizer =  tf.train.GradientDescentOptimizer(lr).minimize(cost)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                for _ in range(int(len(self.x_train)/ batch_size)):
                    rand_index = np.random.choice(len(self.x_train), batch_size)
                    batch_x = self.x_train[rand_index].reshape([-1,784])
                    batch_y = self.y_train[rand_index]
                    sess.run(optimizer, feed_dict = {x: batch_x , y: batch_y})

                # Calculate training accuracy and loss per epoch
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch + 1) + " / " + str(num_epochs) + ": Epoch Training Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))

                
                val_loss = 0
                for val_step in range(int(len(self.x_valid)/ batch_size)):
                    val_batch_x = self.x_valid[val_step*batch_size: ((val_step+1)*batch_size)].reshape([-1,784])
                    val_batch_y = self.y_valid[val_step*batch_size: ((val_step+1)*batch_size)]
                    step_acc, step_loss = sess.run([accuracy, cost], feed_dict={x: val_batch_x, y: val_batch_y})
                    val_loss += step_loss
                
                val_loss = val_loss / (val_step+1)
                print("Validation loss: %s" % val_loss)
            sess.close()
            info = ''
            return ({
                'loss': val_loss,  # this is the a mandatory field to run hyperband
                'info': info # can be used for any user-defined information - also mandatory
            })

    @staticmethod
    def get_configspace():
        
        # create config_space object
        config_space = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1e-1, default_value='1e-2', log=True)
        
        # Create the hyperparameters and add them to the object config_space

        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=64, log=True)

        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=16, log=True)
        filter_size = CSH.CategoricalHyperparameter('filter_size', ['3', '4', '5'])

        config_space.add_hyperparameters([learning_rate, batch_size, num_filters, filter_size])

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,


#hpvis.losses_over_time(all_runs)


plt.savefig("random_search.png")

# TODO: retrain the best configuration (called incumbent) and compute the test error