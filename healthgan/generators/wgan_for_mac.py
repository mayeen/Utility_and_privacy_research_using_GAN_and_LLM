"""Wasserstein GAN with gradient penalties"""
import os
import sys
import time
import pickle as pkl
import pandas as pd

# Force TensorFlow to use tf-keras instead of Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Disable eager execution to use TF1 style
tf.compat.v1.disable_eager_execution()

def data_batcher(data, batch_size):
    """create yield function for given data and batch size"""

    def get_all_batches():
        """yield function (generator) for all batches in data"""
        # shuffle in place each time
        np.random.shuffle(data)

        # get total number of evenly divisible batchs
        # shape of (num_batches, batch_size, n_features)
        batches = data[:(data.shape[0] // batch_size) * batch_size]
        batches = batches.reshape(-1, batch_size, data.shape[1])

        # go through all batches and yield them
        for i, _ in enumerate(batches):
            yield np.copy(batches[i])

    def infinite_data_batcher():
        """creates a generator that yields new batches every time it is called"""
        # once we run out of batches start over
        while True:
            # for each batch in one set of batches
            for batch in get_all_batches():
                yield batch

    return infinite_data_batcher()

def dense(inputs, units, activation=None, name=None):
    """Custom dense layer using tf.Variable"""
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        input_dim = int(inputs.shape[-1])
        w = tf.compat.v1.get_variable(
            'kernel',
            shape=[input_dim, units],
            initializer=tf.compat.v1.glorot_uniform_initializer()
        )
        b = tf.compat.v1.get_variable(
            'bias',
            shape=[units],
            initializer=tf.compat.v1.zeros_initializer()
        )
        output = tf.matmul(inputs, w) + b
        if activation is not None:
            output = activation(output)
        return output


class WGAN():
    """Wasserstein GAN with gradient penalties"""
    params = {
        'base_nodes': 64,
        'critic_iters': 5,  # number of discriminator iterations
        'lambda': 10,  # paramter for gradient penalty
        'num_epochs': 100000  # how long to train for
    }

    def __init__(self,
                 train_filepath,
                 test_filepath,
                 critic_iters=None,
                 base_nodes=None):
        if critic_iters:
            self.params['critic_iters'] = critic_iters

        scratch = pd.read_csv(train_filepath)
        self.col_names = scratch.columns
        train_data = scratch.values
        self.test_data = pd.read_csv(test_filepath).values

        self.params['n_features'] = train_data.shape[1]
        self.params['1.5_n_features'] = round(1.5 * self.params['n_features'])
        self.params['2_n_features'] = 2 * self.params['n_features']

        if base_nodes:
            self.params['base_nodes'] = base_nodes
        self.params['2_base_nodes'] = 2 * self.params['base_nodes']
        self.params['4_base_nodes'] = 4 * self.params['base_nodes']

        self.params['n_observations'] = train_data.shape[0]
        raw_batch_size = int(train_data.shape[0] / self.params['critic_iters'])
        if raw_batch_size >= 100:
            self.params['batch_size'] = (raw_batch_size // 100) * 100
        else:
            self.params['batch_size'] = raw_batch_size

        assert self.test_data.shape[0] > self.params['batch_size'], "Test data smaller than batch size"

        self.train_batcher = data_batcher(train_data, self.params['batch_size'])
        self.print_settings()

        self.real_data = None
        self.gen_loss = None
        self.disc_loss = None
        self.gen_train_op = None
        self.disc_train_op = None
        self.rand_noise_samples = None

        self.disc_loss_all = []
        self.gen_loss_all = []
        self.disc_loss_test_all = []
        self.time_all = []

    def print_settings(self):
        """print the settings"""
        for k, v in self.params.items():
            print(f'{k + ":":18}{v}')
        print()

    def generator(self, inpt):
        """create the generator graph"""
        # CHANGED: tf.contrib -> tf.compat.v1.layers
        
        # first dense layer
        output = dense(
            inpt,
            self.params['2_n_features'],
            activation=tf.nn.relu,
            name='Generator.1')

        # second dense layer
        output = dense(
            output,
            self.params['1.5_n_features'],
            activation=tf.nn.relu,
            name='Generator.2')

        # third dense layer
        output = dense(
            output,
            self.params['n_features'],
            activation=tf.nn.sigmoid,
            name='Generator.3')

        return output

    def discriminator(self, output):
        """create the discriminator graph"""
        # CHANGED: tf.contrib -> tf.compat.v1.layers
        
        # create first dense layer
        output = dense(
            output,
            self.params['base_nodes'],
            activation=tf.nn.leaky_relu,
            name='Discriminator.1')

        # create second dense layer
        output = dense(
            output,
            self.params['2_base_nodes'],
            activation=tf.nn.leaky_relu,
            name='Discriminator.2')

        # create third dense layer
        output = dense(
            output,
            self.params['4_base_nodes'],
            activation=tf.nn.leaky_relu,
            name='Discriminator.3')

        # create fourth dense layer
        output = dense(
            output,
            1,
            activation=None,
            name='Discriminator.4')

        return output

    def create_graph(self):
        """create computation graph"""
        # create the placeholder for real data and generator for fake
        self.real_data = tf.placeholder(
            tf.float32,
            shape=[self.params['batch_size'], self.params['n_features']],
            name="RealData")
        # create a noise data set of size of the number of samples by 100
        noise = tf.random_normal([self.params['batch_size'], 100])
        fake_data = self.generator(noise)

        # run the discriminator for both types of data
        disc_real = self.discriminator(self.real_data)
        disc_fake = self.discriminator(fake_data)

        # create the loss for generator and discriminator
        self.gen_loss = -tf.reduce_mean(disc_fake)
        self.disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # add the gradient penalty to disc loss
        # create random split of data
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1], minval=0, maxval=1)

        # combine real and fake
        interpolates = (alpha * self.real_data) + ((1 - alpha) * fake_data)
        # compute gradients of dicriminator values
        gradients = tf.gradients(
            self.discriminator(interpolates), [interpolates])[0]
        # calculate the 2 norm of the gradients
        slopes = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # subtract 1, square, use lambda parameter to scale
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        self.disc_loss += self.params['lambda'] * gradient_penalty

        # use adam optimizer on losses
        gen_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Generator' in v.name
        ]
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.gen_loss, var_list=gen_params)
        disc_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Discriminator' in v.name
        ]
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.disc_loss, var_list=disc_params)

        # for generating samples
        rand_noise = tf.random_normal([10000, 100], name="RandomNoise")
        self.rand_noise_samples = self.generator(rand_noise)

        with tf.Session() as session:
            _ = tf.summary.FileWriter('./logs_new', session.graph)

    def train(self):
        """run the training loop"""
        # saver object for saving the model
        saver = tf.train.Saver()

        with tf.Session() as session:
            # initialize variables
            session.run(tf.global_variables_initializer())

            for epoch in range(self.params['num_epochs']):
                start_time = time.time()

                disc_loss_list = []
                for i in range(self.params['critic_iters']):
                    # get a batch
                    train = next(self.train_batcher)
                    # run one critic iteration
                    disc_loss, _ = session.run(
                        [self.disc_loss, self.disc_train_op],
                        feed_dict={self.real_data: train})
                    disc_loss_list.append(disc_loss)

                # run one generator train iteration
                gen_loss, _ = session.run([self.gen_loss, self.gen_train_op])

                # save the loss and time of iteration
                self.time_all.append(time.time() - start_time)
                self.disc_loss_all.append(disc_loss_list)
                self.gen_loss_all.append(gen_loss)

                if epoch < 10 or epoch % 100 == 99:
                    # print the results
                    print((f'Epoch: {epoch:5} '
                           f'[D loss: {self.disc_loss_all[-1][-1]:7.4f}] '
                           f'[G loss: {self.gen_loss_all[-1]:7.4f}] '
                           f'[Time: {self.time_all[-1]:4.2f}]'))

                # if at epoch ending 9999 check test loss
                if epoch == 0 or epoch % 1000 == 999:
                    # shuffle test in place
                    np.random.shuffle(self.test_data)
                    test_disc_loss = session.run(
                        self.disc_loss,
                        feed_dict={
                            self.real_data:
                            self.test_data[:self.params['batch_size']]
                        })
                    self.disc_loss_test_all.append(test_disc_loss)
                    print(
                        f'Test Epoch: [Test D loss: {self.disc_loss_test_all[-1]:7.4f}]'
                    )

                # if at epoch ending 99999 generate large
                # CHANGED: Make directory if it doesn't exist
                if not os.path.exists('data'):
                    os.makedirs('data')

                if epoch == (self.params['num_epochs'] - 1):
                    for i in range(10):
                        samples = session.run(self.rand_noise_samples)
                        samples = pd.DataFrame(samples, columns=self.col_names)
                        samples.to_csv(
                            f'data/samples_{epoch}_{self.params["critic_iters"]}_{self.params["base_nodes"]}_synthetic_{i}.csv',
                            index=False)

                # update log every 100
                if epoch < 5 or epoch % 100 == 99:
                    with open(
                            f'log_{self.params["critic_iters"]}_{self.params["base_nodes"]}.pkl',
                            'wb') as f:
                        pkl.dump({
                            'time': self.time_all,
                            'disc_loss': self.disc_loss_all,
                            'gen_loss': self.gen_loss_all,
                            'test_loss': self.disc_loss_test_all
                        }, f)

            saver.save(
                session,
                os.path.join(
                    os.getcwd(),
                    f'model_{self.params["critic_iters"]}_{self.params["base_nodes"]}.ckpt'
                ))

            tf.io.write_graph(session.graph, '.', 'wgan_graph.pbtxt')


if __name__ == '__main__':
    # parse arguments

    # start with a fresh graph
    tf.reset_default_graph()
    base = "/Users/kazimostafashahriar/Main Drive/thesis/Medical data analysis/Code/thesis/data"
    # create object
    wgan = WGAN(
        # CHANGE THESE to the filename you generated with sdv_converter.py
        train_filepath=f"{base}/diabetic_data_subset_sdv.csv", # Example filename
        test_filepath=f"{base}/diabetic_data_subset_sdv.csv",  # Example filename
        critic_iters=int(sys.argv[1]) if len(sys.argv) > 1 else 5,
        base_nodes=int(sys.argv[2]) if len(sys.argv) > 2 else 64)
    # define the computation graph
    wgan.create_graph()
    # train the model
    wgan.train()