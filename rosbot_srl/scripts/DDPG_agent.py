#!/usr/bin/env python
import rospy
import tensorflow as tf
import math
import numpy as np
import os
import numpy.random as nr
from openai_ros.task_envs.husarion import husarion_get_to_position_turtlebot_playground

# Hide some depreacation warnings and disable eager execution
tf.logging.set_verbosity(tf.logging.ERROR)
# specify the relative position
parentDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(parentDirectory, 'training_results')


class ReplayBuffer:
    CER = False

    # TODO: Extend with possibility of restoring sequences
    def __init__(self, obs_dim, act_dim, size, ):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs1_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.obs2_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)
        self.num_experiences = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # replace the oldest entry from memory
        self.size = min(self.size + 1, self.max_size)
        self.num_experiences += 1

    def count(self):
        # return buffer size
        return self.num_experiences

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # this takes the last added sample, unless ptr = 0, then it takes sample 1, this then does violate CER
        if self.CER: idxs[-1] = abs(self.ptr - 1)

        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size)


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.05):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state


class DDPGAgent():
    # Class variables: variables shared across all instances of this class
    pi_lr = 1e-3  # the learning rate for the actor network
    q_lr = 1e-4  # the learning rate for the critic network
    tau = 1e-3
    batch_size = 64  # batch size for training the agent
    gamma = 0.99  # discount rate
    hidden_variables = 512  # number of neurons in the hidden layers
    buffer_size = 1e6  # replay buffer size
    momentum = 0.9

    def __init__(self, obs_dim, act_dim, continuelearning=False):
        folder = "training_results"
        filename = 'ddpg_agent'

        self.model_path = os.path.join(path, "saved_model", filename + ".ckpt")
        self.act_dim = act_dim
        self.graph = tf.Graph()
        self.counter = 0

        with self.graph.as_default():

            # INPUTS TO THE COMPUTATIONAL GRAPH
            self.x_ph = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='state')
            self.a_ph = tf.placeholder(shape=[None, act_dim], dtype=tf.float32, name='action')
            self.x2_ph = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='next_state')
            self.r_ph = tf.placeholder(shape=[None], dtype=tf.float32, name='reward')
            self.d_ph = tf.placeholder(shape=[None], dtype=tf.float32, name='is_done')
            self.is_training = tf.placeholder(tf.bool, shape=[], name="train_cond")

            # DEFINE MAIN ACTOR-CRITIC NETWORKS ------------------------------------------------------------------
            with tf.variable_scope('main'):
                self.pi, self.q = self.actor_critic_mlp(self.x_ph, self.a_ph)
            # Define EMA update
            # Get the weights of the separate layers within the actor network
            self.weights1 = self.get_weights("main/Actor/weights_layer1")
            self.weights2 = self.get_weights("main/Actor/weights_layer2")
            self.weights3 = self.get_weights("main/Actor/weights_layer3")
            self.weights4 = self.get_weights("main/Actor/weights_layer4")
            self.weights5 = self.get_weights("main/Actor/weights_layer5")
            # Reconstruct the whole neural network weights excluding BLN layer and excluding bias
            self.actor_net = [self.weights1, self.weights2, self.weights3, self.weights4, self.weights5]
            # print(self.actor_net)
            # Get the weights of the separate layers within the critic network
            self.weights1 = self.get_weights("main/Critic/weights_layer1")
            self.weights2 = self.get_weights("main/Critic/weights_layer2")
            self.weights3 = self.get_weights("main/Critic/weights_layer3")
            self.weights4 = self.get_weights("main/Critic/weights_layer4")
            self.weights5 = self.get_weights("main/Critic/weights_layer5")
            # Reconstruct the whole neural network weights excluding BLN layer and excluding bias
            self.critic_net = [self.weights1, self.weights2, self.weights3, self.weights4, self.weights5]
            # print(self.critic_net)
            # Define ema with decay
            ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
            # Define the target update
            self.target_update = [ema.apply(self.actor_net), ema.apply(self.critic_net)]
            # Calculate the moving average over the weights of the main actor and main critic network
            self.weights_target_actor_net = [ema.average(x) for x in self.actor_net]
            self.weights_target_critic_net = [ema.average(x) for x in self.critic_net]

            # DEFINE THE TARGET NETWORK --------------------------------------------------------------------------
            with tf.variable_scope('target'):
                self.pi_targ, self.q_targ = self.actor_critic_mlp_target(self.x_ph, self.a_ph)

            self.replay_buffer = ReplayBuffer(obs_dim, act_dim, self.buffer_size)
            # DEFINE LOSS FUNCTIONS -------------------------------------------------------------------------------
            # This has to be here to update the parameters of the batch normalization layers
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                backup = tf.stop_gradient(self.r_ph + self.gamma * self.q_targ)
                self.pi_loss = -tf.reduce_mean(self.q)
                weight_decay = tf.add_n([0.01 * tf.nn.l2_loss(var, name='L2-norm') for var in self.critic_net])
                self.q_loss = tf.reduce_mean((self.q - backup) ** 2, name='MSE-Q') + weight_decay

                # TRAINING FUNCTIONS ----------------------------------------------------------------------------------
                self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr, beta1=0.9, beta2=0.999,
                                                           epsilon=1e-08)  # epsilon=1e-08)
                self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr, beta1=0.9, beta2=0.999,
                                                          epsilon=1e-08)  # epsilon=1e-08)

                self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=self.get_vars('main/Actor'))
                self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=self.get_vars('main/Critic'))

            # Init session ---------------------------------------------------------------------------------------------
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(path + 'logs/', self.sess.graph)

        # Restore session if desired
        if continuelearning:
            print('Loading RL model from memory')
            self.load_model()
        else:
            print("Starting RL model from scratch!")
            self.sess.run(self.init)

    def act(self, state):
        """ returns the action value, resulting from policy pi + noise """
        a = self.sess.run(self.pi, feed_dict={self.x_ph: state, self.is_training: False})[0]
        self.counter += 1

        return a

    def learn(self, episode):
        """ trains the actor and critic networks. Samples minibatch from replay memory """
        batch = self.replay_buffer.sample_batch(self.batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     self.is_training: True
                     }
        # Q-learning update
        outs_q = self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict)

        # Policy update
        outs_pi = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
        return outs_q[0], outs_pi[0]

    def remember(self, state, action, reward, next_state, done):

        """ simply adds a experience tuple to the replay memory """
        self.replay_buffer.store(state, action, reward, next_state, done)

    def save_model(self, episode):

        """ saves all variables in the session to memory """
        print("RL Network storing Model....")
        # make sure the folder exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # save data
        try:
            self.saver.save(self.sess, self.model_path + '_' + str(episode) + '_')
        except:
            raise Exception('Error saving Network')

    def load_model(self):

        """ loads all variables in the session from memory """
        print("RL Network restoring Model.....")
        self.saver.restore(self.sess, self.model_path)

    def get_vars(self, scope):
        return [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if scope in x.name]

    def get_weights(self, NAME):
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, NAME)]

        for v in weights:
            unlisted_weights = v
        return unlisted_weights

    def actor_critic_mlp(self, x, a, ):
        LAYER_SIZE = 512
        state_dim = 9
        action_dim = 2
        with tf.variable_scope('Actor'):
            W1 = tf.Variable(
                tf.random_uniform([state_dim, LAYER_SIZE], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)),
                name='weights_layer1')
            W2 = tf.Variable(
                tf.random_uniform([LAYER_SIZE, LAYER_SIZE], -1 / math.sqrt(LAYER_SIZE), 1 / math.sqrt(LAYER_SIZE)),
                name='weights_layer2')
            W3 = tf.Variable(
                tf.random_uniform([LAYER_SIZE, LAYER_SIZE], -1 / math.sqrt(LAYER_SIZE), 1 / math.sqrt(LAYER_SIZE)),
                name='weights_layer3')
            W4_linear = tf.Variable(tf.random_uniform([LAYER_SIZE, 1], -3e-3, 3e-3), name='weights_layer4')
            W4_angular = tf.Variable(tf.random_uniform([LAYER_SIZE, 1], -3e-3, 3e-3), name='weights_layer5')

            pi = tf.contrib.layers.batch_norm(x, activation_fn=tf.identity, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN1', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, W1, name='dense1')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN2', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, W2, name='dense2')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN3', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, W3, name='dense3')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN4', decay=0.9, epsilon=1e-5)

            linear_action_output = tf.sigmoid(tf.matmul(pi, W4_linear, name='dense4_linear'))
            angular_action_output = tf.tanh(tf.matmul(pi, W4_angular, name='dense5_angular'))
            pi = tf.concat([linear_action_output, angular_action_output], axis=-1)

        with tf.variable_scope('Critic'):
            W1 = tf.Variable(
                tf.random_uniform([state_dim, LAYER_SIZE], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)),
                name='weights_layer1')
            W2 = tf.Variable(tf.random_uniform([LAYER_SIZE, LAYER_SIZE], -1 / math.sqrt(LAYER_SIZE + action_dim),
                                               1 / math.sqrt(LAYER_SIZE + action_dim)), name='weights_layer2')
            W2_action = tf.Variable(tf.random_uniform([action_dim, LAYER_SIZE], -1 / math.sqrt(LAYER_SIZE + action_dim),
                                                      1 / math.sqrt(LAYER_SIZE + action_dim)), name='weights_layer3')
            W3 = tf.Variable(
                tf.random_uniform([LAYER_SIZE, LAYER_SIZE], -1 / math.sqrt(LAYER_SIZE), 1 / math.sqrt(LAYER_SIZE)),
                name='weights_layer4')
            W4 = tf.Variable(tf.random_uniform([LAYER_SIZE, 1], -3e-3, 3e-3), name='weights_layer5')

            q = tf.nn.relu(tf.matmul(x, W1, name='dense1'))
            q = tf.nn.relu(tf.matmul(q, W2, name='dense2') + tf.matmul(pi, W2_action, name='dense3'))
            q = tf.nn.relu(tf.matmul(q, W3, name='dense4'))
            q = tf.identity(tf.matmul(q, W4, name='dense5'))

        return pi, q

    def actor_critic_mlp_target(self, x, a, ):

        with tf.variable_scope('Actor'):
            pi = tf.contrib.layers.batch_norm(x, activation_fn=tf.identity, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN1', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, self.weights_target_actor_net[0], name='dense1')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN2', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, self.weights_target_actor_net[1], name='dense2')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN3', decay=0.9, epsilon=1e-5)
            pi = tf.matmul(pi, self.weights_target_actor_net[2], name='dense3')
            pi = tf.contrib.layers.batch_norm(pi, activation_fn=tf.nn.relu, center=True, scale=True,
                                              updates_collections=None, is_training=self.is_training,
                                              scope='BLN4', decay=0.9, epsilon=1e-5)

            linear_action_output = tf.sigmoid(tf.matmul(pi, self.weights_target_actor_net[3], name='dense4_linear'))
            angular_action_output = tf.tanh(tf.matmul(pi, self.weights_target_actor_net[4], name='dense5_angular'))
            pi = tf.concat([linear_action_output, angular_action_output], axis=-1)

        with tf.variable_scope('Critic'):
            q = tf.nn.relu(tf.matmul(x, self.weights_target_critic_net[0], name='dense1'))
            q = tf.nn.relu(tf.matmul(q, self.weights_target_critic_net[1], name='dense2') +
                           tf.matmul(pi, self.weights_target_critic_net[2], name='dense3'))
            q = tf.nn.relu(tf.matmul(q, self.weights_target_critic_net[3], name='dense4'))
            q = tf.identity(tf.matmul(q, self.weights_target_critic_net[4], name='dense5'))

        return pi, q
