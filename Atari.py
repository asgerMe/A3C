

import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import scipy.signal

import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import re
from time import sleep
from time import time
from scipy.misc import imresize

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def process_frame(frame):
    luminance = np.sum(frame, 2).astype('float32')
    luminance = luminance[27:, :]
    luminance = imresize(luminance, (84, 84)).astype('float32')
    luminance = imresize(luminance, (42, 42)).astype('float32')*(1 / 255.0)
    luminance = np.expand_dims(luminance, axis=3)

    return luminance


class A3C_Network:

    def normalized_columns_initializer(self , std = 1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = (np.random.randn(*shape).astype(np.float32))
            out *= std / np.sum(np.sqrt(np.square(out)))
            return tf.constant(out)

        return _initializer

    def AIconv2d(self, x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
        with tf.variable_scope(name):

            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

            fan_in = np.prod(filter_shape[:3])

            fan_out = np.prod(filter_shape[:2]) * num_filters
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                                collections=collections)
            return tf.nn.conv2d(x, w, stride_shape, pad) + b

    def __init__(self, pix_x , pix_y, scope, trainer, act_space= 6):

        with tf.variable_scope(scope):

            strides1 = int(4)
            strides2 = int(2)

            full_c1 = int(pix_x/(strides1*strides2))
            full_c2 = int(pix_y/(strides1*strides2))

            filters2 = 32

            self.input = tf.placeholder(dtype=tf.float32, shape=(None, 42 , 42, 1), name='frame_input')

            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.input, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='SAME')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='SAME')
            self.conv3= slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv2, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='SAME')
            self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv3, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='SAME')
            self.hidden = slim.fully_connected(slim.flatten(self.conv4), 256, activation_fn=tf.nn.elu)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple = True)

            init_cell_state = tf.constant(value = 0, shape = (1, lstm_cell.state_size.c), dtype =  tf.float32)
            init_hidden_state = tf.constant(value = 0, shape = (1, lstm_cell.state_size.h), dtype = tf.float32)

            self.init_cell = [init_cell_state, init_hidden_state]

            self.c_in = tf.placeholder(dtype = tf.float32,shape =  (1, lstm_cell.state_size.c), name = 'c_in')
            self.h_in = tf.placeholder(dtype = tf.float32, shape = (1, lstm_cell.state_size.h), name = 'h_in')

            self.rnn_in = tf.expand_dims(self.hidden, [0])
            step_size = tf.shape(self.input)[:1]
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_in, self.h_in)
            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
               lstm_cell, self.rnn_in, initial_state=self.state_in,
               time_major=False, sequence_length=step_size)

            self.lstm_outputs = tf.squeeze(self.lstm_outputs, axis = 0)


            condense_to_value = tf.get_variable(dtype=tf.float32, shape = (256), initializer=self.normalized_columns_initializer(std= 1),
                                                name='form_value')


            self.value_output =tf.tensordot(self.lstm_outputs, condense_to_value, [[1], [0]])
            condense_to_actions = tf.get_variable(dtype=tf.float32,
                                                  shape = (256, act_space), initializer=self.normalized_columns_initializer(std= 0.01), name='c_act')
            action_output = tf.tensordot(self.lstm_outputs, condense_to_actions, [[1], [0]], name = 'action1')
            self.norm_actions = tf.nn.softmax(action_output)

            self.test = self.lstm_outputs

            if scope != 'global':
                R = tf.placeholder(dtype=tf.float32, shape=(None), name='perf_reward')
                get_value = tf.placeholder(dtype=tf.float32, shape=(None), name='perf_value')
                get_action = tf.placeholder(dtype=tf.int32, shape=(None), name='perf_action')
                advantage = tf.placeholder(dtype = tf.float32 ,shape = (None) , name = 'advantage')

                self.one_hot_action = tf.one_hot(get_action,act_space)
                self.action_channel1 = tf.multiply(self.norm_actions, self.one_hot_action)
                self.action_channel = tf.reduce_sum(self.action_channel1,1)

                self.clip_action = tf.clip_by_value(self.action_channel,0.000001,9999999)
                self.value_loss = tf.reduce_sum(tf.square(self.value_output-R))
                self.action_loss = tf.reduce_sum(tf.log(self.clip_action)*advantage)
                self.entropy = -tf.reduce_sum(tf.log(self.norm_actions)*self.norm_actions)
                self.full_loss = 0.5*self.value_loss - self.action_loss - 0.01*self.entropy

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.full_loss , local_vars)



                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))

    def create_network(self):
        init = tf.initialize_all_variables()
        return [init, tf.shape(self.norm_actions)]

class Worker(A3C_Network):



    def __init__(self, the_scope, trainer):
        A3C_Network.__init__(self, 42, 42, the_scope, trainer, act_space=3)
        self.gamma = 0.9900
        self.update_local_ops = self.copy_scope('global', the_scope)
        self.the_scope = the_scope
        self.max_episodes = 300
        self.summary_writer = tf.summary.FileWriter("tensorBoard/train_" + the_scope)


    def get_scope(self):
        return self.the_scope

    def copy_scope(self,from_scope,to_scope):
        source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for source_vars, target_vars in zip(source_vars,target_vars):
            op_holder.append(tf.assign(target_vars,source_vars))
        return op_holder

    def discounted_reward(self , reward_buffer, bsv, clip = True):
        discount_vector = np.ones([1,len(reward_buffer)])*self.gamma

        discount_vector[0,0] = 1
        discount_vector = np.cumproduct(discount_vector)

        L = len(reward_buffer)

        discounted_reward_buffer = []
        for R in enumerate(reward_buffer[:]):
            discounted_reward_buffer.append(bsv*np.power(self.gamma,( L - R[0])-1) + np.sum((discount_vector[0:(L - R[0])]*reward_buffer[R[0]:])))
        if clip == True:
            return discounted_reward_buffer[:-1]
        else:
            return discounted_reward_buffer
    def train_worker(self, discounted_reward_buffer,reward_buffer, state_buffer , action_buffer, value_buffer,rnn_state_buffer_h, rnn_state_buffer_c,  sess):

        advantage = np.asarray(self.reward_buffer[:-1]) + self.gamma * np.asarray(
            self.value_buffer[1:]) - np.asarray(self.value_buffer [:-1])
        advantage = self.discounted_reward(advantage, 0, clip = False)


        feed_dict = {self.the_scope + '/frame_input:0': state_buffer[:-1],
                     self.the_scope + '/c_in:0': rnn_state_buffer_c,
                     self.the_scope + '/h_in:0': rnn_state_buffer_h,
                     self.the_scope + '/advantage:0': advantage,
                     self.the_scope + '/perf_action:0': action_buffer[:-1],
                     self.the_scope + '/perf_reward:0': discounted_reward_buffer,
                     self.the_scope + '/perf_value:0': value_buffer[:-1]}

        sess.run([self.value_output, self.full_loss, self.value_loss,self.action_loss,self.entropy] , feed_dict = feed_dict)
        sess.run(self.apply_grads, feed_dict=feed_dict)
        #print(np.shape(sess.run(self.test, feed_dict=feed_dict)))
        #print(sess.run(self.test, feed_dict=feed_dict))

    def run_atari_episode(self , max_steps, episodes , sess ,coord,  saver):

        env = gym.make('PongDeterministic-v0')

        track_loss = 0
        esp = 0
        with sess.as_default() and sess.graph.as_default():

            while not coord.should_stop():
                if esp % 20 == 0 and self.the_scope == 'local_0':
                    saver.save(sess,"/media/asger/ce7a4008-6b8f-447d-9acc-614945aef109/space_invaders_global.ckpt")
                    print('saving episode ', esp)
                esp += 1
                action = 0
                done = 0
                obs_prev = env.reset()
                obs_prev = process_frame(obs_prev)
                rnn_state = sess.run(self.init_cell)

                self.state_buffer = []
                self.reward_buffer = []
                self.actions_buffer = []
                self.value_buffer = []


                while done == 0:

                    self.rnn_state_buffer_c = rnn_state[0]
                    self.rnn_state_buffer_h = rnn_state[1]

                    for  i in range(max_steps):

                        obs_prev = process_frame(obs_prev)
                        feed_dict = {self.the_scope + '/frame_input:0': [obs_prev], self.the_scope + '/c_in:0': rnn_state[0],
                                          self.the_scope + '/h_in:0': rnn_state[1]}

                        norm_actions, value_est, new_rnn_state = sess.run([self.norm_actions, self.value_output, self.lstm_state], feed_dict=feed_dict)

                        if i != max_steps - 1:
                            rnn_state = new_rnn_state

                        norm_actions = norm_actions[0]
                        a = np.random.choice(norm_actions, p=norm_actions)
                        chosen_action = np.argmax(norm_actions == a)

                        obs_next , reward , done , info = env.step(1+chosen_action)

                        if self.the_scope == 'local_0':
                            env.render()
                            print(chosen_action)
                            print(norm_actions)
                            print(value_est[0])


                        self.state_buffer.append(obs_prev)
                        self.actions_buffer.append(chosen_action)
                        self.value_buffer.append(value_est[0])
                        self.reward_buffer.append(reward)


                        if i == max_steps-1 or done == 1:

                            boot_strap_val = self.value_buffer[len(self.value_buffer)-1]
                            discounted_reward_buffer = self.discounted_reward(self.reward_buffer, boot_strap_val)
                            self.train_worker(discounted_reward_buffer, self.reward_buffer, self.state_buffer, self.actions_buffer, self.value_buffer,self.rnn_state_buffer_h,self.rnn_state_buffer_c, sess)
                            sess.run(self.update_local_ops)
                            if done == 1:
                                break

                            self.state_buffer = [self.state_buffer[len(self.state_buffer) - 1]]
                            self.reward_buffer = [self.reward_buffer[len(self.reward_buffer) - 1]]
                            self.actions_buffer = [self.actions_buffer[len(self.actions_buffer) - 1]]
                            self.value_buffer = [self.value_buffer[len(self.value_buffer) - 1]]
                        obs_prev = obs_next



class DeployModel:

    def display_processed_frame(self):
        env = gym.make('Pong-v0')
        get_frame = env.reset()
        for i in range(46):
            act = env.action_space.sample()
            get_frame,_,_,_ = env.step(act)
        env.render()
        processed_frame = process_frame(get_frame)
        processed_frame = np.tile(processed_frame,3)
        plt.imshow(processed_frame)
        plt.show()

        return processed_frame

    def load(self, episodes, path):

        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(path + '.meta')
        with tf.Session() as sess:

            graph = tf.get_default_graph()
            saver.restore(sess,path)
            print('model restored')
            env = gym.make('SpaceInvaders-v0')
            esp = 0
            print_tensors_in_checkpoint_file(file_name=path , tensor_name='', all_tensors=False, all_tensor_names=True)

            placeholder = graph.get_tensor_by_name("local_0/frame_input:0")
            h_in = graph.get_tensor_by_name("local_0/h_in:0")
            c_in = graph.get_tensor_by_name("local_0/c_in:0")

            init_cell_state = np.asarray(np.zeros([1,256]), dtype=np.float32)
            init_hidden_state = np.asarray(np.zeros([1,256]), dtype=np.float32)

            op_to_restore = graph.get_tensor_by_name("local_0/action2:0")
            while esp < episodes:
                obs = env.reset()
                done = 0
                esp += 1
                while done == 0:
                    obs = process_frame(obs)
                    feed_dict = {placeholder:[obs], h_in:init_hidden_state, c_in: init_cell_state}
                    norm_actions = sess.run(op_to_restore, feed_dict = feed_dict)[0]
                    alt = np.random.choice(norm_actions, p=norm_actions)
                    chosen_action = np.argmax(norm_actions == alt)
                    #print(chosen_action)
                    act = env.action_space.sample()
                    obs, reward, done, info = env.step(act)
                    env.render()
                    sleep(1.0/24)
            sess.close()




    def train(self,threads, episodes, max_steps):
        tf.reset_default_graph()
        with tf.device('/cpu:0'):

            global_network = Worker('global',tf.train.RMSPropOptimizer(learning_rate=0))
            cores = multiprocessing.cpu_count()
            Workers = []
            if threads > cores:
                threads = cores



            for q in range(threads):
                learningrate = 0.0003
                print(learningrate)
                a_worker = Worker('local_' + str(q), tf.train.RMSPropOptimizer(learning_rate=learningrate, epsilon=0.1, decay = 0.99))
                Workers.append(a_worker)
                print('deploy_worker' + str(q))


        saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'local_0'))
        with tf.Session() as sess1:
            global buffer
            sess1.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            worker_threads = []
            for w in Workers:
                worker_work = lambda: w.run_atari_episode(max_steps, episodes, sess1,coord, saver)
                t = threading.Thread(target=worker_work)
                t.start()
                sleep(5)
                worker_threads.append(t)


            coord.join(worker_threads)
            sess1.close()
            print('training is complete')


path = '/media/asger/ce7a4008-6b8f-447d-9acc-614945aef109/space_invaders_global.ckpt'

instance = DeployModel()
#instance.load(10,path)
instance.train(100, 100000,5)




