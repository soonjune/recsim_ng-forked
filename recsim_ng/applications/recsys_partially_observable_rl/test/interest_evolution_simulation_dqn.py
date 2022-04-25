# coding=utf-8
# Copyright 2021 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Line as: python3
"""WIP: For testing differentiable interest evolution networks."""

import re
from typing import Any, Callable, Collection, Sequence, Text, Optional

from numpy import gradient

from recsim_ng.core import network as network_lib
from recsim_ng.core import variable
# from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
import datetime

Network = network_lib.Network
Variable = variable.Variable

tf.config.run_functions_eagerly(True) # for debugging

def reset_replay_buffer(history_length, global_batch, batch_size=1, max_length=1000000):
  data_spec =  (
          tf.TensorSpec([global_batch, 1], tf.int32, 'action'),
          (
              tf.TensorSpec([global_batch, history_length], tf.float32, 'ctime_history'),
              tf.TensorSpec([global_batch, history_length], tf.int32, 'docid_history')
          ),
          (
              tf.TensorSpec([global_batch, history_length], tf.float32, 'next_ctime_history'),
              tf.TensorSpec([global_batch, history_length], tf.int32, 'next_docid_history')
          ),
          tf.TensorSpec([global_batch], tf.float32, 'reward')
  )
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec,
    batch_size=batch_size,
    max_length=max_length)
  return replay_buffer




def reset_optimizer(learning_rate, optimizer_name=None):
  if optimizer_name == 'adam':
    return tf.keras.optimizers.Adam(learning_rate)
  elif optimizer_name == 'sgd':
    return tf.keras.optimizers.SGD(learning_rate)
  else:
    print("learning rate is 0.00025 for RMSProp")
    return tf.keras.optimizers.RMSprop(
      learning_rate=0.00025,
      rho=0.95,
      momentum=0.0,
      epsilon=0.00001,
      centered=True,
    )

def update_target_network(dqn, target_dqn, TAU):
    phi = dqn.get_weights()
    target_phi = target_dqn.get_weights()
    for i in range(len(phi)):
        target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
    target_dqn.set_weights(target_phi)


def distributed_train_step(
    tf_runtime,
    episode_length,
    global_batch,
    trainable_variables,
    metric_to_optimize='reward',
    optimizer = None,
    rec=None,
    replay_buffer=None,
    train_info=None,
):
  """Extracts gradient update and training variables for updating network."""
  grads, loss = None, None


  dataset = replay_buffer.as_dataset(sample_batch_size=train_info['batch_size'])
  iterator = iter(dataset)
  cum_reward = 0.0
  for i in range(episode_length):
    last_state = tf_runtime.execute(0)
    ctime_history = last_state['recommender state'].get('ctime_history').get('state')
    docid_history = last_state['recommender state'].get('doc_history').get('state')
    action = last_state['slate docs'].get('slate_ids')
    last_state = tf_runtime.execute(1)
    train_info['timestep'] += 1
    last_metric_value = last_state['metrics state'].get(metric_to_optimize)
    cum_reward += last_metric_value

    next_ctime_history = last_state['recommender state'].get('ctime_history').get('state')
    next_docid_history = last_state['recommender state'].get('doc_history').get('state')
    values = (action, (ctime_history, docid_history), (next_ctime_history, next_docid_history), last_metric_value)
    values_batched = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), values)
    replay_buffer.add_batch(values_batched)

    if (train_info['timestep'] > train_info['start_update']) and (train_info['timestep'] % train_info['update_period'] == 0):
      with tf.GradientTape() as tape:
        print(f"{train_info['timestep']}  start training")
        (b_action, (b_ctime_history, b_docid_history), \
          (b_next_ctime_history, b_next_docid_history), b_reward), unused_info = next(iterator)
        target_qs = rec._target(b_next_docid_history, b_next_ctime_history, batch_size=train_info['batch_size'])
        max_q = tf.math.reduce_max(target_qs, axis=-1, keepdims=True)
        target = tf.reshape(b_reward, (train_info['batch_size'], global_batch, -1)) + train_info['gamma'] * max_q
        
        qs = tf.reshape(rec._model(b_docid_history, b_ctime_history, batch_size=train_info['batch_size'], actions=b_action), (train_info['batch_size'], global_batch, 1))
        loss = tf.reduce_mean(tf.square(qs-target))

      grads = tape.gradient(loss, rec._model.trainable_variables)
      optimizer.apply_gradients(zip(grads, rec._model.trainable_variables))

      train_info['train_loss'](loss)

    if train_info['timestep'] % train_info['target_update_period'] == 0:
      update_target_network(rec._model, rec._target, 1.0)

  return grads, loss, tf.reduce_mean(cum_reward)

def make_runtime(variables):
  """Makes simulation + policy log-prob runtime."""
  variables = list(variables)
  slate_var = [var for var in variables if 'slate docs' == var.name]
  # log_prob_var = log_probability.log_prob_variables_from_direct_output(
  #     slate_var)
  # accumulator = log_probability.log_prob_accumulator_variables(log_prob_var)
  tf_runtime = runtime.TFRuntime(
      network=network_lib.Network(
          variables=list(variables)),
      graph_compile=False)
  return tf_runtime


def make_train_step(
    tf_runtime,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize,
    optimizer=None,
    rec=None,
    replay_buffer=None,
    train_info=None,
):
  """Wraps a traced training step function for use in learning loops."""

  @tf.function
  def distributed_grad_and_train():
    return distributed_train_step(tf_runtime, horizon, global_batch,
                                  trainable_variables, metric_to_optimize,
                                  optimizer, rec, replay_buffer, train_info)

  return distributed_grad_and_train


def run_simulation(
    num_training_steps,
    horizon,
    global_batch,
    learning_rate,
    simulation_variables,
    trainable_variables,
    metric_to_optimize = 'reward',
    rec=None,
    train_info=None,
):
  """Runs simulation over multiple horizon steps while learning policy vars."""
  optimizer = reset_optimizer(learning_rate)
  replay_buffer = reset_replay_buffer(train_info['history_length'], global_batch)
  tf_runtime = make_runtime(simulation_variables)

  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  train_info['train_loss'] = train_loss


  train_step = make_train_step(tf_runtime, horizon, global_batch,
                               trainable_variables, metric_to_optimize,
                               optimizer, rec, replay_buffer, train_info)
  
  # initial transfer model weights to target model network
  update_target_network(rec._model, rec._target, 1.0)
  
  # for logging
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = 'log_dir/dqn/' + current_time
  summary_writer = tf.summary.create_file_writer(log_dir)

  for step in range(num_training_steps):
    _, _, reward = train_step()
    with summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.result(), step=step)
      tf.summary.scalar('reward', reward, step=step)
    print(reward)

