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

# Lint as: python3
"""Train a recommender with the interest_evolution_simulation."""
from absl import app
import interest_evolution_simulation_dqn
import simulation_config_dqn
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


def main(argv):
  del argv
  num_users = 2 #1000

  train_info = dict()
  train_info['timestep'] = 0
  train_info['gamma'] = 0.99
  train_info['batch_size'] = 32
  train_info['update_period'] = 4 #4
  train_info['target_update_period'] = 200 #4000
  train_info['start_update'] = 1000 #1000
  train_info['history_length'] = 15

  (variables, rec), trainable_variables = (
      simulation_config_dqn.create_interest_evolution_simulation_network(
          num_users=num_users, history_length=train_info['history_length']))

  interest_evolution_simulation_dqn.run_simulation(
      num_training_steps=100, #100
      horizon=100,
      global_batch=num_users,
      learning_rate=1e-4,
      simulation_variables=variables,
      trainable_variables=trainable_variables,
      metric_to_optimize='reward',
      rec=rec,
      train_info=train_info)


if __name__ == '__main__':
  app.run(main)
