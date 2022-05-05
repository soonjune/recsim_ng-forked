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
"""Recommendation agents."""
from pty import slave_open
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors as selector_lib
from recsim_ng.entities.recommendation import recommender
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import estimation
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp
# for dqn
from tf_agents.networks import sequential
import itertools
import edward2 as ed


tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))


class DQNModel(tf.keras.Model):
  """A tf.keras model that returns q value for each (user, document) pair."""

  def __init__(self, num_users, num_docs, doc_embed_dim,
               history_length, slate_size):
    super().__init__(name="DQNModel")
    self._num_users = num_users
    self._history_length = history_length
    self._num_docs = num_docs
    self._doc_embed_dim = doc_embed_dim
    self._slate_size = slate_size

    # possilbe set of slates
    # Obtain all possible slates given current docs in the candidate set.
    mesh_args = [list(range(num_docs))] * slate_size
    slates = tf.stack(tf.meshgrid(*mesh_args), axis=-1)
    slates = tf.reshape(slates, shape=(-1, slate_size))
    # Filter slates that include duplicates to ensure each document is picked
    # at most once.
    unique_mask = tf.map_fn(
        lambda x: tf.equal(tf.size(input=x), tf.size(input=tf.unique(x)[0])),
        slates,
        dtype=tf.bool)
    # [num_of_slates, slate_size]
    self.slates = tf.boolean_mask(tensor=slates, mask=unique_mask)


    #embedding layers
    self._doc_proposal_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        input_length=history_length,
        name="doc_prop_embedding")
    self._doc_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        input_length=history_length,
        name="doc_embedding")

    self._user_embeddings = tf.keras.Sequential(name="recs")
    self._user_embeddings.add(tf.keras.layers.LSTM(32))
    self._user_embeddings.add(tf.keras.layers.Dense(32))
    self._user_embeddings.add(tf.keras.layers.LeakyReLU())
    self._user_embeddings.add(
        tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

    fc_layer_params = (doc_embed_dim*2, 32)
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))

    self._net = sequential.Sequential(dense_layers + [q_values_layer])


  def call(self, doc_id_history,
           c_time_history, batch_size=None, actions=None):
    # Map doc id to embedding.
    # [num_users, history_length, embed_dim]
    doc_history_embeddings = self._doc_embeddings(doc_id_history)
    # Append consumed time to representation.
    # [num_users, history_length, embed_dim + 1]
    user_features = tf.concat(
        (doc_history_embeddings, c_time_history[Ellipsis, np.newaxis]), axis=-1)
    # Flatten and run through network to encode history.
    if batch_size or actions:
      # reshape to compute all embeddings in batch (validated)
      user_features = tf.reshape(user_features, (batch_size, self._history_length, -1))
      user_embeddings = self._user_embeddings(user_features)
    #   user_embeddings = tf.reshape(user_embeddings, (batch_size, self._num_users, -1))
    else:
      user_features = tf.reshape(user_features, (self._num_users, self._history_length, -1))
      user_embeddings = self._user_embeddings(user_features)

    # Score is found using user and document features
    # history.
    # [num_docs, embed_dim + 1]
    doc_features = self._doc_proposal_embeddings(
        tf.range(1, self._num_docs + 1, dtype=tf.int32))

    # predict user choice
    cf_scores = tf.einsum("ik, jk->ij", user_embeddings, doc_features)
    unnormalized_scores = tf.math.exp(cf_scores)

    # all_scores = tf.pad(prob_scores, [[0,0], [0,1]], mode='CONSTANT') # no click has score 1
    # all_prob = tf.nn.softmax(all_scores, axis=1)
    
    # calculate q values
    if batch_size or actions:
        inputs = []
        for i in range(self._num_docs):
            doc = tf.expand_dims(doc_features[i], axis=0)
            doc = tf.repeat(doc, [batch_size], axis=0)
            x = tf.concat((user_embeddings, doc), axis=1)
            inputs.append(x)
        qs = self._net(tf.stack(inputs, axis=1))[0]
        qs = tf.squeeze(qs)
    else:
        q_vals = []
        for i in range(self._num_docs):
            doc = tf.expand_dims(doc_features[i], axis=0)
            doc = tf.repeat(doc, [self._num_users], axis=0)
            x = tf.concat((user_embeddings, doc), axis=1)
            q_vals.append(self._net(x)[0])
        qs = tf.squeeze(tf.stack(q_vals, axis=1))


    if actions is not None:
        import pdb; pdb.set_trace()
        slate_sum_q_values = tf.reshape(slate_sum_q_values, (batch_size, self._num_users, -1))
        slate_sum_q_values = tf.gather_nd(slate_sum_q_values, actions)


    slate_q_values = tf.gather(unnormalized_scores * qs, self.slates, axis=1)
    slate_scores = tf.gather(unnormalized_scores, self.slates, axis=1)
    slate_normalizer = tf.reduce_sum(input_tensor=slate_scores, axis=-1) + 1.

    # divide by (no_click_score + scores of items in slate)
    slate_q_values = slate_q_values / tf.expand_dims(slate_normalizer, axis=-1)
    slate_sum_q_values = tf.reduce_sum(input_tensor=slate_q_values, axis=-1)



    return slate_sum_q_values

@gin.configurable
class SlateQRecommender(recommender.BaseRecommender):
  """A recommender agent implements full slate Q-learning based on DQN agent."""

  def __init__(self,
               config,
               model_ctor = DQNModel,
               name="Recommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
    super().__init__(config, name=name)
    self._history_length = config["history_length"]
    self._num_docs = config.get("num_docs")
    self._num_topics = config.get("num_topics")
    self._slate_size = config.get("slate_size")
    self._model = model_ctor(self._num_users, self._num_docs, 32,
                             self._history_length, self._slate_size)
    self._target = model_ctor(self._num_users, self._num_docs, 32,
                             self._history_length, self._slate_size)
    doc_history_model = estimation.FiniteHistoryStateModel(
        history_length=self._history_length,
        observation_shape=(),
        batch_shape=(self._num_users,),
        dtype=tf.int32)
    self._doc_history = dynamic.NoOPOrContinueStateModel(
        doc_history_model, batch_ndims=1)
    ctime_history_model = estimation.FiniteHistoryStateModel(
        history_length=self._history_length,
        observation_shape=(),
        batch_shape=(self._num_users,),
        dtype=tf.float32)
    self._ctime_history = dynamic.NoOPOrContinueStateModel(
        ctime_history_model, batch_ndims=1)

    self._epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 1.0,
    decay_steps = 10000 * 2, #250000
    end_learning_rate = 0.01
    )
    self.total_calls = 0

    self.slates = self._model.slates
    
    # changed as argmax selector
    # self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
    #     self._slate_size, (self._num_users,),
    #     -np.Inf * tf.ones(self._num_users))
    # Call model to create weights
    ctime_history = self._ctime_history.initial_state().get("state")
    docid_history = self._doc_history.initial_state().get("state")
    self._model(docid_history, ctime_history)
    self._target(docid_history, ctime_history)

  def initial_state(self):
    """The initial state value."""
    doc_history_initial = self._doc_history.initial_state().prefixed_with(
        "doc_history")
    ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
        "ctime_history")
    return doc_history_initial.union(ctime_history_initial)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    chosen_doc_idx = user_response.get("choice")
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_doc_idx)
    # Update doc_id history.
    doc_consumed = tf.reshape(
        chosen_doc_features.get("doc_id"), [self._num_users])
    # We update histories of only users who chose a doc.
    no_choice = tf.equal(user_response.get("choice"),
                         self._slate_size)[Ellipsis, tf.newaxis]
    next_doc_id_history = self._doc_history.next_state(
        previous_state.get("doc_history"),
        Value(input=doc_consumed,
              condition=no_choice)).prefixed_with("doc_history")
    # Update consumed time.
    time_consumed = tf.reshape(
        user_response.get("consumed_time"), [self._num_users])
    next_ctime_history = self._ctime_history.next_state(
        previous_state.get("ctime_history"),
        Value(input=time_consumed,
              condition=no_choice)).prefixed_with("ctime_history")
    return next_doc_id_history.union(next_ctime_history)

  def slate_docs(self, previous_state, user_obs,
                 available_docs):
    """The slate_docs value."""
    del user_obs
    ctime_history = previous_state.get("ctime_history").get("state")
    docid_history = previous_state.get("doc_history").get("state")
    epsilon = self._epsilon_fn(self.total_calls)
    self.total_calls += 1

    if np.random.rand() < epsilon:
        slate_indices = tf.expand_dims([np.random.randint(0, len(self.slates)) for _ in range(ctime_history.shape[0])], axis=-1)
    else:
        # selection based on slate optimal
        qvals = self._model(docid_history, ctime_history)
        slate_indices = tf.expand_dims(tf.math.argmax(qvals, axis=1, output_type=tf.dtypes.int32), axis=-1)
    doc_indices = np.asarray([self.slates[int(i)] for i in slate_indices])
    doc_indices = tf.convert_to_tensor(doc_indices)
    slate = available_docs.map(lambda field: tf.gather(field, doc_indices))

    return slate.union(Value(doc_ranks=doc_indices, slate_ids=slate_indices))

  def specs(self):
    state_spec = self._doc_history.specs().prefixed_with("doc_history").union(
        self._ctime_history.specs().prefixed_with("ctime_history"))
    slate_docs_spec = ValueSpec(
        doc_ranks=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones(
                    (self._num_users, self._num_docs)) * self._num_docs)),
        doc_id=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_topics)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                -np.Inf,
                high=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                np.Inf)),
        doc_length=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
        slate_ids=Space(
            spaces.Box(
                low=np.zeros((self._num_users, 1)),
                high=np.ones((self._num_users, 1)) * len(self.slates))))
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))
