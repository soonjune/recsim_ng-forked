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

# python3
"""Tensorflow-specific implementations of value.FieldSpec."""

from typing import Text, Tuple, Union, Sequence

import edward2 as ed  # type: ignore
from gym import spaces
from recsim_ng.core import value
import tensorflow as tf

FieldValue = value.FieldValue
TFInvariant = Union[None, tf.TypeSpec, tf.TensorShape]
_DYNAMIC_DIM = None


class FieldSpec(value.FieldSpec):
  """Base Tensorflow field spec; checks shape consistency."""

  def __init__(self):
    self._is_tensor = False
    self._is_not_tensor = False
    self._tensor_shape = tf.TensorShape(dims=None)

  def check_value(self, field_value):
    """Overrides `value.FieldSpec`.

    If this is called multiple times then the values must satisfy one of these
    conditions:
      - They are all convertible to tensors with compatible `TensorShape`s.
      - None of them are convertible to tensors.

    Args:
      field_value: See `value.FieldSpec`.

    Returns:
      See `value.FieldSpec`.
    """
    try:
      field_value = tf.convert_to_tensor(field_value)
    except TypeError:
      pass

    if isinstance(field_value, tf.Tensor):
      self._is_tensor = True
    else:
      self._is_not_tensor = type(field_value)

    if self._is_tensor and self._is_not_tensor:
      return False, "both Tensor and non-Tensor ({}) values".format(
          self._is_not_tensor.__name__)

    if self._is_not_tensor:
      return True, ""

    shape = field_value.shape
    if not shape.is_compatible_with(self._tensor_shape):
      return False, "shapes {} and {} are incompatible".format(
          shape, self._tensor_shape)
    self._tensor_shape = self._tensor_shape.merge_with(shape)
    return True, ""

  def sanitize(self, field_value, field_name):
    """Overrides `value.FieldSpec`.

    If field_value is a tensor, this method will:
      - Rename the tensor to the name of the corresponding field for ease
      of debugging AutoGraph issues.
      - Set the tensor shape to the most specific known field shape so far.

    Args:
      field_value: See `value.FieldSpec`.
      field_name: Name of the field within the ValueSpec.

    Returns:
      a sanitized field value..
    """
    if self._is_tensor:
      # Tensor manipulations reduce a random variable to its sampled value, so
      # this case must be treated specially to avoid interfering with Edward2
      # tracing. Note that the creation of ed.RandomVariable does not trigger
      # a tracer event as opposed to the creation of a 'named' random variable,
      # e.g. ed.Bernoulli.
      if isinstance(field_value, ed.RandomVariable):
        sample = tf.identity(field_value.value, name=field_name)
        sample.set_shape(self._tensor_shape)
        field_value = ed.RandomVariable(
            field_value.distribution,
            sample_shape=field_value.sample_shape,
            value=sample)
      else:
        field_value = tf.identity(field_value, name=field_name)
        field_value.set_shape(self._tensor_shape)
    return field_value

  def invariant(self):
    return self._tensor_shape if self._is_tensor else None


class DynamicFieldSpec(FieldSpec):
  """Field spec for tensors which may change shape across iterations."""

  def __init__(self, rank, dynamic_dims):
    super().__init__()
    if not dynamic_dims:
      raise ValueError("dynamic_dims must have at least one element. "
                       "If the field has no dynamic dimensions, please use "
                       "`FieldSpec` instead.")
    if rank <= max(dynamic_dims):
      raise ValueError(
          "dynamic_dims contains higher dimensions than the rank of the tensor."
          " `rank` must be greater than `max(dynamic_dims)`.")
    # Promote the tensor shape to make use of the base class compatilibility
    # check for rank correctness.
    self._tensor_shape = self._tensor_shape.with_rank(rank)
    self._dynamic_dims = dynamic_dims
    self._rank = rank

  def check_value(self, field_value):
    """Overrides `value.FieldSpec`.

    If this is called multiple times then the values must satisfy one of these
    conditions:
      - They are all convertible to tensors with compatible `TensorShape`s.
      - None of them are convertible to tensors.

    Args:
      field_value: See `value.FieldSpec`.

    Returns:
      See `value.FieldSpec`.
    """
    ok, error_msg = super().check_value(field_value)
    if not ok:
      return ok, error_msg
    dynamic_tensor_shape = [
        _DYNAMIC_DIM if i in self._dynamic_dims else self._tensor_shape[i]
        for i in range(self._rank)
    ]
    self._tensor_shape = tf.TensorShape(dynamic_tensor_shape)
    return ok, error_msg


class Space(FieldSpec):
  """Tensorflow field spec with a Gym space."""

  def __init__(self, space):
    super().__init__()
    self._space = space

  @property
  def space(self):
    return self._space
