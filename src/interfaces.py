# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the interfaces for agents and environments."""

import abc
from collections.abc import Mapping, Sequence
import pathlib
from typing import Any

import dm_env
import numpy as np

from lm_act.src import config as config_lib


class Environment(dm_env.Environment):
  """Interface for environments."""

  @abc.abstractmethod
  def __init__(
      self,
      config: config_lib.Environment,
      opening_paths: Sequence[pathlib.Path] | None = None,
  ) -> None:
    """Initializes the environment."""

  @property
  @abc.abstractmethod
  def legal_actions(self) -> list[str]:
    """Returns the legal actions."""

  def sample_legal_action(self, rng: np.random.Generator) -> str:
    """Returns a random legal action."""
    # By default, we just return one of the legal actions.
    return rng.choice(self.legal_actions)

  def action_is_illegal(self, action: str) -> bool:
    """Returns whether the action is illegal."""
    # By default, we just check if the action is in the legal actions.
    return action not in self.legal_actions

  def action_is_invalid(self, action: str) -> bool:
    """Returns whether the action is valid."""
    # By default, we just check if the action is a single word.
    return len(action.split()) != 1

  def observation_spec(self) -> Any:
    return NotImplementedError('Unnecessary for this interface.')

  def action_spec(self) -> Any:
    return NotImplementedError('Unnecessary for this interface.')


class Agent(abc.ABC):
  """Interface for agents."""

  @abc.abstractmethod
  def __init__(self, config: config_lib.Agent) -> None:
    """Initializes the agent."""

  @abc.abstractmethod
  def step(
      self,
      observation: Mapping[str, Any],
      env: Environment,
      rng: np.random.Generator,
  ) -> str:
    """Returns the agent's action for the observation, environment, and rng."""
