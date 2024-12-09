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

"""DM Control environment."""

import ast
from collections.abc import Mapping
import dataclasses
import pathlib
from typing import Any, Literal

from dm_control import suite
import dm_env
import numpy as np
from numpy import random

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import interfaces


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentConfig(config_lib.Environment):
  """Configuration for the environment."""

  name: str = 'dm_control'
  domain: str = 'cheetah'
  task: str = 'run'
  observation_type: Literal['dict'] = 'dict'
  seed: int = 0

  def __post_init__(self):
    object.__setattr__(self, 'name', f'{self.name}_{self.domain}_{self.task}')


class DMControl(interfaces.Environment):
  """DM Control environment."""

  def __init__(
      self,
      config: EnvironmentConfig,
      opening_paths: list[pathlib.Path] | None = None,
      openings: list[Mapping[str, Any]] | None = None,
  ) -> None:
    if openings is not None:
      self._openings = openings
    elif opening_paths is not None:
      self._openings = list()
      for opening_path in opening_paths:
        observations = bagz.BagReader(
            (opening_path / 'observations_dict.bag').as_posix()
        )
        self._openings.append(ast.literal_eval(observations[0].decode('utf-8')))
    else:
      raise ValueError('Either `openings` or `opening_paths` must be provided.')

    self._config = config
    self._env = suite.load(
        domain_name=config.domain,
        task_name=config.task,
        task_kwargs=dict(random=config.seed),
    )
    self._action_spec = self._env.action_spec()

  @property
  def domain(self) -> str:
    return self._config.domain

  @property
  def task(self) -> str:
    return self._config.task

  def _prepare_observation(self, time_step: dm_env.TimeStep) -> dm_env.TimeStep:
    time_step.observation['dict'] = str(
        {k: v.tolist() for k, v in time_step.observation.items()}
    )
    return time_step

  def _set_init_state(self, observation: Mapping[str, Any]) -> dm_env.TimeStep:
    """Returns the time step after setting the initial environment state."""
    time_step = self._env.reset()

    with self._env.physics.reset_context():
      match self._config.domain:
        case 'cheetah' | 'hopper':
          self._env.physics.data.qpos[0] = 0
          self._env.physics.data.qpos[1:] = observation['position']
          self._env.physics.data.qvel[:] = observation['velocity']
        case 'point_mass':
          self._env.physics.data.qpos[:] = observation['position']
          self._env.physics.data.qvel[:] = observation['velocity']
        case _:
          raise ValueError(f'Unknown domain: {self._config.domain}')

    self._env.physics.after_reset()
    env_observation = self._env.task.get_observation(self._env.physics)

    # Check that the observation from the environment matches `observation`.
    for key, value in observation.items():
      np.testing.assert_equal(env_observation[key], value)

    time_step = dm_env.TimeStep(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=env_observation,
    )
    return self._prepare_observation(time_step=time_step)

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment."""
    observation = self._openings.pop(0)
    return self._set_init_state(observation)

  def step(self, action: str) -> dm_env.TimeStep:
    """Steps the environment."""
    actions = self._extract_actions(action)
    return self._prepare_observation(self._env.step(actions))

  @property
  def legal_actions(self) -> list[str]:
    """Returns the legal actions."""
    return [
        'A comma-separated list (enclosed by square brackets) of'
        f' {self._action_spec.shape[0]} values between'
        f' {self._action_spec.minimum.tolist()} and'
        f' {self._action_spec.maximum.tolist()}.'
    ]

  def sample_legal_action(self, rng: random.Generator) -> str:
    min_values = self._action_spec.minimum
    max_values = self._action_spec.maximum
    return str(
        (
            (max_values - min_values) * rng.random(len(min_values)) + min_values
        ).tolist()
    )

  def action_is_illegal(self, action: str) -> bool:
    """Returns whether the action is legal."""
    # For DM Control, we treat invalid and illegal actions as the same.
    return self.action_is_invalid(action)

  def action_is_invalid(self, action: str) -> bool:
    """Returns whether the action is valid."""
    try:
      self._extract_actions(action)
      return False
    except ValueError:
      return True

  def _extract_actions(self, action: str) -> np.ndarray:
    if not action.startswith('[') or not action.endswith(']'):
      raise ValueError(f'Invalid action: {action}')
    action = action[1:-1]
    values = np.fromiter(map(float, action.split(',')), dtype=np.float64)
    self._action_spec.validate(values)
    return values
