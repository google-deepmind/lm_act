# Copyright 2025 Google LLC
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

"""Evaluates a single episode."""

import copy
import os
import pathlib
from typing import Any

from absl import logging
import numpy as np

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import constants
from lm_act.src import prompts


_BASE_DIR_PATH = pathlib.Path(
    os.path.join(
        os.getcwd(),
        'data/lm_act/',
    )
)


def _load_demonstrations_and_opening_path(
    rng: np.random.Generator,
    config: config_lib.Experiment,
) -> tuple[list[list[Any]], list[list[Any]], pathlib.Path]:
  """Loads the demonstrations (observations & actions) and the opening path.

  Args:
    rng: The random number generator.
    config: The experiment configuration.

  Returns:
    - The demonstrations episodes, consisting of observations and actions.
    - The opening path, i.e., the path to the opening (which is used to set the
      initial state of the environment for the evaluation episode).

  Raises:
    ValueError: If there are insufficient demonstrations in the directory.
  """
  base_dir_path = _BASE_DIR_PATH / config.environment.name
  demonstration_names = [
      file_name
      for file_name in os.listdir(base_dir_path)
      if file_name.startswith('demonstration')
  ]
  if len(demonstration_names) < config.num_demonstrations + 1:
    raise ValueError(
        f'Insufficient demonstrations in {base_dir_path}: Need at least'
        f' {config.num_demonstrations + 1} but only found'
        f' {len(demonstration_names)}.'
    )

  if config.replay_episode:
    assert config.num_demonstrations == 1
    num_openings = config.num_demonstrations
  else:
    # We need to add 1 to account for the opening that that will be evaluated.
    num_openings = config.num_demonstrations + 1
  demonstration_names = rng.choice(
      demonstration_names,
      size=num_openings,
      replace=False,
      shuffle=False,
  )
  opening_name = demonstration_names[-1]
  demonstration_names = demonstration_names[: config.num_demonstrations]

  demo_observations = list()
  demo_actions = list()

  match config.environment.observation_type:
    case 'rgb':
      rgb_shape = constants.get_rgb_shape(config.environment.name)
      observation_decode_fn = lambda x: np.frombuffer(
          x,
          dtype=np.uint8,
      ).reshape(rgb_shape)
    case 'png':
      # PNG data does not need to be decoded.
      observation_decode_fn = lambda x: x
    case _:
      observation_decode_fn = lambda x: x.decode('utf-8')
  action_decode_fn = lambda x: x.decode('utf-8')

  for demonstration_name in demonstration_names:
    demo_dir_path = base_dir_path / demonstration_name
    observations_path = (
        demo_dir_path
        / f'observations_{config.environment.observation_type}.bag'
    )
    actions_path = (
        demo_dir_path / f'actions_{config.environment.action_type}.bag'
    )
    observations = bagz.BagReader(observations_path.as_posix())
    actions = bagz.BagReader(actions_path.as_posix())
    assert len(observations) == len(actions)
    demo_observations.append(list(map(observation_decode_fn, observations)))
    demo_actions.append(list(map(action_decode_fn, actions)))

  return demo_observations, demo_actions, base_dir_path / opening_name


def _create_demonstration_prompt(
    config: config_lib.Experiment,
    demo_observations: list[list[Any]],
    demo_actions: list[list[Any]],
) -> tuple[str, dict[str, Any]]:
  """Returns the demonstration prompt and content for the given config."""
  content_by_tag = dict()
  demo_prompts = list()

  for demo_idx, (observations, actions) in enumerate(
      zip(demo_observations, demo_actions)
  ):
    for step_idx, (observation, action) in enumerate(
        zip(observations, actions)
    ):
      match config.environment.observation_type:
        case 'fen' | 'coords':
          demo_prompt = f'Observation: {observation} '
        case 'dict':
          demo_prompt = f'Observation: {observation}\n'
        case 'pgn' | 'txt':
          demo_prompt = f'Observation:\n{observation}\n'
        case 'rgb' | 'png':
          tag = f'<IMG_{demo_idx}_{step_idx}>'
          content_by_tag[tag] = observation
          demo_prompt = f'Observation: {tag} '
        case _:
          raise ValueError(
              'Unsupported observation type:'
              f' {config.environment.observation_type}'
          )

      demo_prompt += f'Action: {action}'
      demo_prompts.append(demo_prompt)
      demo_prompts.append('\n')
    demo_prompts.append('\n')

  demonstration_prompt = prompts.build_demonstration_prompt(
      demonstrations=''.join(demo_prompts),
  )
  logging.info('Demonstration prompt: %s', demonstration_prompt)
  return demonstration_prompt, content_by_tag


def _create_trajectory_prompt(
    config: config_lib.Experiment,
    observations: list[Any],
    actions: list[Any],
    legal_actions: list[str],
) -> tuple[str, dict[str, Any]]:
  """Returns the trajectory prompt and content for the given config."""
  content_by_tag = dict()
  trajectory_prompts = list()

  # The first action is a dummy action so we place it at the end of the list.
  actions = np.roll(copy.deepcopy(actions), -1)

  for step_idx, (observation, action) in enumerate(zip(observations, actions)):
    match config.environment.observation_type:
      case 'fen' | 'coords':
        trajectory_prompt = f'Observation: {observation} '
      case 'dict':
        trajectory_prompt = f'Observation: {observation}\n'
      case 'pgn' | 'txt':
        trajectory_prompt = f'Observation:\n{observation}\n'
      case 'rgb' | 'png':
        tag = f'<IMG_{config.num_demonstrations}_{step_idx}>'
        content_by_tag[tag] = observation
        trajectory_prompt = f'Observation: {tag} '
      case _:
        raise ValueError(
            'Unsupported observation type:'
            f' {config.environment.observation_type}'
        )

    if config.prompt.include_past_actions and step_idx < len(actions) - 1:
      trajectory_prompt += f'Action: {action}'

    if trajectory_prompt:
      trajectory_prompts.append(trajectory_prompt)
      trajectory_prompts.append('\n')

  trajectory_prompt = prompts.build_trajectory_prompt(
      config=config,
      trajectory=''.join(trajectory_prompts),
      legal_actions=legal_actions,
  )
  logging.info('Current trajectory prompt: %s', trajectory_prompt)
  return trajectory_prompt, content_by_tag


def evaluate_episode_replay(
    episode_idx: int,
    config: config_lib.Experiment,
) -> int:
  """Returns the number of correctly replayed actions for a single episode."""

  # Every episode has to initialize the RNG with a different seed.
  rng = np.random.default_rng(seed=episode_idx)

  logging.info('Setting up the agent: %s.', config.agent.name)
  agent = constants.get_agent_builder(config.agent.name)(config=config.agent)

  logging.info('Loading the demonstrations and the evaluation opening name.')
  demo_observations, demo_actions, opening_path = (
      _load_demonstrations_and_opening_path(rng=rng, config=config)
  )
  assert len(demo_observations) == 1
  assert len(demo_actions) == 1

  logging.info('Replaying episode %d (opening %s).', episode_idx, opening_path)

  logging.info('Creating the demonstration chunks.')
  demonstration_prompt, demonstration_prompt_data = (
      _create_demonstration_prompt(
          config=config,
          demo_observations=demo_observations,
          demo_actions=demo_actions,
      )
  )

  num_correctly_replayed_actions = 0

  for step, (demo_observation, demo_action) in enumerate(
      zip(demo_observations[0], demo_actions[0])
  ):
    trajectory_prompt, trajectory_prompt_data = _create_trajectory_prompt(
        config=config,
        observations=demo_observations[0][: step + 1],
        actions=[None] + demo_actions[0][:step],  # Dummy initial action.
        legal_actions=list(),  # We cannot compute the legal actions.
    )
    sample = agent.step(
        observation={
            'prompt': demonstration_prompt + trajectory_prompt,
            'prompt_data': demonstration_prompt_data | trajectory_prompt_data,
        },
        environment=None,
        rng=rng,
    )
    replayed_action_is_correct = sample == demo_action
    num_correctly_replayed_actions += replayed_action_is_correct

    logging.info({
        'demo_observation': demo_observation,
        'demo_action': demo_action,
        'sample': sample,
        'replayed_action_is_correct': replayed_action_is_correct,
    })

  return num_correctly_replayed_actions


def evaluate_episode(
    episode_idx: int,
    config: config_lib.Experiment,
) -> tuple[float, int, int, int, int]:
  """Evaluates a single episode."""

  # Every episode has to initialize the RNG with a different seed.
  rng = np.random.default_rng(seed=episode_idx)

  logging.info('Setting up the agent: %s.', config.agent.name)
  agent = constants.get_agent_builder(config.agent.name)(config=config.agent)

  logging.info('Loading the demonstrations and the evaluation opening name.')
  demo_observations, demo_actions, opening_path = (
      _load_demonstrations_and_opening_path(rng=rng, config=config)
  )

  logging.info(
      'Evaluating episode %d with opening %s.', episode_idx, opening_path
  )

  logging.info('Creating the demonstration chunks.')
  demonstration_prompt, demonstration_prompt_data = (
      _create_demonstration_prompt(
          config=config,
          demo_observations=demo_observations,
          demo_actions=demo_actions,
      )
  )

  logging.info('Setting up the environment: %s.', config.environment.name)
  env = constants.get_environment_builder(config.environment.name)(
      config=config.environment,
      opening_paths=[opening_path],
  )
  time_step = env.reset()

  observations = [time_step.observation[config.environment.observation_type]]
  rewards = [time_step.reward]
  actions = [None]  # Dummy action for the initial observation.

  num_illegal_actions = num_invalid_actions = num_empty_actions = 0

  for _ in range(config.num_evaluation_steps):
    if time_step.last():
      break

    trajectory_prompt, trajectory_prompt_data = _create_trajectory_prompt(
        config=config,
        observations=observations,
        actions=actions,
        legal_actions=env.legal_actions,
    )
    sample = agent.step(
        observation=time_step.observation
        | {
            'prompt': demonstration_prompt + trajectory_prompt,
            'prompt_data': demonstration_prompt_data | trajectory_prompt_data,
        },
        environment=env,
        rng=rng,
    )

    sample_is_empty = not sample
    num_empty_actions += sample_is_empty

    if sample_is_invalid := env.action_is_invalid(sample):
      num_invalid_actions += 1
      # If the sample is invalid, we also always consider it illegal.
      sample_is_illegal = True
      num_illegal_actions += 1
    elif sample_is_illegal := env.action_is_illegal(sample):
      num_illegal_actions += 1

    action = env.sample_legal_action(rng) if sample_is_illegal else sample

    logging.info({
        'observation': time_step.observation[
            config.environment.observation_type
        ],
        'reward': time_step.reward,
        'action': action,
        'sample': sample,
        'sample_is_invalid': sample_is_invalid,
        'sample_is_illegal': sample_is_illegal,
        'sample_is_empty': sample_is_empty,
        'step_type': int(time_step.step_type),
    })

    time_step = env.step(action)
    observations.append(
        time_step.observation[config.environment.observation_type]
    )
    rewards.append(time_step.reward)
    actions.append(action)

  logging.info({
      'rgb': (
          time_step.observation['rgb']
          if 'rgb' in time_step.observation
          else None
      ),
      'observation': time_step.observation[config.environment.observation_type],
      'reward': time_step.reward,
      'action': None,
      'sample': None,
      'sample_is_invalid': None,
      'sample_is_illegal': None,
      'sample_is_empty': None,
      'step_type': int(time_step.step_type),
  })

  score = sum(rewards[1:])  # Skip the first reward since it is always None.
  num_steps = len(rewards) - 1

  return (
      score,
      num_steps,
      num_invalid_actions,
      num_illegal_actions,
      num_empty_actions,
  )
