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

"""Evaluates an agent on the LMAct benchmark."""

from collections.abc import Sequence
import logging

from absl import app
from absl import flags
import immutabledict
import numpy as np
import tqdm

from lm_act.src import config as config_lib
from lm_act.src import evaluate
from lm_act.src.agents import chess as chess_agent
from lm_act.src.agents import crossword as crossword_agent
from lm_act.src.agents import grid_world as grid_world_agent
from lm_act.src.agents import random as random_agent
from lm_act.src.agents import tic_tac_toe as tic_tac_toe_agent
from lm_act.src.environments import chess
from lm_act.src.environments import crossword
from lm_act.src.environments import dm_control
from lm_act.src.environments import grid_world
from lm_act.src.environments import tic_tac_toe


_ENVIRONMENT = flags.DEFINE_enum(
    name='environment',
    default='tic_tac_toe',
    enum_values=[
        'chess',
        'crossword',
        'dm_control',
        'grid_world',
        'tic_tac_toe',
    ],
    help='The environment to evaluate.',
)
_OBSERVATION_TYPE = flags.DEFINE_enum(
    name='observation_type',
    default='txt',
    enum_values=['coords', 'dict', 'fen', 'pgn', 'png', 'rgb', 'txt'],
    help='The observation representation to evaluate.',
)
_ACTION_TYPE = flags.DEFINE_enum(
    name='action_type',
    default='txt',
    enum_values=['txt', 'san'],
    help='The action representation to evaluate.',
)
_AGENT = flags.DEFINE_enum(
    name='agent',
    default='random',
    enum_values=[
        'random',
        'chess_stockfish',
        'crossword_oracle',
        'grid_world_shortest_path',
        'tic_tac_toe_minimax',
    ],
    help='The agent to evaluate.',
)
_NUM_DEMONSTRATIONS = flags.DEFINE_integer(
    name='num_demonstrations',
    default=0,
    help='The number of demonstrations to use.',
)
_NUM_EVALUTION_EPISODES = flags.DEFINE_integer(
    name='num_evaluation_episodes',
    default=100,
    help='The number of episodes to evaluate.',
)
_NUM_EVALUATION_STEPS = flags.DEFINE_integer(
    name='num_evaluation_steps',
    default=100,
    help='The number of steps to evaluate.',
)

_CONFIG_BY_ENVIRONMENT = immutabledict.immutabledict({
    'chess': chess.EnvironmentConfig,
    'crossword': crossword.EnvironmentConfig,
    'dm_control': dm_control.EnvironmentConfig,
    'grid_world': grid_world.EnvironmentConfig,
    'tic_tac_toe': tic_tac_toe.EnvironmentConfig,
})
_CONFIG_BY_AGENT = immutabledict.immutabledict({
    'random': random_agent.RandomAgentConfig,
    'chess_stockfish': chess_agent.StockfishAgentConfig,
    'crossword_oracle': crossword_agent.OracleAgentConfig,
    'grid_world_shortest_path': grid_world_agent.ShortestPathAgentConfig,
    'tic_tac_toe_minimax': tic_tac_toe_agent.MinimaxAgentConfig,
})


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.getLogger().setLevel(logging.WARNING)

  print(f'Environment: {_ENVIRONMENT.value}')
  print(f'Observation type: {_OBSERVATION_TYPE.value}')
  print(f'Agent: {_AGENT.value}')
  print(f'Num evaluation episodes: {_NUM_EVALUTION_EPISODES.value}')

  scores = list()
  num_steps = list()
  num_invalid_actions = list()
  num_illegal_actions = list()
  num_empty_actions = list()

  for episode in tqdm.trange(_NUM_EVALUTION_EPISODES.value):
    (
        episode_score,
        episode_num_steps,
        episode_num_invalid_actions,
        episode_num_illegal_actions,
        episode_num_empty_actions,
    ) = evaluate.evaluate_episode(
        episode_idx=episode,
        config=config_lib.Experiment(
            num_demonstrations=_NUM_DEMONSTRATIONS.value,
            num_evaluation_steps=_NUM_EVALUATION_STEPS.value,
            agent=_CONFIG_BY_AGENT[_AGENT.value](
                action_type=_ACTION_TYPE.value,
            ),
            environment=_CONFIG_BY_ENVIRONMENT[_ENVIRONMENT.value](
                observation_type=_OBSERVATION_TYPE.value,
                action_type=_ACTION_TYPE.value,
            ),
            prompt=config_lib.Prompt(
                show_legal_actions=None,
                use_chain_of_thought=None,
            ),
        ),
    )

    scores.append(episode_score)
    num_steps.append(episode_num_steps)
    num_invalid_actions.append(episode_num_invalid_actions)
    num_illegal_actions.append(episode_num_illegal_actions)
    num_empty_actions.append(episode_num_empty_actions)

    logging.info({
        'episode': episode,
        'score': episode_score,
        'num_steps': episode_num_steps,
        'num_invalid_actions': episode_num_invalid_actions,
        'num_illegal_actions': episode_num_illegal_actions,
        'num_empty_actions': episode_num_empty_actions,
    })

  print(f'Average score: {np.mean(scores):.2f}')
  print(f'Average num steps: {np.mean(num_steps):.2f}')
  print(f'Average num invalid actions: {np.mean(num_invalid_actions):.2f}')
  print(f'Average num illegal actions: {np.mean(num_illegal_actions):.2f}')
  print(f'Average num empty actions: {np.mean(num_empty_actions):.2f}')


if __name__ == '__main__':
  app.run(main)
