# LMAct: A Benchmark for In-Context Imitation Learning with Long Multimodal Demonstrations

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/lm_act/master/overview.svg" alt="Overview figure"/>
</p>

This repository provides an implementation of our ICML 2025 paper [LMAct: A Benchmark for In-Context Imitation Learning with Long Multimodal Demonstrations](https://arxiv.org/abs/2412.01441).

> In this paper, we present a benchmark to pressure-test today’s frontier
models’ multimodal decision-making capabilities in the very long-context regime
(up to one million tokens) and investigate whether these models can learn from
large numbers of expert demonstrations in their context.
We evaluate the performance of Claude 3.5 Sonnet, Gemini 1.5 Flash, Gemini 1.5
Pro, Gemini 2.0 Flash Experimental, GPT-4o, o1-mini, o1-preview, and o1 as
policies across a battery of simple interactive decision-making tasks: playing
tic-tac-toe, chess, and Atari, navigating grid worlds, solving crosswords, and
controlling a simulated cheetah.
We study increasing amounts of expert demonstrations in the context — from no
demonstrations to 512 full episodes.
Across our tasks, models rarely manage to fully reach expert performance, and
often, presenting more demonstrations has little effect.
Some models steadily improve with more demonstrations on a few tasks.
We investigate the effect of encoding observations as text or images and the
impact of chain-of-thought prompting.
To help quantify the impact of other approaches and future innovations, we open
source our benchmark that covers the zero-, few-, and many-shot regimes in a
unified evaluation.

## Contents

```
.
|
├── crafter                     - Crafter (needs to be downloaded)
|
├── data                        - Expert demonstrations (need to be downloaded)
|
├── src
|   ├── agents
|   │   ├── chess.py            - Stockfish agent (chess expert)
|   │   ├── crossword.py        - Oracle agent (crossword expert)
|   │   ├── grid_world.py       - Shortest path agent (grid world expert)
|   │   ├── random.py           - Random action agent
|   │   └── tic_tac_toe.py      - Minimax agent (tic-tac-toe expert)
|   |
|   ├── bagz.py                 - Readers for our .bag data files
|   ├── config.py               - Experiment configurations
|   ├── constants.py            - Project constants
|   |
|   ├── environments
|   │   ├── chess.py            - Chess environment
|   │   ├── crossword.py        - Crossword environment
|   │   ├── dm_control.py       - DM Control environment
|   │   ├── grid_world.py       - Grid world environment
|   │   └── tic_tac_toe.py      - Tic-tac-toe environment
|   |
|   ├── evaluate.py             - Evaluation loop
|   ├── interfaces.py           - Project interfaces
|   ├── main.py                 - Experiment launch script
|   └── prompts.py              - Prompt-building functionality
|
├── Stockfish                       - Stockfish (needs to be installed)
|
├── README.md
└── requirements.txt                - Dependencies
```

## Installation

Clone the source code into a local directory:

```bash
git clone https://github.com/google-deepmind/lm_act.git
cd lm_act
```

This repository requires Python 3.11.
`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).
Then, create and activate the conda environment:

```bash
conda create --name lm_act python=3.11
conda activate lm_act
```

Install `pip` and use it to install all the dependencies:

```bash
conda install pip
pip install -r requirements.txt
```

### Installing Crafter

Download the crafter repository:

```bash
git clone https://github.com/danijar/crafter.git
```

### Installing Stockfish

Download and compile the latest version of Stockfish (for Unix-like systems):

```bash
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```

### Downloading the Expert Demonstrations

To download our expert demonstrations to the correct locations, run the
following command:

```bash
cd data
./download.sh
cd ..
```

## Usage

Before running any code, make sure to activate the conda environment and set the
`PYTHONPATH`:

```bash
conda activate lm_act
export PYTHONPATH=$(pwd)/..
```

To evaluate an agent, run the following command:
```bash
python src/main.py \
  --environment=tic_tac_toe \
  --observation_type=txt \
  --agent=random \
  --num_demonstrations=0
```

## Citing this work

```latex
@inproceedings{ruoss2025lmact,
  author       = {Anian Ruoss and
                  Fabio Pardo and
                  Harris Chan and
                  Bonnie Li and
                  Volodymyr Mnih and
                  Tim Genewein},
  title        = {{LMAct}: A Benchmark for In-Context Imitation Learning with
                  Long Multimodal Demonstrations
  booktitle    = {{ICML}},
  year         = {2025},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
