# Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning

This is an official implementation of ICML 25 paper ["Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning"](https://icml.cc/virtual/2025/poster/44143).

## Introduction

HYGMA introduces a novel approach that combines hypergraph convolution networks (HGCN) with dynamic agent grouping to enhance coordination in multi-agent reinforcement learning. 

<div align="center">
  <img src="framework.png" alt="HYGMA Framework" width="800"/>
  <p>Figure 1: Overview of the HYGMA framework</p>
</div>

Our method:
1. Dynamically clusters agents into groups based on state history using spectral clustering
2. Builds hypergraph structures to model intra-group and inter-group relationships
3. Leverages hypergraph convolution networks to extract group-aware features
4. Integrates with value-decomposition methods (QMIX) and actor-critic frameworks

HYGMA demonstrates significant performance improvements over existing MARL algorithms by capturing complex group interactions and adapting agent relationships throughout training.

## Installation instructions

Our implementation is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [MAGIC](https://github.com/CORE-Robotics-Lab/MAGIC/tree/main). Uses [SMAC](https://github.com/oxwhirl/smac), Predator-Prey, [Google Research Football](https://github.com/google-research/football)as the environment.

## Run an experiment 

```shell
python3 src/main.py --config=hygma --env-config=sc2 with env_args.map_name=3s_vs_5z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Citation

If you find our paper and implementation helpful, please consider citing us.