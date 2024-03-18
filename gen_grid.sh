#!/usr/bin/env bash

CONFIG=CGT/cluster-CGTSoft+GPS+Linear
GRID=CGT/grid_cgt_soft
REPEAT=1
MAX_JOBS=3

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${CONFIG}.yaml \
  --grid configs/${GRID}.txt \
  --out_dir configs