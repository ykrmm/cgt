CONFIG_DIR=/users/k/karmimy/These/cgt/configs/cifar10-CGTMinAttn+Transformer+Linear_grid_grid_cgt_soft
REPEAT=1
MAX_JOBS=6
MAIN=${4:-main}
(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    echo $CONFIG
    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      ((CUR_JOBS >= MAX_JOBS)) && wait -n
      echo "Job launched: $CONFIG"
      python $MAIN.py --cfg $CONFIG --repeat $REPEAT --mark_done &
      ((CUR_JOBS < MAX_JOBS)) && sleep 1
      ((++CUR_JOBS))
    fi
  done

  wait
)