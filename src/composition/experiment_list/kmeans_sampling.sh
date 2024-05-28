#!/bin/bash

PYTHON_FILE="/workspaces/ModelHubTest/src/composition/composition.py"
ROOT_DIR="/workspaces/ModelHubTest/src/data/experiments/composition_experiments/serious_business/random_sampling"

n_samples=250
steps=3

for (( i=0; i<$steps; i++ ))
do
  cmd="python $PYTHON_FILE --desc=\"Testing for random sampling\" --root_dir=\"$ROOT_DIR\" --is_gt=False --n_samples=$n_samples --batch_processing=True --m_samples=26 --runs=15 --gpu_id=0 --sampling=\"kmeans\""
  echo $cmd
  echo "--------------------------------------------------"
  eval $cmd
  n_samples=$((n_samples - 50))
done

n_samples=100

while [ $n_samples -ge 10 ]
do
  cmd="python $PYTHON_FILE --desc=\"Testing for random sampling\" --root_dir=\"$ROOT_DIR\" --is_gt=False --n_samples=$n_samples --batch_processing=True --m_samples=26 --runs=15 --gpu_id=0 --sampling=\"kmeans\""
  echo $cmd
  echo "--------------------------------------------------"
  eval $cmd
  n_samples=$((n_samples - 10))
done
