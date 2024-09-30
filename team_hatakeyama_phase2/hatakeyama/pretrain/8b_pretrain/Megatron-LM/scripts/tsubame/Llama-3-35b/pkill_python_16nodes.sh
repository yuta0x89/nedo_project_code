#!/bin/bash
for i in {3..5} {7..14} {16..20}; do
  sbatch --nodelist=slurm0-a3-ghpc-[$i] --gpus-per-node=0 --time=30-00:00:00 -c 1 pkill_python_child.sh
done
