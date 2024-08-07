#!/bin/bash
for i in {1..2}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    pkill python
    pkill wandb
    exit
EOF
done