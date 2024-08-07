#!/bin/bash
for i in {2..5} {8..16} {18..20}; do
#for i in {3..15}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    pkill python
    pkill wandb
    exit
EOF
done