#!/bin/bash
for i in {1..20}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    pkill python
    pkill wandb
    exit
EOF
done