#!/bin/bash
for i in {2..5} {8..16} {18..20}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    nvidia-smi
    exit
EOF
done