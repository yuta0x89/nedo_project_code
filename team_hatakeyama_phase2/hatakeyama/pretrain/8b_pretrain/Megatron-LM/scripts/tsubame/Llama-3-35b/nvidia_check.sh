#!/bin/bash
for i in {3..5} {7..14} {16..20}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    nvidia-smi
    exit
EOF
done