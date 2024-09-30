#!/bin/bash
for i in {3..5} {7..14} {16..20}; do
  echo $i
  ssh slurm0-a3-ghpc-$i << 'EOF'
    find /var/tmp -maxdepth 1 -user ext_kan_hatakeyama_s_gmail_com -print0 | xargs -0 rm -rf
    exit
EOF
done