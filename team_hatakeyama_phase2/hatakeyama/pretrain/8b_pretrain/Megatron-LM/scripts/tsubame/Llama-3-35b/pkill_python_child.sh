#!/bin/bash

pkill python
echo "pkill python end"
echo "pkill wandb end"

find /var/tmp -maxdepth 1 -user ext_kan_hatakeyama_s_gmail_com -print0 | xargs -0 rm -rf