#!/bin/bash

# 引数のチェック
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hostname> <stop|start>"
    exit 1
fi

# 引数からホスト名と動作を取得
hostname=$1
action=$2

# 動作に応じて書き込む内容を設定
if [ "$action" == "stop" ]; then
    value="0"
elif [ "$action" == "start" ]; then
    value="1"
else
    echo "Invalid action: $action"
    echo "Action must be 'stop' or 'start'"
    exit 1
fi

# 1. ジョブIDを取得する
job_ids=$(squeue | grep "${hostname}$" | grep 0705cc_ | awk '{print $1}')

# 2. それぞれのジョブIDに対して対応するファイルを書き換える
for id in $job_ids; do
  echo "Processing job ID: $id"
  echo "$value" > /storage5/shared/corpus/synthetic/SyntheticTexts/flags/${id}.txt
done