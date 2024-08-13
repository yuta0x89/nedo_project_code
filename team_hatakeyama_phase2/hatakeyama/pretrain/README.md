
# 環境構築
- [環境構築script](./share-jk_pretrain_env.sh)
- [YAML](./share-jk_pretrain_env.yaml)

# [8bモデルの事前学習](./8b_pretrain/Megatron-LM/scripts/tsubame/tanuki-8b/)
- Megatron-LMを使用
- 学習スクリプトが格納されたフォルダは[こちら](./8b_pretrain/Megatron-LM/scripts/tsubame/tanuki-8b)
- GCP上での実行コマンドの例
  - sbatch --nodelist=slurm0-a3-ghpc-[2-5,8-16,18-20] --gpus-per-node=8 --time=30-00:00:00 -c 200 _run0805_synth_train.sh
# [8bモデルから8x8bモデルの生成](./moe_generation/)
  - HuggingFace形式のモデルを生成後､[mergekit](./moe_generation/mergekit/) で8x8bモデルを生成
    - [Merge時は0.1%のランダムノイズを付与した](./moe_generation/mergekit/moe_eight_noise0001.yaml)
  - [Megatron to Huggingfaceへの変換スクリプト](./convert_utils/)
  - [その後､HuggingFace to Megatron形式に変換](./moe_generation/convert/)
# [8x8bモデルの事前学習](./8x8b_pretrain/Megatron-LM/scripts/tsubame/moe_test_hatakeyama/16nodes/)
  - Megatron-LMを使用
  - Best-fit packing
  - MoE関連の仕様変更
    - z lossは表示するのみで､lossには含めない
    - 各expertの出力を正規化
  - 学習スクリプトが格納されたフォルダは[こちら](./8x8b_pretrain/Megatron-LM/scripts/tsubame/moe_test_hatakeyama/16nodes)
    - GCP上での実行コマンドの例
    - sbatch --nodelist=slurm0-a3-ghpc-[2-5,8-16,18-20] --gpus-per-node=8 --time=30-00:00:00 -c 200 _run0806tanuki8x8b_10th_4k_pack.sh