

# [8bモデルの事前学習](./8b_pretrain/Megatron-LM/scripts/tsubame/tanuki-8b/)
# [8bモデルから8x8bモデルの生成](./moe_generation/)
    - HuggingFace形式のモデルを生成後､mergekitで8x8bモデルを生成
    - その後､HuggingFace to Megatron形式に変換
# [8x8bモデルの事前学習](./8x8b_pretrain/Megatron-LM/scripts/tsubame/moe_test_hatakeyama/16nodes/)
    - Best-fit packing
    - MoE関連の仕様変更
      - z lossは表示するのみで､lossには含めない
      - 各expertの出力を正規化
