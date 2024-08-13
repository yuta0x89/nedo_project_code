# データ合成に関する種々のスクリプトが格納されたフォルダです
- vllmを使って推論します｡

- 例
  - [Calm3を使ってランダムなマルチターン会話を生成](./SyntheticTexts/0723free_multiturn_clean.py)
  - [Calm3を使ってランダムなマルチターン会話を生成(2ターン目の指示は人間が設定)](./SyntheticTexts/0801fixed_multiturn.py)
  - [Calm3を使ってロジカルな会話データを生成](./SyntheticTexts/0802logical_multiturn.py)
  - [特定のテキストをもとに教科書調の文章などを生成](./SyntheticTexts/0616cc_tanuki_f1_gen.sh)

- [ParquetToJsonlフォルダ](./ParquetToJsonl/)
  - 合成したデータをクリーニングしたり､一つのparquetファイルに統合するscript集です(基本的には不要)