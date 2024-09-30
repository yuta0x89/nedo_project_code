# synthetic posttraining data

- ポストトレーニング用の合成データ生成の取り組みをまとめたリポジトリです

# データセット
## HuggingFace
### SFT

|データセット名|ドメイン|サンプル数|ライセンス|
|:---|:---|:---|:---|
|[team-hatakeyama-phase2/MathInstruct-Japanese](https://huggingface.co/datasets/team-hatakeyama-phase2/MathInstruct-Japanese)|math|108,055|Apache-2.0|
|[team-hatakeyama-phase2/WebInstructSub_ja_100k_2nd](https://huggingface.co/datasets/team-hatakeyama-phase2/WebInstructSub_ja_100k_2nd)|math|100,000|Apache-2.0|
|[team-hatakeyama-phase2/WebInstructSub_ja_100k_3rd](https://huggingface.co/datasets/team-hatakeyama-phase2/WebInstructSub_ja_100k_3rd)|math|100,000|Apache-2.0|
|[team-hatakeyama-phase2/OpenBookQA-Japanese](https://huggingface.co/datasets/team-hatakeyama-phase2/OpenBookQA-Japanese)|stem|5,957|Apache-2.0|
|[team-hatakeyama-phase2/Open-Platypus-Japanese](https://huggingface.co/datasets/team-hatakeyama-phase2/Open-Platypus-Japanese)|math, reasoning|13,883|CC-BY-4.0|

### SFT with Ask-LLM

|データセット名|ドメイン|サンプル数|ライセンス|
|:---|:---|:---|:---|
|[team-hatakeyama-phase2/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k_ask_llm_train)|translation|20,000|Apache-2.0|
|[team-hatakeyama-phase2/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k_ask_llm_train)|other|10,101|Apache-2.0|
|[team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4_ask_llm_train)|math|11,392|Apache-2.0|
|[team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4_ask_llm_train)|math|929|Apache-2.0|
|[team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4_ask_llm_train)|coding|1,914|Apache-2.0|
|[team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M1_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M1_ask_llm_train)|other|4,000|Apache-2.0|
|[team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M2_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M2_ask_llm_train)|other|7,000|Apache-2.0|
|[team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M3_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/self-rewarding_instruct_AIFT_M3_ask_llm_train)|other|7,111|Apache-2.0|
|[team-hatakeyama-phase2/OpenBookQA-Japanese_train_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/OpenBookQA-Japanese_train_ask_llm_train)|stem|4,957|Apache-2.0|
|[team-hatakeyama-phase2/Synthetic-JP-10-Turns-Roleplay-Dialogues-Nemotron-4-1k_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/Synthetic-JP-10-Turns-Roleplay-Dialogues-Nemotron-4-1k_ask_llm_train)|roleplay|1,007|Apache-2.0|
|[team-hatakeyama-phase2/Synthetic-Calm3-MT-Coding-complex-69k_train_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/Synthetic-Calm3-MT-Coding-complex-69k_train_ask_llm_train)|coding|69,667|Apache-2.0|
|[team-hatakeyama-phase2/AutoMultiTurnByCalm3-22B-Corrected-reformatted_train_ask_llm](https://huggingface.co/datasets/team-hatakeyama-phase2/AutoMultiTurnByCalm3-22B-Corrected-reformatted_train_ask_llm)|muti-turn QA|59,084|Apache-2.0|
|[team-hatakeyama-phase2/ramdom-to-fixed-multiturn-Calm3_20240806filtered_ask_llm](https://huggingface.co/datasets/team-hatakeyama-phase2/ramdom-to-fixed-multiturn-Calm3_20240806filtered_ask_llm)|muti-turn QA|11,000|Apache-2.0|
|[team-hatakeyama-phase2/Hachi-Alpaca_v1.0_cleaned_ask_llm](https://huggingface.co/datasets/team-hatakeyama-phase2/Hachi-Alpaca_v1.0_cleaned_ask_llm)|stem|28,900|Apache-2.0|
|[team-hatakeyama-phase2/databricks-dolly-15k-ja-regen-nemotron_ask_llm_train](https://huggingface.co/datasets/team-hatakeyama-phase2/databricks-dolly-15k-ja-regen-nemotron_ask_llm_train)|stem|14,698|CC-BY-3.0|


### DPO

|データセット名|ドメイン|サンプル数|ライセンス|
|:---|:---|:---|:---|
|[team-hatakeyama-phase2/hh-rlhf-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/hh-rlhf-calm3)|harmless, helpfulness|12,000|MIT|
|[team-hatakeyama-phase2/aya-ja-nemotron-dpo](https://huggingface.co/datasets/team-hatakeyama-phase2/aya-ja-nemotron-dpo)|stem|6,259|Apache-2.0|
|[team-hatakeyama-phase2/aya-ja-evol-instruct-calm3-dpo](https://huggingface.co/datasets/team-hatakeyama-phase2/aya-ja-evol-instruct-calm3-dpo)|stem|31,295|Apache-2.0|
|[team-hatakeyama-phase2/WebInstructSub_ja_100k_1st](https://huggingface.co/datasets/team-hatakeyama-phase2/WebInstructSub_ja_100k_1st)|math|112,640|Apache-2.0|



## 環境構築

```
pip install -r requirements.txt
```

## 動作環境

- NVIDIA H100 80GB 1台

## ask_llm

- 指定したデータセットに対してask_llmにより品質スコアを付与

```python
python scorer_askllm.py
```

## aya_ja

- `aya_ja`のデータセットの応答文を`Nemotron-4-340B`で再生成する

```python
python generate_aya_ja_nemotron.py
```

- dpo用にrejectedの応答文を生成

```python 
python generate_aya_ja_rejected_batch.py
```

## aya_ja_evol_instruct

- evol-instructの進化元のプロンプトを作成

```python
python prepare_evol_instruct.py
```

- evol-instructの進化後のプロンプトを生成

```python
python generate_aya_ja_evol_prompt_batch.py
```

- evol-instructの進化後のプロンプトに対するchosenの応答文を生成

```python
python generate_aya_ja_evol_chosen_batch.py
```

- evol-instructの進化後のプロンプトに対するrejectedの応答文を生成

```python
python generate_aya_ja_evol_rejected_batch.py
```

## hh_rlhf_ja_calm3

- `llm-jp/hh-rlhf-12k-ja`の応答文を再生成

```python
python generate_hh_rlhf_ja_batch.py
```

- 合成データのチェック
  - デバッグモードで目視確認

```python
python check_hh_rlhf.py
```

## math_instruct_ja

- `TIGER-Lab/MathInstruct`の和訳

```python
python generate_mathinstruct_ja_batch.py
```

- 合成データの整形

```python
python check_mathinstruct_ja.py
```

## openbookqa_ja

- `allenai/openbookqa`の和訳

```python
python generate_openbookqa_ja_batch.py
```

- 合成データの整形

```python
python check_openbookqa_ja.py
```

## openplatypus_ja

- `garage-bAInd/Open-Platypus`の和訳

```python
python generate_openplatypus_ja_batch.py
```

- 合成データの整形

```python
python check_openbookqa_ja.py
```

## webinstructsub_ja

- chosenの応答文の生成

```python
python generate_webinstructsub_ja_chosen_batch.py
```

- rejectedの応答文の生成

```python
python generate_webinstructsub_ja_rejected_batch.py
```

- 合成データのチェック
  - デバッグモードで目視確認

```python
python check_webinstructsub_ja.py
```
