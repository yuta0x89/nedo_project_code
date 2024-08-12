
# title


# Folder index
## [hatakeyama](hatakeyama)
    - 事前学習(8b, 8x8b)
    - データ合成

## [sftlab](sftlab)
- 事後学習(SFT)(8b, 8x8b)
  
## [polab](polab)
- 事後学習(DPO)(8b, 8x8b)

## [ota](ota)
- データ合成
  - [Iterative DPO, LLM-as-a-Judge](ota/iterative-dpo)
  - [Topic-Hub, Persona-Hub](ota/topic-hub)
  - [Magpie](ota/magpie)
  - [ルールベース](ota/rule-based)
- データに対するスコア付け
  - [Ask-LLM](ota/ask-llm)
  - [言語判定](ota/lang-identifier)
- Nemotron-4 推論環境構築
  - [vLLM FP8 量子化](ota/nemotron-vllm-fp8)
  - [Megatron](ota/nemotron-megatron)
- 上記手法で作成データセット (Hugging Face へのリンク)
  - SFT
    - [team-hatakeyama-phase2/synth-topic-jp-basic-math-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-math-calm3)
    - [team-hatakeyama-phase2/synth-topic-jp-basic-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-reasoning-calm3)
    - [team-hatakeyama-phase2/synth-topic-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-nemotron-4)
    - [team-hatakeyama-phase2/synth-topic-jp-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-calm3)
    - [team-hatakeyama-phase2/synth-topic-jp-highschoolmath-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-highschoolmath-nemotron-4)
    - [team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4)
    - [team-hatakeyama-phase2/synth-persona-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-reasoning-nemotron-4)
    - [team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4)
    - [team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4)
    - [team-hatakeyama-phase2/synth-magpie-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-reasoning-nemotron-4)
    - TODO: upload coding calm3
  - DPO
    - TODO: upload basic math, basic reasoning
  - 事前学習
    - [team-hatakeyama-phase2/synth-rule-arithmetic-qa](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-rule-arithmetic-qa)

# ライセンス
- ライセンスはMITとなります(一部､他のレポジトリからcloneしたコードが含まれるフォルダが存在します｡そのフォルダ内のライセンスは､元のライセンスに従います)｡
