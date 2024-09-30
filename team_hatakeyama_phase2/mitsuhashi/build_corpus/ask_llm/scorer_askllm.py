import os
import copy
import torch
import random
import datasets
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_askllm import AskLLM


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return

def parse_oai_message(messages):
    sentence = list()
    for idx, message in enumerate(messages):
        role = 'system' if idx%2== 0 else 'user'
        sentence.append(f"{role}: {message['content']}")
    sentence = ''.join(sentence)
    return sentence

def parse_dolly(input, instruction, answer):
    sentence = list()
    sentence.append(f'user: {input}\n{instruction}\nsystem: {answer}')
    sentence = ''.join(sentence)
    return sentence

def parse_openbook(question_stem, response):
    sentence = list()
    sentence.append(f'user: {question_stem}{input}\nsystem: {response}')
    sentence = ''.join(sentence)
    return sentence

def parse_self_rewarding(instruction, input, output):
    sentence = list()
    sentence.append(f'user: {instruction}{input}\nsystem: {output}')
    sentence = ''.join(sentence)
    return sentence


ask_prompt_translation = "Does the following rubric meet the criteria?\n1.Accuracy of meaning: Is the original meaning accurately conveyed?\n2.Naturalness and fluency: Does the translation read naturally in the target language?\n3.Contextual and purposeful appropriateness: Are appropriate expressions chosen according to the context?\n4.Cultural appropriateness: Are cultural references and idioms appropriately translated?\n5.Consistency: Are terms and expressions used consistently throughout the document?",

ask_prompt_dialogue = "Does the following rubric meet the criteria?\n1. Context understanding and continuity: Is information from previous turns properly understood and utilized?\n2. Appropriateness and relevance of responses: Are the user's questions and statements accurately responded to?\n3. Dialogue deepening and development: Are new perspectives and information appropriately introduced to develop the dialogue?\n4. Consistency and character maintenance: Is a consistent tone and attitude maintained throughout the dialogue?\n5. Flexibility and adaptability: Can unexpected user inputs or topic changes be appropriately handled? Can clarification be sought or misunderstandings corrected when necessary?"

ask_prompt_roleplay = "Does the following rubric meet the criteria?\n1. Character consistency: Are the character's personality, background, and knowledge consistently maintained?\n2. Adaptability to situations: Can it respond flexibly to unexpected developments?\n3. Naturalness of dialogue: Is the conversation natural and fluent, using human-like speech and expressions?\n4. Goal achievement: Can it develop according to the purpose of the role-play (education, training, entertainment, etc.)?\n5. Ethical considerations: Does it avoid inappropriate content and discriminatory expressions?"

ask_prompt_math = "Is it possible to solve this problem with the information provided?"

ask_prompt_code = "Is it possible to execute the given code without any revisions?"

ask_prompt_reasoning = "Does the following rubric meet the criteria?\n1. Contextual understanding: Is the given question appropriately understood?\n2. Consistency between question and response: Is the response aligned with the question, based on a proper understanding of what answer is being sought?\n3. High quality of response: Rather than an answer anyone could give, does the response contain much useful information and provide new information to users?\n4. Response that stimulates imagination: In addition to knowledge that could be found through research, does the response provide new insights to users?\n5. Logical thinking: For the given question, is the response constructed by building one's own logic and reaching an answer in the fewest steps?"

ask_prompt_dialogue_and_code = "Does the following rubric meet the criteria?\nExecutability: 1. Is it possible to execute the given code without any revisions?\n2. Context understanding and continuity: Is information from previous turns properly understood and utilized?\n3. Appropriateness and relevance of responses: Are the user's questions and statements accurately responded to?\n4. Dialogue deepening and development: Are new perspectives and information appropriately introduced to develop the dialogue?\n5. Consistency and character maintenance: Is a consistent tone and attitude maintained throughout the dialogue?\n6. Flexibility and adaptability: Can unexpected user inputs or topic changes be appropriately handled? Can clarification be sought or misunderstandings corrected when necessary?"

login(token=os.environ.get('HUGGINGFACE_API_KEY'))

SEED = 42
seed_everything(SEED)

dataset_names = [
    'Aratako/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k',
    'Aratako/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k',
    'Aratako/Synthetic-JP-10-Turns-Roleplay-Dialogues-Nemotron-4-1k',
    'team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4',
    'team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4',
    'team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4',
    'team-hatakeyama-phase2/synth-magpie-jp-reasoning-nemotron-4',
    'kanhatakeyama/databricks-dolly-15k-ja-regen-nemotron',
    'team-hatakeyama-phase2/OpenBookQA-Japanese',
    'HachiML/self-rewarding_instruct',
    'HachiML/self-rewarding_instruct',
    'HachiML/self-rewarding_instruct',
    'team-hatakeyama-phase2/Synthetic-Calm3-MT-Coding-complex-69k',
    'HachiML/Hachi-Alpaca',
    'kanhatakeyama/ramdom-to-fixed-multiturn-Calm3',
    'team-hatakeyama-phase2/AutoMultiTurnByCalm3-22B-Corrected-reformatted',
]

ask_prompts = [
    ask_prompt_translation,
    ask_prompt_dialogue,
    ask_prompt_roleplay,
    ask_prompt_math,
    ask_prompt_code,
    ask_prompt_math,
    ask_prompt_reasoning,
    ask_prompt_reasoning,
    ask_prompt_reasoning,
    ask_prompt_reasoning,
    ask_prompt_reasoning,
    ask_prompt_reasoning,
    ask_prompt_dialogue_and_code,
    ask_prompt_reasoning,
    ask_prompt_dialogue,
    ask_prompt_dialogue,
]

splits = [
    'train',
    'train',
    'train',
    'train',
    'train',
    'train',
    'train',
    'train',
    'train',
    'AIFT_M1',
    'AIFT_M2',
    'AIFT_M3',
    'train',
    'v1.0_cleaned',
    '20240806filtered',
    'train',
]

model_name = 'cyberagent/calm3-22b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="cuda",
    )

tbar = enumerate(zip(dataset_names, ask_prompts, splits))
for step, (dataset_name, ask_prompt, split) in tbar:

    stem = dataset_name.split('/')[-1]

    dataset = load_dataset(dataset_name)
    dataset = dataset[split]
    # dataset = dataset[split].select(range(0, 105)) # テスト用
    print(dataset)

    dataset = dataset.to_pandas()

    if stem == 'Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'Synthetic-JP-Conversations-Magpie-Nemotron-4-10k':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'Synthetic-JP-10-Turns-Roleplay-Dialogues-Nemotron-4-1k':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'synth-persona-jp-math-nemotron-4':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'synth-magpie-jp-coding-nemotron-4':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'synth-magpie-jp-math-nemotron-4':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'synth-magpie-jp-reasoning-nemotron-4':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'databricks-dolly-15k-ja-regen-nemotron':
        dataset['sentences'] = list(
            map(
                parse_dolly,
                dataset['reg_Input'],
                dataset['reg_Instruction'],
                dataset['reg_Answer'],
                )
            )
    elif stem == 'OpenBookQA-Japanese':
        dataset['sentences'] = list(
            map(
                parse_openbook,
                dataset['question_stem'],
                dataset['response'],
                )
            )
    elif stem == 'self-rewarding_instruct':
        dataset['sentences'] = list(
            map(
                parse_self_rewarding,
                dataset['instruction'],
                dataset['input'],
                dataset['output_example']
                )
            )
    elif stem == 'Hachi-Alpaca':
        dataset['sentences'] = list(
            map(
                parse_self_rewarding,
                dataset['instruction'],
                dataset['input'],
                dataset['output']
                )
            )
    elif stem == 'Synthetic-Calm3-MT-Coding-352k':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'Synthetic-Calm3-MT-Coding-complex-69k':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'ramdom-to-fixed-multiturn-Calm3':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))
    elif stem == 'AutoMultiTurnByCalm3-22B-Corrected-reformatted':
        dataset['sentences'] = list(map(parse_oai_message, dataset['messages']))

    dataset = datasets.Dataset.from_pandas(dataset)

    prompt_template_prefix = "###\n"
    prompt_template_postfix = f"""
###

{ask_prompt}

OPTIONS: yes / no
ANSWER:"""

    yes_tokens = [" yes", " Yes"]

    llm = AskLLM(
        tokenizer,
        model,
        prompt_template_prefix=prompt_template_prefix,
        prompt_template_postfix=prompt_template_postfix,
        yes_tokens=yes_tokens,
        max_tokens=2048,
        )

    asked_scores = list()
    tbar = tqdm(enumerate(dataset), total=len(dataset))
    for step, record in tbar:

        datapoint = record['sentences']
        scores = llm.ask([datapoint])
        score = scores.tolist()[0]
        text = datapoint[:100].replace("\n", " ")
        asked_scores.append(score)
        # print(f"score: {score:.4f}\ttext: {text}")

    print(f"処理が完了しました。合計 {len(asked_scores)} 個のスコアを計算しました。")

    dataset = dataset.to_pandas()
    dataset['ask_llm_score'] = asked_scores
    dataset = datasets.Dataset.from_pandas(dataset)
    dataset = dataset.remove_columns('sentences')
    dataset.push_to_hub(f"{stem}_{split}_ask_llm", private=True)
