import json
import random
from datasets import load_dataset


ds = load_dataset('ryota39/aya_ja_nemotron', split='train')

ground_truths = list()
for record in ds:
    ground_truth = record['targets']
    for idx in range(5):
        ground_truths.append(ground_truth)


filename = './out/aya_ja_evol/aya_ja_evol_chosen-calm3-22b-dpo.jsonl'
filename_out = './out/aya_ja_evol/aya_ja_evol-calm3-dpo.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record, ground_truth in zip(records, ground_truths):

    seed_prompt = record['prompt'].split('system: あなたは親切なAIアシスタントです。\nuser: ')[-1]
    seed_prompt = seed_prompt.replace('\nassistant: ', '')
    messages = [
        {'role': 'system', 'content': 'あなたは親切なAIアシスタントです。'},
        {'role': 'user', 'content': seed_prompt}
        ]
    record['prompt'] = messages
    record['answer'] = ground_truth
    corpus.append(record)

random.shuffle(corpus)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
