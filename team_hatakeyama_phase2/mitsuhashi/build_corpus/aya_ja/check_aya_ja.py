import json
import random
from datasets import load_dataset


ds = load_dataset('ryota39/aya_ja_nemotron', split='train')

ground_truths = list()
for record in ds:
    ground_truths.append(record['targets'])

filename = './out/aya_ja/aya_ja-calm3-22b.jsonl'
filename_out = './out/aya_ja/aya_ja-dpo.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record, ground_truth in zip(records, ground_truths):
    record['answer'] = ground_truth
    corpus.append(record)

random.shuffle(corpus)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
