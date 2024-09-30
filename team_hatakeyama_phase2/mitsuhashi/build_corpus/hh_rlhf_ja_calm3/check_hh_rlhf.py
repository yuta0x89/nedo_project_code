import json

filename = './out/hh-rlhf-ja/hh-rlhf-12k-ja-calm3-22b.jsonl'
filename_out = './out/hh-rlhf-ja/hh-rlhf-calm3.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record in records:
    corpus.append(record)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
