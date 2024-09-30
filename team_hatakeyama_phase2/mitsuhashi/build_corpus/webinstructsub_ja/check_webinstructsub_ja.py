import json
import random


filename = './out/openwebinstructsub_ja/WebInstructSub_ja_100k_1st.jsonl'
filename_out = './out/openwebinstructsub_ja/WebInstructSub_ja_100k_1st.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record in records:
    corpus.append(record)

random.shuffle(corpus)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
