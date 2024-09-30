import json

filename = './out/aya_ja_evol/aya_ja_evol_rejected-calm3-22b.jsonl'
filename_out = './out/aya_ja_evol/aya_ja_evol_rejected-calm3.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record in records:

    seed_prompt = record['prompt'].split('system: あなたは親切なAIアシスタントです。\nuser: ')[-1]
    seed_prompt = seed_prompt.replace('\nassistant: ', '')
    messages = [
        {'role': 'system', 'content': 'あなたは親切なAIアシスタントです。'},
        {'role': 'user', 'content': seed_prompt}
        ]
    record['prompt'] = messages
    corpus.append(record)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
