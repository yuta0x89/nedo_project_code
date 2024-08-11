import json

filename = './out/openplatypus-ja/openplatypus-ja-calm3-22b.jsonl'
filename_out = './out/openplatypus-ja/openplatypus-ja.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record in records:

    instruction = record['instruction'].replace('\n\n', '')
    response = record['response']


    sep_Q = '質問: '
    sep_A = '答え: '

    flag_Q = sep_Q in instruction or sep_A in instruction
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        instruction = instruction.split(sep_A)[0]
        instruction = instruction.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問:\n'
    sep_A = '答え:\n'

    flag_Q = sep_Q in instruction or sep_A in instruction
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        instruction = instruction.split(sep_A)[0]
        instruction = instruction.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問\n'
    sep_A = '答え\n'

    flag_Q = sep_Q in instruction or sep_A in instruction
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        instruction = instruction.split(sep_A)[0]
        instruction = instruction.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問:'
    sep_A = '答え:'

    flag_Q = sep_Q in instruction or sep_A in instruction
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        instruction = instruction.split(sep_A)[0]
        instruction = instruction.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    record['instruction'] = instruction
    record['response'] = response

    corpus.append(record)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
