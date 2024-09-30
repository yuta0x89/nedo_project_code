import json

filename = './out/openbookqa_ja/openbookqa-ja-calm3-22b_train.jsonl'
filename_out = './out/openbookqa_ja/openbookqa-ja_train.json'

with open(filename, mode='r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f.readlines()]

corpus = list()
for record in records:
    # '### 質問\nオークの木の種が植えられ、その場所に歩道が敷かれると、やがて木が高くなり、根が歩道を越える必要があります。これはつまり、\nA. 根が割れる可能性がある\nB. 根が死ぬ可能性がある\nC. 根がコンクリートを壊す可能性がある\nD. 根がばらばらになる可能性がある\n\n### 答え\nオークの木が歩道の近くにある場合、その根が歩道を破損する可能性があるため、答えはCです。'

    question_stem = record['question_stem'].replace('\n\n', '')
    response = record['response']


    sep_Q = '### 質問\n'
    sep_A = '### 答え\n'

    flag_Q = sep_Q in question_stem or sep_A in question_stem
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        question_stem = question_stem.split(sep_A)[0]
        question_stem = question_stem.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問: '
    sep_A = '答え: '

    flag_Q = sep_Q in question_stem or sep_A in question_stem
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        question_stem = question_stem.split(sep_A)[0]
        question_stem = question_stem.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問:\n'
    sep_A = '答え:\n'

    flag_Q = sep_Q in question_stem or sep_A in question_stem
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        question_stem = question_stem.split(sep_A)[0]
        question_stem = question_stem.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]


    sep_Q = '質問\n'
    sep_A = '答え\n'

    flag_Q = sep_Q in question_stem or sep_A in question_stem
    flag_A = sep_Q in response or sep_A in response
    if flag_Q or flag_A:

        question_stem = question_stem.split(sep_A)[0]
        question_stem = question_stem.split(sep_Q)[-1]

        response = response.split(sep_A)[-1]

    record['question_stem'] = question_stem
    record['response'] = response

    corpus.append(record)

with open(filename_out, mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
