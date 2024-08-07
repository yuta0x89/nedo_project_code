# %%
# %%
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
from genre_list import genre_list
import re
import sys
args = sys.argv
import time


# job idを取得
# job_id=os.environ['$SLURM_JOB_ID']
job_id = args[1]
flag_file_path = f"flags/{job_id}.txt"

with open(flag_file_path, "w") as f:
    f.write("1")

def load_flag():
    with open(flag_file_path, "r") as f:
        flag = f.read().strip()

    print("flag: ", flag)
    print("flag==1: ", flag == "1")

    return flag == "1"
def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length=9999
    return max_length

def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

n_turns=3
batch_size=100
non_math_code_ratio=0.5
out_dir = "0723multiturn_cl"
#out_dir = "0724multiturn_oum"

pid = os.getpid()
seed=int(pid)+int(datetime.now().timestamp())
print("seed: ",seed)
random.seed(seed)



# %%

# %%
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}_{random.randint(0,10000)}.jsonl"


# %%

model_name = "cyberagent/calm3-22b-chat"
model_name="nitky/Oumuamua-7b-instruct-v2"
tensor_parallel_size=1
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=4000,
          # max_model_len=7000,
         #  gpu_memory_utilization=0.9,
         tensor_parallel_size=tensor_parallel_size,
          )

def llm_gen(llm,prompt_list,temperature=0.7,top_k=50):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            repetition_penalty=1.2,
            top_k=50,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%
def question_to_prompt(question,role,history=[]):
    prompt=f"""<|im_start|>system
{role}
<|im_end|>"""

    if len(history)>0:
        for q,a in history:
            prompt+=f"""<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
{a}<|im_end|>"""
    prompt+=f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    return prompt


# %%
jobs="""会社員
公務員
自営業
医師
看護師
エンジニア
デザイナー
教師
販売員
サービス業
農業従事者
漁業従事者
建設業従事者
製造業従事者
運送業従事者
金融業従事者
保険業従事者
不動産業従事者
IT関連業従事者
コンサルタント
作家
アーティスト
ミュージシャン
俳優
タレント
スポーツ選手
研究者
学生
主婦/主夫
無職
"""
job_list=jobs.split("\n")
job_list=[i for i in job_list if i!=""]

character_text="""感情的知性が高い
責任感が強い
創造的である
同情的である
好奇心が強い
優れたコミュニケーション能力を持つ
論理的思考ができる
適応能力が高い
勤勉である
強い倫理観を持つ
協力的である
細部にまで注意を払う
問題解決能力が高い
楽観的である
情熱的である
適度な自信を持つ
誠実である
客観的である
オープンマインドである
ユーモアのセンスがある
粘り強い
文化的敏感性を持つ
忍耐強い
積極的である
革新的な考え方をする
リーダーシップの素質を持つ
巻き返す能力がある
多様性を尊重する
チームプレーヤーである
謙虚である
規律性がある
迅速な対応ができる
良心的な判断をする
気配りができる
他人への敬意を払う
楽観的な見通しを持つ
自制心がある
信頼に足る
機転が利く
分析的である
積極的に学ぶ姿勢を持つ
親切である
献身的である
思いやりがある
積極的である
仕事への熱意がある
創造性に富む
熱心である
尊敬に値する
倫理観が高い
革新的なアイディアを出す
戦略的思考ができる
思いやりのある
洞察力がある
尊大な態度がない
情熱的である
積極的な姿勢を持つ
細かい配慮ができる
信頼できる
社交的である
独立心がある
常に向上心を持つ
率先して行動する
強い意志を持つ
決断力がある
思いやりの心を持つ
積極的に問題解決にあたる
創造的なアイディアを提案する
積極的に知識を習得する
親しみやすい
積極的にコミュニケーションを取る
率先して課題に取り組む
積極的に新しいことに挑戦する
強いリーダーシップを発揮する
常に学ぶ姿勢を忘れない
積極的にチームに貢献する
細かいところにも気を配る
積極的に意見を述べる
積極的に課題解決にあたる
積極的に新しいスキルを習得する
積極的にチームワークを発揮する
積極的に問題解決に挑む
積極的に新しいアイディアを出す
積極的に新しいチャレンジをする
強い責任感を持つ
細かいところにも気を配る
積極的に新しい技術を学ぶ
積極的に新しい環境に適応する
強い意志を持って物事に取り組む
積極的に新しい経験をする
荒々しい
性格の悪い
細かい
気難しい
"""
character_list=character_text.split("\n")
character_list=[i for i in character_list if i!=""]

response_text="""過去の発言を参照しながら､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､正確性に疑問を呈する反論を生成しなさい､
過去の発言を参照しながら､信頼性に疑問を呈する反論を生成しなさい､
過去の発言を参照しながら､強い反論を生成しなさい､
過去の発言を参照しながら､ロジカルな反論を生成しなさい､
過去の発言を参照しながら､感情的な反論を生成しなさい､
過去の発言を参照しながら､見落としていた点を突く反論を生成しなさい､
過去の発言を参照しながら､核心突く反論を生成しなさい､
過去の発言を参照しながら､反論を生成しなさい､
過去の発言を参照しながら､賛成意見と疑問を生成しなさい､
過去の発言を参照しながら､内容を深く掘り下げる､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答に反対する旨の､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答に賛成する旨の､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答に疑問を呈する旨の､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､話題を変えたい旨の､追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答の信頼性を再確認する旨の追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､さらなる関連情報を求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､異なる視点からの追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､実践的な応用についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､具体的な例を求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､データや統計を求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､関連する最新の研究やニュースを求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答の前提を確認する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答の影響を考慮する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答の長所を強調する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､回答の短所を指摘する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､将来的な展望についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､異なる分野への応用可能性についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､倫理的な観点からの追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､具体的な手順や方法を詳細に求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､他の研究や意見と比較する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､理論的な基盤を確認する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､具体的な成功事例を求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､失敗事例やリスクについての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､関連する実験結果を求める追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､社会的な影響についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､環境への影響を考慮する追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､費用対効果についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､トレンドや未来予測についての追加の質問や反論を1つ生成しなさい
過去の発言を参照しながら､要点を求める質問を生成しなさい
過去の発言を参照しながら､要点を箇条書きで求める質問を生成しなさい
過去の発言を要約して再生成させる指示を一つ生成しなさい｡
過去の発言をJSONで要約して再生成させる指示を一つ生成しなさい｡
過去の発言をYAMLで要約して再生成させる指示を一つ生成しなさい｡
過去の発言を箇条書きで要約して再生成させる指示を一つ生成しなさい｡
過去の発言のデータ解析を求める指示を一つ生成しなさい｡"""

response_list=response_text.split("\n")
response_list=[i for i in response_list if i!=""]

# %%


while True:
    seed=int(pid)+int(datetime.now().timestamp())
    print("seed: ",seed)
    random.seed(seed)
    parallel_conversations=[{"qid":i,"conversations":[]} for i in range(batch_size)]
    for turn_id in range(n_turns):
        flag = load_flag()
        if flag:
            print("flag is true. continue processing")
        else:
            print("flag!=1. finish data processing ")
            raise ValueError("finish!")
 
        print("turn_id",turn_id)
        #はじめのターンはランダムな質問
        if turn_id==0:
            prompt_list=[]
            for qid in range(len(parallel_conversations)):
                job=random.choice(job_list)
                character=random.choice(character_list)
                role=f"あなたは{job}です。{character}性格です。"
                genre=random.choice(genre_list)+","+random.choice(genre_list)
                if random.random()<non_math_code_ratio:
                    command=f"{genre}に関する指示や質問を一つだけしてください。質問や指示のみを出力し､それ以外は何も含めないこと"
                else:
                    command=f"{genre}に関する数学やプログラミングの指示や質問を一つだけしてください。質問や指示のみを出力し､それ以外は何も含めないこと"
                prompt_list.append(question_to_prompt(command,role))
            print(prompt_list[:3])  
        #2ターン目以降は新たな話題
        else:
            prompt_list=[]
            for qid in range(len(parallel_conversations)):
                job=random.choice(job_list)
                character=random.choice(character_list)
                role=f"あなたは{job}で､userです。{character}性格です。"
                response=random.choice(response_list)
                command=f"{response}  端的な質問や指示のみを出力し､それ以外は何も含めないこと｡"
                old_conversation_text=""
                for q,a in parallel_conversations[qid]["conversations"]:
                    old_conversation_text+=f"""Q: {q} A: {a}\n"""
                command+=f"""過去のやりとり: {old_conversation_text}"""
                #prompt_list.append(question_to_prompt(command,role,parallel_conversations[qid]["conversations"]))
                prompt_list.append(question_to_prompt(command,role))

        question_list=llm_gen(llm,prompt_list)

        #解答する
        prompt_list=[]
        for qid in range(len(parallel_conversations)):
            character=random.choice(character_list)
            role=f"あなたはアシスタントです。{character}性格です。"
            command=f"次の質問に日本語で回答しなさい｡"
            prompt_list.append(question_to_prompt(question_list[qid],role,parallel_conversations[qid]["conversations"]))
        answer_list=llm_gen(llm,prompt_list,temperature=0.01,top_k=1)

        for qid in range(len(parallel_conversations)):
            parallel_conversations[qid]["conversations"].append((question_list[qid],answer_list[qid]))

    #書き出し    
    for record in parallel_conversations:
        text=""
        remove_flag=False
        for q,a in record["conversations"]:

            #長すぎるものは削除(壊れた出力)
            if get_longest_phrase_length(q)>100 or get_longest_phrase_length(a)>100:
                remove_flag=True
                break
            if is_abnormal_text(q) or is_abnormal_text(a):
                remove_flag=True
                break

            text+=f"""user: {q} assistant: {a}\n"""
        text=text.strip()
        record["text"]=text
        if text=="":
            continue

        record.pop("conversations")

        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                                                                    



