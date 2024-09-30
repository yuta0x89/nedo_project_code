"""
commoncatalog-cc-by-sa-ja-complexの準備
python tools/commoncatalog-cc-by-sa-ja-complex/to_llava_format_curation_calm3.py

========== count qa pairs ==========
question_ja_calm3 54999
====================================
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

INSTRUCTIONS = {
    "question_ja_calm3": None,
}

DATA_PATH = "./dataset/commoncatalog-cc-by-sa-ja-complex"


def check_abnormal_answer(data, key):
    answer_ja = data[key]
    #answer_en = data[key.replace("_ja", "")].lower()

    curation_list_ja = [
        #"ｔ個",
        #"知りません",
        #"ￒ",
        #"わかりません",
        #"分かりません",
        "ごめんなさい",
        "すみません",
        #"ｒ本",
        #"ｓ匹",
        #"ｒ匹",
        #"ｒつ",
        #"説明はつけない",
        "申し訳",
        "訳され",
    ]

    #curation_list_en = [
    #    #"translat",
    #]

    for item in curation_list_ja:
        if item in answer_ja:
            return True

    #for item in curation_list_en:
    #    if item in answer_en:
    #        return True

    return False


def create_llava_format(question, answer, image_filename):
    """
    Args:
        question (str): 質問プロンプト
        answer (str): 回答
        image_filename (str): 拡張子付きの画像ファイル名(image.jpgなど)
    """
    llava_format = {}
    conversations = []

    conversation_user = {"from": "ユーザー", "value": f"{question}\n<image>"}

    conversation_system = {"from": "システム", "value": answer}
    conversations.append(conversation_user)
    conversations.append(conversation_system)

    llava_format["image"] = image_filename
    llava_format["conversations"] = conversations

    # print(llava_format)

    return llava_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLAVA format dataset")
    args = parser.parse_args()

    dataset = load_dataset(
        "team-hatakeyama-phase2/commoncatalog-cc-by-sa-ja-complex-calm3", cache_dir="./cache"
    )

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for data in dataset["train"]:
        image_filename = str(data["photoid"]) + "." + data["ext"]
        for key in INSTRUCTIONS.keys():
            answer = data["answer_ja_calm3"]

            # 不要なデータを除去
            if check_abnormal_answer(data, key):
                #print(answer)
                #print("===============")
                continue

            # 文章が長過ぎるデータは除去
            if len(answer) <= 2000:
                llava_formats[key].append(
                    create_llava_format(data[key], answer, image_filename)
                )

    print("========== count qa pairs ==========")
    for key, item in llava_formats.items():
        print(key, len(item))
    print("====================================")

    for key in INSTRUCTIONS.keys():
        output_filename = key

        caption_path = Path(
            DATA_PATH, output_filename + "_curation" + ".json"
        )

        with open(caption_path, mode="w") as f:
            json.dump(llava_formats[key], f, indent=2, ensure_ascii=False)
