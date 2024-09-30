"""
wit_baseの準備
python tools/wit_base/to_llava_format.py

========== count qa pairs ==========
2000文字制限前: 358493
2000文字制限後: 341842
====================================
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

random.seed(42)

INSTRUCTION_POOLS = [
    "画像に写っているものの名前を教えてください。",
    "写真に映っているものが何かを説明してください。",
    "この画像に含まれるものについて教えてください。",
    "この写真に写っている物の名前を説明してください。",
    "この写真に写っているものが何かを教えてください。",
    "画像に映っている物についての知識を教えてください。",
    "写真の中にある物が何かを説明してください。",
    "写真に映っている物の関連情報を提供してください。",
    "写真の中の物が何か、名前を教えてください。",
    "この写真に写っている物が何であるかを説明してください。",
    "この画像に写っているものの背景情報を教えてください。",
    "この画像に含まれている物が何であるかを説明してください。",
]

DATA_PATH = "./dataset/wit_base"


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

    with open("tools/wit_base/wit_base.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    llava_formats = []

    for line in tqdm(lines):
        data = json.loads(line)

        wit_features = data["wit_features"]
        language_list = wit_features["language"]

        for i, lang in enumerate(language_list):
            if lang == "ja":
                hierarchical_section_title = wit_features["hierarchical_section_title"][i]
                page_title = wit_features["page_title"][i]

                if hierarchical_section_title == page_title:
                    image_filename = data["image_name"]

                    #print(image_filename)

                    answer = wit_features["context_page_description"][i]

                    if answer:
                        answer = answer.strip().replace("\n", "")

                        if len(answer) <= 2000:
                            llava_formats.append(
                                create_llava_format(
                                    random.choice(INSTRUCTION_POOLS), answer, image_filename
                                )
                            )

    print("========== count qa pairs ==========")
    print(len(llava_formats))
    print("====================================")

    caption_path = Path(DATA_PATH, "wit_base_llava_format" + ".json")

    with open(caption_path, mode="w") as f:
        json.dump(llava_formats, f, indent=2, ensure_ascii=False)
