import argparse
import pandas as pd
import json
import os
import glob
import random
from src.repeated_phrase import is_repetitive_japanese
from src.clean_utils import clean_text_list
from tqdm import tqdm

#ca系のテキストの統合とクリーニング

def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    period_count = text.count('.')
    ratio = word_count / period_count if period_count > 0 else word_count
    #print(ratio)
    return ratio > threshold


#ca系textのクリーン
def clean_ca_text(text):
    noise_keywords=[
"生成",
"質問と回答",
"テキスト",
    ]

    lines=text.split("\n")
    new_lines=[]
    for line in lines:
        for key in noise_keywords:
            found=False
            if line.find(key)>=0:
                found=True
                break
        if not found:
            new_lines.append(line)

    text="\n".join(new_lines)
    return text.strip()

def main(n_records_per_text, max_records_per_jsol, out_dir, column_name, input_path, do_clean):
    print("checking files ...")
    data_dirs = glob.glob(f"{input_path}/*.parquet")
    data_dirs+= glob.glob(f"{input_path}/*.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    print("files to convert")
    print(data_dirs)

    print("do clean:",do_clean)
    if do_clean=="False" or do_clean=="false":
        do_clean=False
        print("do not clean")

    record_count = 0
    out_file_idx = 0
    for data_path in data_dirs:
        print("processing ",data_path)
        
        if data_path.find(".parquet")>0:
            df = pd.read_parquet(data_path)
        elif data_path.find(".jsonl"):
            try:
                df = pd.read_json(data_path,lines=True)
            except:
                continue
        else:
            raise ValueError("wrong file: ",data_path)

        # 通常のテキスト
        if column_name != "ja":
            text_list = df[column_name].tolist()[:n_records_per_text]
            if do_clean:
                text_list = clean_text_list(text_list)
            cleaned_text_list=text_list
        # バイリンガルのテキスト
        else:
            ja_text_list = df["ja"].tolist()[:n_records_per_text]
            en_text_list = df["eng"].tolist()[:n_records_per_text]
            cleaned_text_list = []
            for ja_text, en_text in tqdm(zip(ja_text_list, en_text_list)):
                ja_text=clean_ca_text(ja_text)
                if do_clean:
                    try:
                        cleaned_ja_text = is_repetitive_japanese(ja_text)
                    except:
                        cleaned_ja_text=""
                
                    if cleaned_ja_text == "" or en_text == "":
                        continue
                    #英語はたまにピリオドがない変な文章が入るので、弾く
                    if is_abnormal_text(en_text):
                        continue
                if random.random() < 0.5:
                    text = ja_text+"\n\n"+en_text
                else:
                    text = en_text+"\n\n"+ja_text
                cleaned_text_list.append(text)
        text_list = list(set(cleaned_text_list))

        # 書き出し
        for text in tqdm(text_list):
            record_count += 1
            if record_count > max_records_per_jsol :
                out_file_idx += 1
                record_count = 0
            out_file = f"{out_dir}/{out_file_idx}.jsonl"
            with open(out_file, "a") as f:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text data and save to JSONL files.")
    parser.add_argument("--n_records_per_text", type=int,
                        default=10**15, help="Number of records per text for debug")
    parser.add_argument("--max_records_per_jsol", type=int,
                        default=1*10**7, help="Maximum records per JSONL file")
    parser.add_argument("--out_dir", type=str,
                        default="processed", help="Output directory")
    parser.add_argument("--column_name", type=str,
                        default="output_text", help="Column name containing text")
    parser.add_argument("--input_path", type=str,
                        default="data", help="Input directory path")
    parser.add_argument("--do_clean", type=str, default="True",
                        help="Whether to clean the text")

    args = parser.parse_args()

    main(args.n_records_per_text, args.max_records_per_jsol,
         args.out_dir, args.column_name, args.input_path, args.do_clean)
