"""
python gen_image.py --num_workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from html2image import Html2Image
from tqdm import tqdm

from datasets import load_dataset

IMAGE_DIR = "./tmp/images"

hti = Html2Image(output_path=IMAGE_DIR, disable_logging=True)

#subset_name = "ner-wikipedia-dataset"
subset_name = "wikipedia-22-12-ja-embeddings"

dataset = load_dataset(
    "team-hatakeyama-phase2/text2html", subset_name, cache_dir="./cache"
)
sizes = [(768, 768), (1280, 1280), (1280, 768), (768, 1280)]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_files(directory, ext):
    for filename in os.listdir(directory):
        if filename.endswith(f".{ext}"):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)


# HTMLのパース関数
def parse_html(html_text):
    html_text = "<!DOCTYPE html>" + html_text.split("<!DOCTYPE html>")[-1]
    html_text = html_text.split("</html>")[0] + "</html>"
    return html_text


# HTMLを画像に変換する関数
def convert_to_image(data):
    i, content = data

    html_text = content["html"]

    if "<!DOCTYPE html>" in html_text and "</html>" in html_text:
        html_text = parse_html(html_text)

        content["text_len"] = len(content["text"])
        content["html"] = html_text

        results = []
        for size in sizes:
            image_file = f"{i}_{'-'.join(map(str, size))}.jpg"
            hti.screenshot(
                html_str=html_text,
                save_as=image_file,
                size=size,
            )

            content["file_name"] = image_file
            content["size"] = "-".join(map(str, size))

            results.append(content.copy())

        return results


def main(num_workers):
    # データセットのリストを作成
    dataset_list = [(i, content) for i, content in enumerate(dataset["train"])]

    print("=========== data len ==========")
    print(len(dataset_list))
    print("===============================")

    metadata_list = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_to_image, data) for data in dataset_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                metadata_list.extend(result)

    return metadata_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HTML to images with multiprocessing."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()

    delete_files(IMAGE_DIR, "jsonl")
    delete_files(IMAGE_DIR, "jpg")

    metadata_list = main(args.num_workers)

    with open(f"{IMAGE_DIR}/metadata.jsonl", "w", encoding="utf-8") as f:
        for html_text in metadata_list:
            f.write(json.dumps(html_text, ensure_ascii=False) + "\n")

    upload_dataset = load_dataset(
        "imagefolder", data_dir=IMAGE_DIR, cache_dir="./cache"
    )
    upload_dataset.push_to_hub(
        "team-hatakeyama-phase2/Synthetic-TextWebImages", subset_name, private=True
    )

    delete_files(IMAGE_DIR, "jsonl")
    delete_files(IMAGE_DIR, "jpg")
