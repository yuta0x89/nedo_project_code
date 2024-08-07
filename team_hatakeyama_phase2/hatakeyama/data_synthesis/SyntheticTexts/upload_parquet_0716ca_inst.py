import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import HfApi, logging
import glob
from tqdm import tqdm
jsonl_dir = "0715ca_auto_instruct/"
jsonl_list = glob.glob(f"{jsonl_dir}/*.jsonl")
jsonl_list.sort()

logging.set_verbosity_debug()
hf = HfApi()

chunk_size = 10000000  # 50万件ごとに分割

# 一時的にデータを保持するためのリスト
temp_data = []
i = 0  # チャンクのカウンター

for path in tqdm(jsonl_list):
    filename = path.split("/")[-1]
    dataset_name = filename.split(".")[0]

    # JSONLファイルを読み込む
    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        print(e)
    
    # 一時リストにデータを追加
    temp_data.append(df)

    # 一時リストのデータを結合
    combined_df = pd.concat(temp_data, ignore_index=True)

    # チャンクサイズを超える場合、Parquetに変換してアップロード
    while len(combined_df) >= chunk_size:
        chunk = combined_df[:chunk_size]
        combined_df = combined_df[chunk_size:]
        # 重複削除
        chunk= chunk.drop_duplicates(subset=['text'])

        
        table = pa.Table.from_pandas(chunk)
        parquet_path = f"{jsonl_dir}/{dataset_name}_part{i + 1}.parquet"
        pq.write_table(table, parquet_path)
        
        # Parquetファイルをアップロード
        hf.upload_file(path_or_fileobj=parquet_path,
                       path_in_repo=f"data/{dataset_name}_part{i + 1}.parquet",
                       repo_id="team-hatakeyama-phase2/0716-calm3-22b-random-genre-inst",
                       repo_type="dataset")
        i += 1
    
    # 処理したデータをtemp_dataから削除
    temp_data = [combined_df]

# 残りのデータもParquetに変換してアップロード
if len(combined_df) > 0:

    combined_df = combined_df.drop_duplicates(subset=['text'])
    table = pa.Table.from_pandas(combined_df)
    parquet_path = f"{jsonl_dir}/{dataset_name}_part{i + 1}.parquet"
    pq.write_table(table, parquet_path)
    
    hf.upload_file(path_or_fileobj=parquet_path,
                   path_in_repo=f"data/{dataset_name}_part{i + 1}.parquet",
                   repo_id="team-hatakeyama-phase2/0716-calm3-22b-random-genre-inst",
                   repo_type="dataset")
