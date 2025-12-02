import pandas as pd

# 讀取 parquet 檔
df = pd.read_parquet("/home/bruce/.cache/huggingface/lerobot/ethanCSL/test_0903/data/chunk-000/episode_000000.parquet")

# 看前五筆資料
print(df.columns)
print(df["observation.state"])

