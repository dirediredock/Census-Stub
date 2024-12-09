import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_csv("../DATA_10.csv")

file_map = {
    (0, 0, 0): "xx_xx_xx.csv",
    (1, 1, 1): "CN_CE_CS.csv",
    (0, 1, 0): "xx_CE_xx.csv",
    (1, 0, 0): "CN_xx_xx.csv",
    (0, 0, 1): "xx_xx_CS.csv",
    (1, 0, 1): "CN_xx_CS.csv",
    (0, 1, 1): "xx_CE_CS.csv",
    (1, 1, 0): "CN_CE_xx.csv",
}

file_handles = {}

for key, filename in file_map.items():
    file_handles[filename] = open(filename, "w")
    file_handles[filename].write(",".join(df.columns) + "\n")

for _, row in df.iterrows():
    key = (row["CN"], row["CE"], row["CS"])
    filename = file_map[key]
    file_handles[filename].write(",".join(map(str, row.values)) + "\n")

for file in file_handles.values():
    file.close()
