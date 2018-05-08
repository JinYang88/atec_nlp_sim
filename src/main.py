import mypandas as pd

df = pd.read_csv("../data/train.tsv", "\t")
print(df.head())