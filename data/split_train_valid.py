#coding=utf-8

import pandas as pd
import sys

if sys.version_info < (3, 4):
    reload(sys)
    sys.setdefaultencoding('utf-8')



data_df = pd.read_csv("atec_nlp_sim_train.tsv", sep="\t", names=["id","text1","text2","label"])


pos_rate = data_df[data_df["label"]==1].shape[0]*1.0 / data_df.shape[0]
valid_sample_num = 5000

print("[{}] positive sample in all data".format(pos_rate))


pos_sample_df = data_df[data_df['label']==1].sample(n=int(valid_sample_num * pos_rate), random_state=42)
neg_sample_df = data_df[data_df['label']==0].sample(n=int(valid_sample_num * (1-pos_rate)), random_state=42)

valid_df = pd.concat([pos_sample_df, neg_sample_df])
valid_df['id'] = valid_df['id'].astype(int)
valid_df['label'] = valid_df['label'].astype(int)
valid_df.to_csv("valid.tsv", sep="\t", index=False, encoding="utf-8", header=False)


train_df = data_df[~data_df.isin(valid_df)]
train_df.dropna(inplace=True)
train_df['id'] = train_df['id'].astype(int)
train_df['label'] = train_df['label'].astype(int)
train_df.to_csv("train.tsv", sep="\t", index=False, encoding="utf-8", header=False)


train_pos_rate = train_df[train_df["label"]==1].shape[0]*1.0 / train_df.shape[0]
valid_pos_rate = valid_df[valid_df["label"]==1].shape[0]*1.0 / valid_df.shape[0]

print("train pos rate : [{}], valid pos rate : [{}]".format(train_pos_rate, valid_pos_rate))

valid_df.drop(["label"], axis=1, inplace=True)
valid_df.to_csv("test.tsv", sep="\t", encoding="utf-8", header=False, index=False)


