# coding=utf-8
import sys

if sys.version_info < (3, 4):
    reload(sys)
    sys.setdefaultencoding('utf-8')

import torch
import re
import numpy as np
import string
import json
import io
import jieba
from collections import defaultdict


def word_diff(cut1, cut2):
    return set(cut1) - set(cut2)

def set2string(word_set, word_count):
    res = ""
    for word in word_set:
        res += "{}:{}".format(word, word_count[word])
        res += " "
    return res

jieba.load_userdict("../txt/dict.txt")
data_df = pd.read_csv("../../data/atec_nlp_sim_train.tsv", sep="\t", names=['id','text1', 'text2', 'label'])

data_df["text1_cut"] = data_df['text1'].map(jieba.lcut)
data_df["text2_cut"] = data_df['text2'].map(jieba.lcut)

word_diff_list = []
diff_1 = defaultdict(int)
diff_2 = defaultdict(int)

for idx, line in data_df.iterrows():
    diff = word_diff(x['text1_cut'], x['text2_cut'])
    word_diff_list.append(diff)
    for word in diff:
        if line['label'] == 1:
            diff_1[word] += 1
        else:
            diff_2[word] += 1

word_diff_count_list = []
for idx, line in data_df.iterrows():

    word_diff_count_list.append(set2string)
