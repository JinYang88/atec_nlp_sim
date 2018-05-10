import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import pandas as pd
import re
import numpy as np
# from utils import datahelper
import gensim
import jieba


jieba.load_userdict("../txt/dict.txt")

stopwords = set([])
with io.open("../txt/stopwords.txt", encoding='utf-8') as fr:
    for line in fr:
        stopwords.add(line.strip())


train_df = pd.read_csv("../../data/atec_nlp_sim_train.tsv", sep="\t", names=['id','text1', 'text2', 'label'], nrows=4)

sentences = []
valid_num = 0
for idx, line in train_df.iterrows():
    text1 = line['text1']
    text2 = line['text2']
    words = jieba.lcut(text1) + jieba.lcut(text2)
    words = [word for word in words if word not in stopwords]
    # for item in words:
    #     print(item)

    # sys.exit()
    sentences.append(words)


model = gensim.models.Word2Vec(sentences, min_count=3, size=300,
window=3, iter=100, sg=0)

with open("../../data/embedding.txt", 'w') as fw:
    for k in model.wv.vocab:
        fw.write("{} {}\n".format(k, ' '.join(model[k].astype(str))))