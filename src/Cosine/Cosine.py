# coding: utf-8
import sys
if sys.version_info < (3, 4):
    reload(sys)
    sys.setdefaultencoding('utf-8')
sys.path.append("../")
import utils.datahelper as datahelper
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import io
import jieba 

SUBMIT = False


def seq2vec(text, word_vec, stopwords):
    words = [word.strip() for word in jieba.lcut(text) if word not in stopwords]
    vec = np.zeros(300)
    in_num = 0
    for word in words:
        word =  word.encode("utf-8")
        if word in word_vec:
            # print(word_vec[word])
            vec += word_vec[word]
            in_num += 1
    return vec / in_num if in_num > 0 else vec

def cosine_simi(vec1, vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2)))  

def o_dis(vec1, vec2):
    return np.linalg.norm(vec1 - vec2) 

if __name__ == '__main__':
    if SUBMIT:
        infile = sys.argv[1]
        outfile = sys.argv[2]
    else:
        infile = "../../data/train.tsv"
        outfile = "../../data/predict.tsv"

    train_list = []
    stopwords = set([])
    simiwords = {}
    jieba.load_userdict("../txt/dict.txt")

    word_vec = datahelper.load_glove_as_dict("../../data/embedding_300d.txt")

    # print(word_vec.keys())
    with io.open("../txt/stopwords.txt", encoding='utf-8') as fr:
        for line in fr:
            stopwords.add(line.strip())


    data_df = pd.read_csv(infile, sep="\t", names=["id", "text1", "text2", "label"])


    data_df["text1_cut"] = data_df['text1'].map(jieba.lcut)
    data_df["text2_cut"] = data_df['text2'].map(jieba.lcut)

    if not SUBMIT:
        pred = []
        simi = []
        for idx, row in data_df.iterrows():
            vec1 = seq2vec(row["text1"], word_vec, stopwords)
            vec2 = seq2vec(row["text2"], word_vec, stopwords)
            simi.append(o_dis(vec1, vec2))
        simi= np.array(simi)
        data_df['pred'] = (simi - min(simi)) / max(simi)


        for threshold in np.arange(0, 1, 0.01):
            pred = data_df['pred'].map(lambda x: 1 if x > threshold else 0)
            acc = round(accuracy_score(pred, data_df['label']), 2)
            f1 = round(f1_score(pred, data_df['label']), 2)
            avg = (acc + f1) / 2  
            print("Using [{}] as threshold, acc is [{}], f1 is [{}], avg if [{}]".format(threshold, acc, f1, avg))

