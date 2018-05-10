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

    sys.exit()

    ground_truth = []
    with io.open(infile, encoding='utf-8') as fr:
        for line in fr:
            line = re.split("\t", line)
            # fw.write("{}\t{}\t{}\n".format(line[0], line[1], line[2]))
            line[1] = preprocess(line[1], stopwords, simiwords)
            line[2] = preprocess(line[2], stopwords, simiwords)
            # print(line[1], line[2])
            train_list.append(line)
            if not SUBMIT:
                ground_truth.append(int(line[3]))

    if SUBMIT:
        threshold = 0.45
        predict = []
        jaccard_info = []
        for sample in train_list:
            if cal_j(sample[1], sample[2]) > threshold:
                predict.append(1)
            else:
                predict.append(0)
        with open(outfile, 'w') as fw:
            for idx, line in enumerate(predict):
                fw.write("{}\t{}\n".format(train_list[idx][0], line))


    else:
        for threshold in np.arange(0,1,0.05):
            predict = []
            jaccard_info = []
            for sample in train_list:
                # jaccard_info.append([sample[1], sample[2], sample[3], cal_j(sample[1], sample[2])])
                if cal_j(sample[1], sample[2]) > threshold:
                    predict.append(1)
                else:
                    predict.append(0)
            # save_df(jaccard_info)
            acc = round(accuracy_score(predict, ground_truth), 2)
            f1 = round(f1_score(predict, ground_truth), 2)
            avg = (acc + f1) / 2  
            print("Using [{}] as threshold, acc is [{}], f1 is [{}], avg if [{}]".format(threshold, acc, f1, avg))
