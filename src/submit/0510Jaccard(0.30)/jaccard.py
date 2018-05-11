SUBMIT = True

import io
import sys
import re
import jieba
if not SUBMIT:
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def save_df(list_of_list, name="save.csv"):
    if not SUBMIT:
        import pandas as pd
        pd.DataFrame(list_of_list, columns=['text1', 'text2', 'label', 'jaccard']).to_csv(name, index=False, encoding='gb18030')


def cal_j(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    avg_len = (len(set1) + len(set2)) / 2
    min_len = min(len(set1),len(set2))
    return len(set1 & set2) * 1.0 / (len(set1) + len(set2) - len(set1 & set2))
    # return len(set1 & set2) * 1.0 / min_len

def wordcut(line):
    return [i for i in line]

def preprocess(line, stopwords, simiwords):
    # for word, subword in simiwords.iteritems():
    #     if word in line:
    #         line = re.sub(word, subword, line)

    # wordlist = [word for word in wordcut(line.strip()) if word not in stopwords]
    wordlist = [word for word in jieba.lcut(line.strip()) if word not in stopwords]
    
    return wordlist

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
    jieba.load_userdict("./txt/dict.txt")


    with io.open("./txt/stopwords.txt", encoding='utf-8') as fr:
        for line in fr:
            stopwords.add(line.strip())

    # with io.open("simiwords.txt", encoding='utf-8') as fr:
    #     for line in fr:
    #         words = re.split(",", line.strip())
    #         simiwords[words[0]] = words[1]


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
                jaccard_info.append([sample[1], sample[2], sample[3], cal_j(sample[1], sample[2])])
                if cal_j(sample[1], sample[2]) > threshold:
                    predict.append(1)
                else:
                    predict.append(0)
            acc = round(accuracy_score(predict, ground_truth), 2)
            f1 = round(f1_score(predict, ground_truth), 2)
            avg = (acc + f1) / 2  
            print("Using [{}] as threshold, acc is [{}], f1 is [{}], avg if [{}]".format(threshold, acc, f1, avg))

        save_df(jaccard_info)

