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

jieba.load_userdict("./txt/dict.txt")

class BatchWrapper:
    def __init__(self, dl, iter_columns):
        self.dl, self.iter_columns = dl, iter_columns # we pass in the list of attributes for x &amp;amp;amp;amp;lt;g class="gr_ gr_3178 gr-alert gr_spell gr_inline_cards gr_disable_anim_appear ContextualSpelling ins-del" id="3178" data-gr-id="3178"&amp;amp;amp;amp;gt;and y&amp;amp;amp;amp;lt;/g&amp;amp;amp;amp;gt;

    def __iter__(self):
        for batch in self.dl:
            yield (getattr(batch, attr) for attr in self.iter_columns)
  
    def __len__(self):
        return len(self.dl)

def normalizeString(s):
    strips = re.sub('[<>]', "", string.punctuation)
    s = s.lower().strip(strips + '\n')
    # s = re.sub(r"<br />",r" ",s)
    # s = re.sub(r'(\W)(?=\1)', '', s)
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # s = re.sub(r"@.*$", r" ", s)

    return s


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = np.array(vec, dtype=float)
    return word_vec


def wordlist_to_matrix(pretrain_path, wordlist, device, dim=200):
    word_vec = load_glove_as_dict(filepath=pretrain_path)
    word_vec_list = []
    oov = 0
    for idx, word in enumerate(wordlist):
        try:
            if sys.version_info < (3, 4):
                vector = np.array(word_vec[word.encode('utf-8')], dtype=float).reshape(1,dim)
            else:
                vector = np.array(word_vec[word], dtype=float).reshape(1,dim)
        except:
            oov += 1
            # print(word)
            vector = np.random.rand(1, dim)
        word_vec_list.append(torch.from_numpy(vector))
    wordvec_matrix = torch.cat(word_vec_list)
    print("Load embedding finished.")
    print("Total words count: {}, oov count: {}.".format(wordvec_matrix.size()[0], oov))
    return wordvec_matrix if device == -1 else wordvec_matrix.to(device)


def normal_cut(text):
    simiwords_dict = json.load(io.open("../txt/simiwords.json", encoding='utf-8'))
    # print(simiwords_dict)
    cut_res = []
    rest_res = []
    for key, regex_list in simiwords_dict.items():
        for regex in regex_list:
            regex = "(.*){}(.*)".format(regex)
            # print(regex, text)
            search_res = re.match(regex, text)
            if search_res:
                single_rest = search_res.groups()
                text = " ".join(single_rest)
                # print(text)
                cut_res.append(key)
                # break
    return filter(lambda x: x.strip()!='', cut_res + jieba.lcut(text))


if __name__ == '__main__':
    # 用花呗刷卡可以吗  能用花呗扫一扫付款吗
    print("---")
    for item in (normal_cut("我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号 ")):
        # print(item.decode('utf-8'))
        print(item)
