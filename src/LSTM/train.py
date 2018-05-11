# coding: utf-8
import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import re
import torch
from myenv.torchtext import data
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import io
import time
import utils.datahelper as datahelper
import jieba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

torch.manual_seed(42)

test_mode = 0  # 0 for train+test 1 for test
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16
embedding_dim = 300
hidden_dim = 400
out_dim = 1

epochs = 30
print_every = 500
bidirectional = True


print('Reading data..')
jieba.load_userdict("../txt/dict.txt")
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, use_vocab=True, tokenize=jieba.lcut)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)

train = data.TabularDataset(
        path='../../data/train.tsv', format='tsv',
        fields=[('Id', ID), ('Text1', TEXT), ('Text2', TEXT), ('Label', LABEL)], skip_header=True)
valid = data.TabularDataset(
        path='../../data/valid.tsv', format='tsv',
        fields=[('Id', ID), ('Text1', TEXT), ('Text2', TEXT), ('Label', LABEL)], skip_header=True)

TEXT.build_vocab(train, min_freq=3)
print('Building vocabulary Finished.')
word_matrix = datahelper.wordlist_to_matrix("../txt/embedding_300d.bin", TEXT.vocab.itos, device, embedding_dim)



train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text1) + len(x.Text2), shuffle=True, device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')


def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model.eval()
    res_list = []
    label_list = []
    loss = 0
    
    for text1, text2, label in data_dl:
        y_pred = model(text1, text2)
        loss += loss_func(y_pred, label).data.cpu()
        y_pred = y_pred.data.max(1)[1].cpu().numpy()
        res_list.extend(y_pred)
        label_list.extend(label.data.cpu().numpy())
        
    acc = accuracy_score(res_list, label_list)
    Precision = precision_score(res_list, label_list)
    Recall = recall_score(res_list, label_list)
    F1 = f1_score(res_list, label_list)

    with open("res_list.txt", 'w') as fw:
        for item in res_list:
            fw.write('{}\n'.format(item))
    
    return loss, (acc, Precision, Recall, F1)


class LSTM_angel(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size,wordvec_matrix, bidirectional):
        super(LSTM_angel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(2)
    
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(wordvec_matrix)
        self.word_embedding.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Linear(3, 200)
        self.dropout1 = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(200)
        self.linear2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(p=0.1)
        # self.batchnorm2 = nn.BatchNorm1d(200)
        self.linear3 = nn.Linear(200, 200)
        self.dropout3 = nn.Dropout(p=0.1)
        # self.batchnorm3 = nn.BatchNorm1d(200)
        self.linear4 = nn.Linear(200, 200)
        self.dropout4 = nn.Dropout(p=0.1)
        # self.batchnorm4 = nn.BatchNorm1d(200)
        self.linear5 = nn.Linear(200, 2)
        
    def forward(self, text1, text2, hidden_init=None) :
        text1_word_embedding = self.word_embedding(text1)
        text2_word_embedding = self.word_embedding(text2)
        text1_seq_embedding = self.lstm_embedding(self.lstm1, text1_word_embedding, hidden_init)
        text2_seq_embedding = self.lstm_embedding(self.lstm2, text2_word_embedding, hidden_init)
        dot_value = torch.bmm(text1_seq_embedding.view(text1.size()[0], 1, self.hidden_dim), text2_seq_embedding.view(text1.size()[0], self.hidden_dim, 1)).view(text1.size()[0], 1)
        dist_value = self.dist(text1_seq_embedding, text2_seq_embedding).view(text1.size()[0], 1)
        jaccard_value = self.jaccard(text1, text2)
        jaccard_value = jaccard_value.to(device)

        feature_vec = torch.cat((dot_value, dist_value, jaccard_value), dim=1)

#         feature_vec = torch.cat((text1_seq_embedding,text2_seq_embedding), dim=1)

        linearout_1 = self.linear1(feature_vec)
        linearout_1 = F.relu(linearout_1)
        linearout_1 = self.dropout1(linearout_1)
        # linearout_1 = self.batchnorm1(linearout_1)

        linearout_2 = self.linear2(linearout_1)
        linearout_2 = F.relu(linearout_2)
        linearout_2 = self.dropout2(linearout_2)
        # linearout_2 = self.batchnorm2(linearout_2)

        linearout_3 = self.linear3(linearout_2)
        linearout_3 = F.relu(linearout_3)
        linearout_3 = self.dropout3(linearout_3)
        # linearout_3 = self.batchnorm3(linearout_3)

        linearout_4 = self.linear4(linearout_3)
        linearout_4 = F.relu(linearout_4)
        linearout_4 = self.dropout4(linearout_4)
        # linearout_4 = self.batchnorm4(linearout_4)


        linearout_5 = self.linear5(linearout_4)

        return F.log_softmax(linearout_5, dim=1)


    def jaccard(self, list1, list2):
        reslist = []
        for idx in range(list1.size()[0]):
            set1 = set(list1[idx].data.cpu().numpy())
            set2 = set(list2[idx].data.cpu().numpy())
            jaccard = len(set1 & set2) * 1.0 / (len(set1) + len(set2) - len(set1 & set2))
            reslist.append(jaccard)
        return torch.FloatTensor(reslist).view(-1, 1)


    def lstm_embedding(self, lstm, word_embedding ,hidden_init):
        lstm_out,(lstm_h, lstm_c) = lstm(word_embedding, None)
        if self.bidirectional:
            seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        else:
            seq_embedding = lstm_h.squeeze(0)
        return seq_embedding


print('Initialing model..')
MODEL = LSTM_angel(len(TEXT.vocab), embedding_dim, hidden_dim, batch_size, word_matrix, bidirectional=bidirectional)
MODEL.to(device)

# for item in list(filter(lambda p: p.requires_grad, MODEL.parameters())):
#     print(item.size())
# print()
# sys.exit()

best_state = None
max_metric = 0

# Train
if not test_mode:
    loss_func = nn.NLLLoss()
    parameters = list(filter(lambda p: p.requires_grad, MODEL.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-4)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text1, text2, label in train_dl:
            MODEL.train()
            y_pred = MODEL(text1, text2)
            loss = loss_func(y_pred, label)
#             print(y_pred[0:3], label[0:3])
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                print("Evaluating....")
                loss, (acc, Precision, Recall, F1) = predict_on(MODEL, valid_dl, loss_func, device)
                batch_end = time.time()
                if F1 > max_metric:
                    best_state = MODEL.state_dict()
                    max_metric = F1
                    print("Saving model..")
                    torch.save(best_state, '../model_save/LSTM_angel.pth')           
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. F1 is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1, float(loss)))

