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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32
embedding_dim = 300
hidden_dim = 400

epochs = 30
print_every = 300
bidirectional = True

print('Reading data..')
jieba.load_userdict("../txt/dict.txt")
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=12, batch_first=True, use_vocab=True, tokenize=jieba.lcut)
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

train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: max(len(x.Text1),len(x.Text2)), shuffle=True, device=device, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)

train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')

def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model.eval()
    save_list = []
    res_list = []
    label_list = []
    loss = 0
    
    for text1, text2, label in data_dl:
        y_pred = model(text1, text2)
        save_list.extend(y_pred)
        loss += loss_func(y_pred, label).data.cpu()
        y_pred = y_pred.data.max(1)[1].cpu().numpy()
        res_list.extend(y_pred)
        label_list.extend(label.data.cpu().numpy())
        
    acc = accuracy_score(res_list, label_list)
    Precision = precision_score(res_list, label_list)
    Recall = recall_score(res_list, label_list)
    F1 = f1_score(res_list, label_list)

    with open("wide_deep_res.txt", 'w') as fw:
        for item in save_list:
            fw.write('{},{}\n'.format(item[0],item[1]))
    
    return loss, (acc, Precision, Recall, F1)


class wide_deep(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size,wordvec_matrix, bidirectional):
        super(wide_deep, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(2)
    
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(wordvec_matrix)
        self.word_embedding.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        
        self.mp = nn.MaxPool1d(hidden_dim, stride=1)
        
        self.deep1 = nn.Linear(2 * hidden_dim, 200)
        self.dropout1 = nn.Dropout(p=0.1)
        self.deep2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(p=0.1)
        self.deep3 = nn.Linear(200, 200)
        self.dropout3 = nn.Dropout(p=0.1)
        self.deep4 = nn.Linear(200, 200)
        self.dropout4 = nn.Dropout(p=0.1)
        self.deep5 = nn.Linear(200, 200)
        self.dropout5 = nn.Dropout(p=0.1)
        
        self.merge_layer = nn.Linear(200 + 5, 2)
        
    def forward(self, text1, text2, hidden_init=None) :
        text1_word_embedding = self.word_embedding(text1)
        text2_word_embedding = self.word_embedding(text2)
        text1_seq_embedding, text1_max_embedding = self.lstm_embedding(self.lstm1, text1_word_embedding, hidden_init)
        text2_seq_embedding, text2_max_embedding = self.lstm_embedding(self.lstm2, text2_word_embedding, hidden_init)

        dot_value = self.cal_dot(text1_seq_embedding, text2_seq_embedding)
        dist_value = self.cal_dist(text1_seq_embedding, text2_seq_embedding)
#         dot_value = torch.bmm(text1_seq_embedding.view(text1.size()[0], 1, self.hidden_dim), text2_seq_embedding.view(text1.size()[0], self.hidden_dim, 1)).view(text1.size()[0], 1)
#         dist_value = self.dist(text1_seq_embedding, text2_seq_embedding).view(text1.size()[0], 1)

#         print(text1.size(), text2.size())
#         print(text1_max_embedding.size())
#         print("--")
#         print(text2_max_embedding.size())
    
        max_dot_value = self.cal_dot(text1_max_embedding, text2_max_embedding)
        max_dist_value = self.cal_dist(text1_max_embedding, text2_max_embedding)
        
        
        jaccard_value = self.jaccard(text1, text2)
        jaccard_value = jaccard_value.to(device)
        
        deep_feature = torch.cat((text1_seq_embedding,text2_seq_embedding), dim=1)
        wide_feature = torch.cat((dot_value, dist_value, max_dot_value, max_dist_value, jaccard_value), dim=1) 
    
        merged = self.deep1(deep_feature)
        merged = F.relu(merged)
        merged = self.dropout1(merged)
        # merged = self.batchnorm1(merged)

        merged = self.deep2(merged)
        merged = F.relu(merged)
        merged = self.dropout2(merged)
        # merged = self.batchnorm2(merged)

        merged = self.deep3(merged)
        merged = F.relu(merged)
        merged = self.dropout3(merged)
        # merge = self.batchnorm3(merge)

        merged = self.deep4(merged)
        merged = F.relu(merged)
        merged = self.dropout4(merged)
        # merge = self.batchnorm4(merge)
        merged_deep = self.deep5(merged)

#         print(merged_deep)
#         print(wide_feature)
        
        deep_wide_feature = torch.cat((merged_deep, wide_feature), dim=1)
        
#         print(deep_wide_feature)
        
        output = self.merge_layer(deep_wide_feature)

        return F.log_softmax(output, dim=1)

    def cal_dot(self, embedding1, embedding2):
        return torch.bmm(embedding1.view(embedding1.size()[0], 1, embedding1.size()[1]), embedding2.view(embedding1.size()[0], embedding1.size()[1], 1)).view(embedding1.size()[0], 1)

    def cal_dist(self, embeddig1, embedding2):
        return self.dist(embeddig1, embedding2).view(embeddig1.size()[0], 1)
    
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
        return seq_embedding, self.mp(lstm_out).view(word_embedding.size()[0], -1)

print('Initialing model..')
MODEL = wide_deep(len(TEXT.vocab), embedding_dim, hidden_dim, batch_size, word_matrix, bidirectional=bidirectional)
MODEL.to(device)

# for item in list(filter(lambda p: p.requires_grad, MODEL.parameters())):
#     print(item.size())
# print()
# sys.exit()

best_state = None
max_metric = 0

# Train
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
                torch.save(best_state, '../model_save/wide_deep.pth')           
            print('{}/{} batch, {}/{} epoch. Time: {}s. F1: {}, precision: {}, recall: {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), F1, Precision, Recall, float(loss)))


