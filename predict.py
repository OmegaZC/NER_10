#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import pickle as pkl
import numpy as np
from importlib import import_module
import warnings

warnings.filterwarnings("ignore")

key = {
    0: '一般不满',
    1: '比较不满',
    2: '非常不满-渠道敏感',
    3: '非常不满-费用敏感',
    4: '非常不满-服务敏感'
}

class Predict:
    def __init__(self, model_name='TextRCNN', dataset='THUCNews', embedding='embedding_SougouNews.npz', use_word=False):
        if use_word:
            self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]  # char-level
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.pad_size = self.config.pad_size
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))

    def build_predict_text(self, texts):
        words_lines = []
        seq_lens = []
        for text in texts:
            words_line = []
            token = self.tokenizer(text)
            seq_len = len(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend(['<PAD>'] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get('<UNK>')))
            words_lines.append(words_line)
            seq_lens.append(seq_len)

        return torch.LongTensor(words_lines), torch.LongTensor(seq_lens)

    def predict(self, query):
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs)
        return key[int(num)]

    def predict_list(self, querys):
        # 返回预测的索引
        data = self.build_predict_text(querys)
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs, dim=1)
            pred = [key[index] for index in list(np.array(num))]
        return pred


if __name__ == "__main__":
    import os
    pred = Predict('TextRCNN')
    # 预测一条
    # query = "【费用争议帐期】*【争议金额(元)】* 客户就前工单*08540 问题再次来电反映，表示对之前处理结果不满意，要求再次联系处理，请协助，谢谢"
    # print(pred.predict(query))
    # 预测一个列表
    # querys = ["学费太贵怎么办？", "金融怎么样"]
    # print(pred.predict_list(querys))
    querys = []
    true_list = []
    text_data_path = os.path.join(os.getcwd(),
                                  '../../../Project/Classification of Telecom Complaints/Chinese-Texxt-Classification-pytorch-all_2.0/THUCNews/data/test.txt')
    for line in open(text_data_path , 'r' , encoding='utf-8'):
        line = line.strip('\n')
        text, cls = line.split('\t')
        querys.append(text)
        true_list.append(cls)

    #
    # print(pred.predict_list(querys[-100:]))
    # print([key[int(cls)] for cls in true_list[-100:]])

    pred_list = pred.predict_list(querys[-100:])
    true_list = [key[int(cls)] for cls in true_list[-100:]]


    for i in range(100):
        print(f'pred:{pred_list[i]} , \t true:{true_list[i]}')







