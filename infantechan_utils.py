"""仅供测试使用"""
"""
拆分数据集
将原始语料按照特定比例拆分成训练集、验证集、测试集(train.txt, dev.text, test.txt)
参数：
    data_path:总语料路径
    save_path:拆分后语料路径
    split_rate:拆分比率
return:
    if save_path is not None:None=>将按照特定比例拆分成训练集、验证集、测试集保存在指定路径
    else:tag_weight=>不均衡样本重采样比率
"""

from collections import defaultdict
import os
import numpy as np


def split_data(data_path, split_rate, save_path=None):
    data_path = os.path.join(os.getcwd(), data_path)
    if save_path is not None: save_path = os.path.join(os.getcwd(), save_path)
    all_data = defaultdict(list)
    for line in open(data_path, 'r', encoding='utf-8'):
        line = line.strip('\n')
        text, cls = line.split('\t')
        all_data[int(float(cls))].append(text + '\t' + str(int(float(cls)) - 1))
    # 样本数量分布字典
    data_describe = {k: len(all_data[k]) for k in all_data}
    # 字典排序
    data_describe_sorted = {k - 1: data_describe[k] for k in sorted(data_describe)}
    print(data_describe_sorted)

    all_data_len = 0
    for k, v in data_describe.items(): all_data_len += v
    tag_weight = 1 - np.array([data_describe[t] / all_data_len for t in range(1, max([k for k in data_describe]) + 1)])
    print(tag_weight)
    train, dev, test = [], [], []
    # 按照样本分布来拆分train，dev，test
    for k in all_data:
        np.random.shuffle(all_data[k])
        train_size, dev_size, test_size = [int(len(all_data[k]) * rate) for rate in split_rate]
        train += [d for i, d in enumerate(all_data[k]) if i <= train_size]
        dev += [d for i, d in enumerate(all_data[k]) if train_size < i <= train_size + dev_size]
        test += [d for i, d in enumerate(all_data[k]) if i > train_size + dev_size]

    # 将每个数据集打乱
    for da in [train, dev, test]:
        np.random.shuffle(da)

    if save_path is not None:
        def save_data(save_data, save_path):
            with open(save_path, 'w', encoding='utf-8') as f:
                for line in save_data:
                    f.write(line + '\n')

        save_data(train, os.path.join(save_path, 'train.txt'))
        save_data(dev, os.path.join(save_path, 'dev.txt'))
        save_data(test, os.path.join(save_path, 'test.txt'))
    else:
        return tag_weight
    pass


"""数据剥离"""
def remove_label(data_path , save_path):
    data_path = os.path.join(os.getcwd(), data_path)
    save_path = os.path.join(os.getcwd(), save_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        for line in open(data_path, 'r', encoding='utf-8'):
            line = line.strip('\n')
            text, cls = line.split('\t')
            f.write(text + '\n')
    pass
"""
数据增强

"""

from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch import nn

"""
模型参数
"""

"""1. 构建数据增强bert模型"""


class DaEnhanModel(BertPreTrainedModel):
    # 初始化
    def __init__(self, config):
        super().__init__(config)
        self.num_words = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        # 初始化参数的方法
        self.init_weights()
        # MLM head is not trained
        # 把BertOnlyMLMHead里所有的参数不进行梯度更新
        for param in self.cls.parameters():
            param.requires_grad = False

    # 前向运算
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        # mlm
        logits = self.cls(last_hidden_states)
        return logits


"""2. 构建分类词汇表"""
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

"""2.1 创建label_name_data"""
"""2.1.1 label_name_in_doc:在单条语料中查找标签"""

"""
label_name_in_doc:在现有文本中查找标签名称并标记索引，用[MASK]替换超出词典范围的标签名称
参数：
    opt:全局参数配置项
    doc:单条语料str
return:
    找到了匹配的标签:tuple(' '.join(new_doc) , label_idx)
        ' '.join(new_doc):  原始语料str,每个字词中间空格分隔，但opt.tokenizer.get_vocab()中没有的词由[UNK]表示
        label_idx:          替换后的文本和对应位置的索引 
    没找到匹配的标签:None

"""


def label_name_in_doc(opt, doc):
    # 拆分文本
    """
    opt.tokenizer.tokenize():分词
    "Wall St. Bears Claw Back Into..." => ['wall', 'st', '.', 'bears', 'claw', 'back', 'into',...]
    参数:
        doc:单条语料
    return:
        doc:拆分后的语料
    """
    doc = opt.tokenizer.tokenize(doc)
    # 创建一个和语料长度一致，全部值都是-1的张量序列作为label
    label_idx = -1 * torch.ones(opt.max_len, dtype=torch.long)
    new_doc = []
    wordpcs = []
    idx = 1  # 由于[CLS] token，所以索引从1开始
    for i, wordpc in enumerate(doc):
        # 添加词汇和子词到wordpcs
        '''
        当词以##开头时，则为字词（前缀、后缀等），遇到字词时，所有字词将会放到wordpcs中，
        直到下一个不是字词时，将这个字词完整拼接。  => word = ''.join(wordpcs)
        '''
        wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
        if idx >= opt.max_len - 1:  # 最后一个索引应当是 [SEP]
            break

        if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
            word = ''.join(wordpcs)
            # 如果词汇出现在类别标签中，label序列中对应位置标记为class id
            if word in opt.label2class:
                label_idx[idx] = opt.label2class[word]
                # 用[MASK]标记替换标记化器词汇表中没有的标签名
                if word not in opt.tokenizer.get_vocab():
                    wordpcs = [opt.tokenizer.mask_token]
            new_word = ''.join(wordpcs)
            # opt.tokenizer.unk_token:[UNK]
            if new_word != opt.tokenizer.unk_token:
                idx += len(wordpcs)
                new_doc.append(new_word)
            wordpcs = []
    # 如果找到了匹配的标签，返回替换后的文本和对应位置的索引，否则返回None
    # (label_idx >= 0).any()：但凡有一个大于零的数出现，则会进入条件
    if (label_idx >= 0).any():
        return ' '.join(new_doc), label_idx
    else:
        return None


"""2.1.2 查找包含标签名的语料"""

"""
label_name_occurrence:查找包含标签名的语料
参数：
    opt:全局参数配置项
    docs:所有语料大列表，列表每个元素为一条语料
return:
    input_ids_with_label_name:Tensor[7810,200]
        tensor([[101,2974,2194, ...,0,0,0],[101,11991,4486,...,0,0,0], ...
    attention_masks_with_label_name:Tensor[7810,200]
        tensor([[1, 1, 1,  ..., 0, 0, 0],...,]])
    label_name_idx:Tensor[7810,200]不是标签名的词为-1,是标签的词为标签类别号
        tensor([[-1,  3, -1,  ..., -1, -1, -1],...,]])
"""
from tqdm import tqdm

def label_name_occurrence(opt, docs):
    # 含有标签的语料列表
    text_with_label = []
    # 语料标签列表
    label_name_idx = []
    for doc in tqdm(docs):
        result = label_name_in_doc(opt, doc)
        # 当result不为None时（即该条语料含有标签时）
        if result is not None:
            text_with_label.append(result[0])
            # result[1].unsqueeze(0)升维    Tensor(200,) => Tensor(1,200)
            label_name_idx.append(result[1].unsqueeze(0))
    # 如果有符合条件的文本，就把文本转换为模型输入用的tensor并返回
    # 如果没有，就返回一些值全为1的张量
    if len(text_with_label) > 0:
        """
        opt.tokenizer():BertTokenizer.from_pretrained(options.bert_model, model_max_length=options.max_len)
        参数：
            text_with_label:一维list，每个元素是一条含有标签词的语料(str)
            add_special_tokens=True 转换后的内容中添加[CLS]和[SEP]的token id
            max_length=self.max_len 指定文本最大长度
            padding='max_length' padding策略，依据指定的max_length来填充
            return_attention_mask=True 返回结果中是否包含attention_mask
            truncation=True 超出max_length的文本自动截断
            return_tensors='pt' 返回pytorch张量
        return:
            input_ids:Tensor(len(text_with_label)(7810),max_length(200))
                tensor([[101,2974,2194, ...,0,0,0],[101,11991,4486,...,0,0,0], ...
            token_type_ids:Tensor(len(text_with_label)(7810),max_length(200))
                tensor([[1, 1, 1,  ..., 0, 0, 0],...,]])
            attention_mask:Tensor(len(text_with_label)(7810),max_length(200))不是标签名的词为-1,是标签的词为标签类别号
                tensor([[-1,  3, -1,  ..., -1, -1, -1],...,]])
        """
        encoded_dict = opt.tokenizer(text_with_label, add_special_tokens=True, max_length=opt.max_len,
                                     padding='max_length', return_attention_mask=True, truncation=True,
                                     return_tensors='pt')
        input_ids_with_label_name = encoded_dict['input_ids']
        attention_masks_with_label_name = encoded_dict['attention_mask']
        # label_name_idx是list，每个元素是一个Tensor(1,max_length),cat后是Tensor(len(text_with_label)(7810),max_length(200))
        label_name_idx = torch.cat(label_name_idx, dim=0)
    else:
        input_ids_with_label_name = torch.ones(0, opt.max_len, dtype=torch.long)
        attention_masks_with_label_name = torch.ones(0, opt.max_len, dtype=torch.long)
        label_name_idx = torch.ones(0, opt.max_len, dtype=torch.long)
    return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx
    pass


"""2.1.3 根据提供的语料生成,用于创建“词汇近义词表”的模型tensor"""

"""
create_label_name_dataset:根据提供的语料生成,用于创建“分类词汇表”的模型tensor
参数：
    opt:全局参数配置项
    text_file:用于训练的未标记文本语料库(每行一条语料)路径(文件名)
    loader_name:保存筛选分类词汇表使用的张量数据文件路径(文件名)
return:
    label_name_data:数据集合字典{'input_ids':tensor() , 'attention_masks':tensor() , 'labels':tensor()}
{'input_ids': tensor([[  101,  2974,  2194,  ...,     0,     0,     0],
        [  101, 11991,  4486,  ...,     0,     0,     0],
        [  101,  2091, 23393,  ...,     0,     0,     0],
        ...,
        [  101,  2137,  4671,  ...,     0,     0,     0],
        [  101, 15340,  3550,  ...,     0,     0,     0],
        [  101,  6646, 20996,  ...,     0,     0,     0]]), 
'attention_masks': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), 
'labels': tensor([[-1,  3, -1,  ..., -1, -1, -1],
        [-1, -1, -1,  ..., -1, -1, -1],
        [-1, -1, -1,  ..., -1, -1, -1],
        ...,
        [-1, -1, -1,  ..., -1, -1, -1],
        [-1, -1, -1,  ..., -1, -1, -1],
        [-1, -1, -1,  ..., -1, -1, -1]])}
"""
def create_label_name_dataset(opt, file, loader_name):
    # 拼接文件路径
    loader_file = os.path.join(opt.dataset_dir, loader_name)
    # 1.尝试加载存盘文件
    # 1.1 有存盘文件，直接加载
    if os.path.exists(loader_file):
        print(f"从 {loader_file} 文件中加载包含标签名的语料张量")
        label_name_data = torch.load(loader_file)
    # 1.2 无分盘文件
    else:
        print(f"从 {os.path.join(opt.dataset_dir, file)} 文件中读取语料")
        # 2.1 读取语料
        corpus = open(os.path.join(opt.dataset_dir, file), encoding="utf-8")
        # 逐行读取，每条语料去除前后空格后作为docs列表的一个元素
        docs = [doc.strip() for doc in corpus.readlines()]
        print("检索包含类别词汇的语料")
        # 2.2 拿出所有有标签的语料的元素
        input_ids_with_label_name, \
        attention_masks_with_label_name, \
        label_name_idx = label_name_occurrence(opt, docs)
        # 2.3 检查所有语料中是否存在标签名，如果一个没有则终止
        assert len(input_ids_with_label_name) > 0, "语料中没有发现匹配的标签名!"
        label_name_data = {
            "input_ids": input_ids_with_label_name,
            "attention_masks": attention_masks_with_label_name,
            "labels": label_name_idx
        }
        # 2.4 数据存盘
        print(f"包含标签名的语料张量存入文件 {loader_file}")
        torch.save(label_name_data, loader_file)
    return label_name_data
    pass


"""2.2 构建类别词汇表"""
import os
import torch
from tqdm import tqdm
# import numpy as np
# from nltk.corpus import stopwords
from collections import defaultdict
"""2.2.1 根据已填充的数据集字典，创建并返回DataLoader"""

"""
make_dataloader:根据已填充的数据集字典，创建并返回DataLoader
参数:
    data_dict:数据集合字典{'input_ids':tensor() , 'attention_masks':tensor() , 'labels':tensor()}(label_name_data)
    batch_size:批次大小
return:
    dataset_loader:数据加载器
        如果data_dict中存在'label'则为训练,dataloader则需要放"input_ids"、"attention_masks"、"label"
        如果data_dict中不存在'label'则为预测，dataloader则需要放"input_ids"、"attention_masks"
"""
def make_dataloader(data_dict, batch_size):
    if "labels" in data_dict:
        """
        TensorDataset:可以用来对tensor进行打包
        """
        dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
    else:
        dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset_loader


"""2.2.2 构建词汇近义词表"""
"""
- 1. 针对每个标签词汇位置通过MLM预测
- 2. 筛选出TOP50个最相似的词，对于每一个类别的标签名称(Label Name)根据词频大小、结合停用词共筛选出TOP100个词，最终构建词汇近义词表
filter_keywords:过滤掉停用词和多重分类词
    根据词频筛选各分类前100个token
    删除不是纯字符组成的token
    删除长度为1的token
    删除标记为stopword的token
    各分类中词汇保证唯一
parameter:
    opt:全局配置参数
    category_words_freq:各分类token索引出现的次数
    category_vocab_size:词频筛选各分类前个数
return:
    category_vocab:存储结构 {category_id:[token_id,...],...}
        {0: array([ 4331,  2576,  8801,  2231,  3864,  3761,  7072,  3537,  8677,... 
"""
from tqdm import tqdm
def filter_keywords(opt, category_words_freq, category_vocab_size=100):
    # 每个token在语料中的类别列表
    # all_words:defaultdict(<class 'list'>, {4331: [0], 2576: [0], 8801: [0], ... , 2449: [0, 1, 2, 3], 2373: [0, 1, 2, 3]...
    all_words = defaultdict(list)
    # 筛选后的分类token字典
    # sorted_dicts:{0: {4331: 228.0, 2576: 226.0, 8801: 205.0, 2231: 200.0, 3864: 196.0,...}
    sorted_dicts = {}
    # 每个分类中的token计数排序，只保留前100(category_vocab_size)个  {0:{token_id:counts,...},...}
    for i, cat_dict in category_words_freq.items():
        """1. 先进行排序并截取前100(category_vocab_size)个"""
        sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
        # 将本次类别筛选出来的前100个词字典{词在vocab中的id号：在语料中出现总次数}保存成sorted_dicts的类别号对应的value
        sorted_dicts[i] = sorted_dict
        for word_id in sorted_dict:
            """2. 制作all_words"""
            # 将每个词出现过的类别号保存到all_words{词在vocab中的id：[对应词出现过的类别号(没有则为空)]}
            all_words[word_id].append(i)
    # 查找在多个分类中出现的token
    """3. 制作一词多类词典"""
    repeat_words = []
    for word_id in all_words:
        if len(all_words[word_id]) > 1:
            repeat_words.append(word_id)
    # 提取每个分类中的token_id
    category_vocab = {}
    for i, sorted_dict in sorted_dicts.items():
        category_vocab[i] = np.array(list(sorted_dict.keys()))
    # 中文停用词
    stopwords_vocab = opt.stopword
    pbar = tqdm(category_vocab.items())
    for i, word_list in pbar:
        delete_idx = []
        for j, word_id in enumerate(word_list):
            # opt.inv_vocab:反向词典 给id换回真实字词
            word = opt.inv_vocab[word_id]
            # 不删除分类标签
            if word in opt.label_name_dict[i]:
                continue
            """4. 删除无用token"""
            # 删除不是纯字母的token，长度为1的token，stopword的token，在多个分类中匹配的词汇
            if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                delete_idx.append(j)
        # 删除后的词汇
        category_vocab[i] = np.delete(category_vocab[i], delete_idx)

    return category_vocab


"""
create_category_vocabulary:构建类别词汇表
parameter:
    opt:全局配置参数
    model:模型
    label_name_data:数据集合字典{'input_ids':tensor() , 'attention_masks':tensor() , 'labels':tensor()}
    loader_name:保存vocab的位置
    top_pred_num:提取每个分类token值最大的前几个索引个数
    category_vocab_size:词频筛选各分类前个数
return:
    category_vocab
"""


def create_category_vocabulary(opt, model, label_name_data, loader_name, top_pred_num=40, category_vocab_size=50):
    # 尝试从文件中直接加载分类词汇表
    loader_file = os.path.join(opt.dataset_dir, loader_name)
    if os.path.exists(loader_file):
        print(f"从 {loader_file} 文件中加载分类词汇表")
        category_vocab = torch.load(loader_file)
    else:
        print("构建分类词汇表")

        model.eval()
        # ["input_ids","attention_masks","labels"]
        label_name_dataset_loader = make_dataloader(label_name_data, opt.eval_batch_size)
        # 统计分类标签的出现频率 {0:{},1:{},2:{},3:{}}
        category_words_freq = {i: defaultdict(float) for i in range(opt.num_words)}

        for batch in tqdm(label_name_dataset_loader):
            with torch.no_grad():
                input_ids = batch[0].to(opt.device)
                input_mask = batch[1].to(opt.device)
                label_pos = batch[2].to(opt.device)
                match_idx = label_pos >= 0
                # 进行MLM推理
                # predictions[batch_size, max_seq_len, vocab_len]
                predictions = model(input_ids,
                                    token_type_ids=None,
                                    attention_mask=input_mask)
                # 过滤有分类值的logits，提取每个分类token值最大的前50个索引
                """
                torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor):筛选前k个
                参数:
                    input   -> 输入tensor
                    k       -> 指明是得到前k个数据以及其index
                    dim     -> 指定在哪个维度上排序， 默认是tensor最后一个维度
                    sorted  -> 是否排序
                    largest -> False表示返回第k个最小值
                return:the output tuple of (Tensor, LongTensor) that can be optionally given to be used as output buffers
                    [0]values=tensor():前k的概率logits值
                    [1]indices=tensor():前k个子在vocab中的id号
                        在predictions[batch_size, max_seq_len, vocab_len]的vocab_len这一维度的index号(即在vocab中的idx号)
                """
                # _(得分值),sorted_res(index号):[一个批次内是类别词的个数,top_pred_num]
                _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                # 提取出语料中所有的类别词的类别号
                label_idx = label_pos[match_idx]
                # 遍历拿出每个类别词的前50候选词
                # enumerate(sorted_res)会50(top_pred_num)个50个拿
                for i, word_list in enumerate(sorted_res):
                    for j, word_id in enumerate(word_list):
                        # 分别统计各分类token索引出现的次数
                        category_words_freq[label_idx[i].item()][word_id.item()] += 1

        # 过滤掉停用词和属于多分类的词汇后结果存入self.category_vocab
        # 存储结构 {category_id:[token_id,...],...}
        category_vocab = filter_keywords(opt, category_words_freq, category_vocab_size)

        # 保存到文件
        torch.save(category_vocab, loader_file)
    vocab_dict = {}
    for i, cat_vocab in category_vocab.items():
        # print(f"Class {i} category vocabulary: {[opt.inv_vocab[w] for w in cat_vocab]}\n")
        # print(f"\t{opt.label_name_dict[i][0]} 在训练集语料中的近义词: {[opt.inv_vocab[w] for w in cat_vocab]}\n")
        if len([opt.inv_vocab[w] for w in cat_vocab]) >= 1:
            print(f"\t{opt.label_name_dict[i if i == 0 else i-1][0]} 在训练集语料中的近义词: {[opt.inv_vocab[w] for w in cat_vocab]}\n")
            vocab_dict[opt.label_name_dict[i if i == 0 else i-1][0]] = [opt.inv_vocab[w] for w in cat_vocab]
    return category_vocab , vocab_dict


"""3 运行"""
"""3.1 全局配置参数"""
"""3.1.1 从文件中读取标签名"""
"""
read_label_names-从文件中读取标签名
参数：
    opt:全局参数配置项
return:
    label_name_dict:类别号对应标签字典
        {0: ['politics'], 1: ['sports', 'basketball', 'football'], 2: ['business'], 3: ['technology']}
    label2class:标签对应类别号字典
        {'politics': 0, 'sports': 1, 'basketball': 1, 'football': 1, 'business': 2, 'technology': 3}
"""
def read_label_names(opt):
    # label_name_file:类别词    （politics \t sports \t business \t technology）
    label_name_file = open(os.path.join(opt.dataset_dir, opt.label_names_file) , 'r' , encoding='utf-8')
    # label_names:['politics\n', 'sports\n', 'business\n', 'technology\n']
    label_names = label_name_file.readlines()
    # 读取每个标签中的单词list，以行号作为类别id存入字典 {0:[word1,word2,...], 1:[word1,word2,...], ...}
    label_name_dict = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in enumerate(label_names)}
    print(f"每个类别使用的标签名称分别是: {label_name_dict}")
    # 所有标签类别映射字典
    label2class = {}
    for class_idx in label_name_dict:
        for word in label_name_dict[class_idx]:
            # assert标记用作标签名称的单词
            assert word not in label2class, f"\"{word}\" 作为标签名，被应用在多分类任务中"
            # 类别->类别id
            label2class[word] = class_idx
    return label_name_dict, label2class

"""3.1.2 全局配置参数"""
import torch
import argparse
from addict import Dict
# from process import read_label_names
from transformers import BertTokenizer
from wobert import WoBertTokenizer
class ArgsParse:

    @staticmethod
    def parse():
        # 命令行参数解析器
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # 数据集所在目录
        parser.add_argument('--dataset_dir', default='THUCNews/lotclass_data', help='语料数据集目录')
        parser.add_argument('--train_file', default='train_no_label.txt', help='用于训练的未标记文本语料库(在数据集目录下)；每行一个文档')
        # 类别标签文件
        parser.add_argument('--label_names_file', default='label_names.txt', help='包含标签名称的文件(在数据集目录下)')

        # 构建分类词汇表用数据存盘文件
        parser.add_argument('--label_name_load_file', default='label_name_data.pt', help='保存筛选分类词汇表使用的张量数据文件')
        # 分类词汇表存盘文件
        parser.add_argument('--category_vocab_load_file', default='category_vocab.pt', help='保存筛选后每个分类的词汇表')
        # bert模型相关
        parser.add_argument('--bert_model', default='junnyu/wobert_chinese_plus_base', help='bert模型名称')
        # 模型训练参数
        parser.add_argument('--eval_batch_size', type=int, default=64, help='用于评估的每个GPU的批量大小；批量越大，训练越快')

        parser.add_argument('--top_pred_num', type=int, default=50, help='语言模型MLM顶部预测截止值')
        parser.add_argument('--category_vocab_size', type=int, default=100, help='每个类别的类别词汇大小')

        parser.add_argument('--max_len', type=int, default=512, help='文档被填充/截断的长度')
        # 中文停用词文件
        parser.add_argument('--stopword_file' , default = 'stopword_all.txt' , help='停用词文件名')
        return parser

    @staticmethod
    def extension(args):
        # 扩展为全局配置对象
        options = Dict(args.__dict__)
        # 预加载模型
        options.pretrained_lm = 'junnyu/wobert_chinese_plus_base'
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer
        options.tokenizer = WoBertTokenizer.from_pretrained(options.bert_model, model_max_length=options.max_len)
        # 词典
        options.vocab = options.tokenizer.get_vocab()
        # 词典长度
        options.vocab_size = len(options.vocab)
        # [MASK]的token_id
        options.mask_id = options.vocab[options.tokenizer.mask_token]
        # 反向词典
        options.inv_vocab = {k:v for v, k in options.vocab.items()}
        # 读取类别标签和词典表中所有的词汇，存入不同的字典
        options.label_name_dict, options.label2class = read_label_names(options)
        # 类别数量
        options.num_words = len(options.label_name_dict)
        # 中文停用词表
        options.stopword = [line.strip() for line in open(os.path.join(options.dataset_dir, options.stopword_file) , 'r' , encoding='utf-8')]

        return options

    def get_parser(self):
        # 初始化参数解析器
        parser = self.parse()
        # 初始化参数
        parser = self.initialize(parser)
        # 解析命令行参数
        # args = parser.parse_args([])
        args = parser.parse_args()
        # 扩展参数
        options = self.extension(args)
        return options


"""3.2 main_get_near_vocab"""
def main_get_near_vocab():
    opt = ArgsParse().get_parser()

    # 创建模型对象
    model = DaEnhanModel.from_pretrained(
        opt.pretrained_lm,
        output_attentions=False,
        output_hidden_states=False,
        num_labels=opt.num_words
    )
    model.to(opt.device)

    # 构建分类别词汇表
    label_name_data = create_label_name_dataset(opt, opt.train_file, opt.label_name_load_file)
    category_vocab , vocab_dict = create_category_vocabulary(
        opt = opt,
        model = model,
        label_name_data = label_name_data,
        loader_name = opt.category_vocab_load_file,
        top_pred_num=opt.top_pred_num,
        category_vocab_size=opt.category_vocab_size
    )
    return category_vocab , vocab_dict


"""停用词加载"""
def stopwords_list(stopword_file_path):
    return [line.strip() for line in open(os.path.join(os.getcwd(), stopword_file_path) , 'r' , encoding='utf-8')]


"""数据增强"""
import jieba
import random

def data_enhance(data_path , enhance_rate_dict , save_path , stop_num = 200 , power_data_path = None ,
                 stop_word_path = 'THUCNews/lotclass_data/stopword_all.txt'):
    # 读取语料
    data_path = os.path.join(os.getcwd(), data_path)
    if save_path is not None: save_path = os.path.join(os.getcwd(), save_path)
    all_data_org_dict = defaultdict(list)
    all_data_dict = defaultdict(list)
    all_words_dict = defaultdict(int)
    power_data = []
    for line in open(data_path, 'r', encoding='utf-8'):
        line = line.strip('\n')
        power_data.append(line)
        text, cls = line.split('\t')
        all_data_dict[int(float(cls))].append(text)
        all_data_org_dict[cls].append(line)
        # 每句话内添加词频
        for word in jieba.lcut(text):
            all_words_dict[word] += 1

    # 样本数量分布字典
    data_describe = {k: len(all_data_dict[k]) for k in all_data_dict}
    # 字典排序
    data_describe_sorted = {k : data_describe[k] for k in sorted(data_describe)}
    org_data_count = 0
    for k,v in data_describe_sorted.items():
        org_data_count+=v
    print(f'导入语料数据情况{data_describe_sorted},总体语料个数{org_data_count}')

    # 字频字典排序
    all_words_dict = list(sorted(all_words_dict.items(), key=lambda t: t[1] , reverse=True))
    # 统计data中最多出现的前100词
    all_words_dict_use = all_words_dict[:stop_num]
    print(all_words_dict_use)
    words = [word[0] for word in all_words_dict_use]
    # 将停用词过滤掉
    effect_words = []
    stop_words = stopwords_list(stop_word_path)
    for word in words:
        if word not in stop_words and word != ' ': effect_words.append(word)
    print(len(effect_words) , effect_words)
    # 将这些词保存
    # save_path = label_names.txt   THUCNews/lotclass_data/label_names.txt
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            for line in effect_words:
                f.write(line + '\n')
        print(f'可替换近义词文件{save_path}保存成功')
    # 拿出数据增强字典
    _ , vocab_dict = main_get_near_vocab()
    print(vocab_dict)

    """数据增强重采样"""
    # 重新读取
    power_data_dict = all_data_org_dict.copy()
    # all_data_org_dict{'0':['这是一条小样数据  0',...


    # 计算每个样本可替换语句的个数
    enable_power_num = []
    for cls,texts in all_data_org_dict.items():
        for text in texts:
            for replace_word in vocab_dict:
                if replace_word in text:
                    enable_power_num.append(cls)
                    break
    power_num_dict = {cls:enable_power_num.count(str(cls)) for cls in data_describe_sorted}
    print(f'可增强语料个数：{power_num_dict}')

    for cls,power_num in enhance_rate_dict.items():

        cls_add_num = len(all_data_org_dict[cls]) * enhance_rate_dict[cls]
            # text为每一条语料    => "这是一条小样数据    cls"
        for text in tqdm(all_data_org_dict[cls]):
            # 查找该条语料是否有可替换词语
            for replace_word in vocab_dict:
                if replace_word in text:
                    new_text = text.replace(replace_word , random.choice(vocab_dict[replace_word]))
                    power_data.append(new_text)
                    power_data_dict[cls].append(new_text)
                    cls_add_num -= 1
                    # 每遍单条语料只换一次
                    continue
                    # if len(power_data) > org_data_count * 3:break
            if cls_add_num <= 0:break
        # if len(power_data) > org_data_count * 3: break


    # 重新打乱数据集
    np.random.shuffle(power_data)

    if power_data_path is None:
        # 样本数量分布字典
        data_describe = {k: len(power_data_dict[k]) for k in power_data_dict}
        # 字典排序
        data_describe_sorted = {k: data_describe[k] for k in sorted(data_describe)}
        print(data_describe_sorted)
    else:
        # 样本数量分布字典
        data_describe = {k: len(power_data_dict[k]) for k in power_data_dict}
        # 字典排序
        data_describe_sorted = {k: data_describe[k] for k in sorted(data_describe)}
        print(f'{data_describe_sorted},\t总语料个数{len(power_data)}')

        with open(power_data_path, 'w', encoding='utf-8') as f:
            for line in power_data:
                f.write(line + '\n')
        print(f'数据增强后的文件：{power_data_path}保存成功')
    pass


if __name__ == '__main__':
    # split_data(r'THUCNews\data\allcorpus.txt' , [0.8,0.1,0.1], save_path= r'THUCNews\data')
    split_data(
        r'../../../Project/Classification of Telecom Complaints/Chinese-Texxt-Classification-pytorch-all_2.0/THUCNews/data/allcorpus.txt', [0.8, 0.1, 0.1])
    # remove_label(r'THUCNews\data\train.txt' , r'THUCNews\lotclass_data\train_no_label.txt')
    data_enhance(
        '../../../Project/Classification of Telecom Complaints/Chinese-Texxt-Classification-pytorch-all_2.0/THUCNews/data/train.txt', {'0':8 , '1':1, '3': 7 , '4':5}, 'THUCNews/lotclass_data/label_names.txt',
        power_data_path='THUCNews/data/power_train.txt')
