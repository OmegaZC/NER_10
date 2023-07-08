import os
import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast

from config import ArgsParse



def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # 在batch_size * ent_type_size这一维度上求平均损失
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    # 先取出前两个维度
    batch_size, ent_type_size = y_pred.shape[:2]
    # 降维
    # y_true => (batch_size * ent_type_size , seq_len * seq_len)
    # y_pred => (batch_size * ent_type_size , seq_len * seq_len)
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)

def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits

class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs,position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            # [batch_size , seq_len , 128]
            input_shape = inputs.shape
            # 提取批次和序列长度
            batch_size, seq_len = input_shape[0], input_shape[1]
            # 创建seq_len长度相同的float类型向量   => [1 , seq_len]    [None]<=>.reshape((1,-1))
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        # 编码向量【0. , 1. , ... , 31.]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        # 爱因斯坦求和公式：通过符号表达对于目标计算或转换操作
        # embeddings：[1,seq_len,32]
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        # 上面计算出三维张量结果，通过sin、cos转换后，stack堆叠到最后一个维度   [1,seq_len,32,2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # => [1 , seq_len , 64]    reshape=>sin、cos交叉合并
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        # 传统transformer做法'+'
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encoder: bert-base
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2) #原版的dense2是(inner_dim * 2, ent_type_size * 2)

    """
    sequence_masking：为一个batch里所有token中填充的位置赋值极大负数（mask）
    参数：
        x:logit[batch_size , ent_type_size , seq_len , seq_len]
        mask:attention_mask[batch_size , seq_len]
        value:填充的值
        axis:维度/秩
    return:
        x * mask + value * (1 - mask)
    """
    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            # mask升维至与x同等维度
            # 1. 先扩充出ent_type_size维度    [batch_size , seq_len] => [batch_size , 1(ent_type_size) , seq_len]
            for _ in range(axis - 1):
                # torch.unsqueeze(mask, n):在n的维度上添加一个维度
                mask = torch.unsqueeze(mask, 1)
            # 2. 再将余下的seq_len维度补齐   [batch_size,1(ent_type_size),seq]=>[batch_size,1(ent_type_size),seq,1(seq)]
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            """"
            *时维度不对齐，广播机制会按照从后往前的顺序自动对齐维度
            例：x * mask:                                                     行                   列
                [batch_size , ent_type_size , seq_len , seq_len] * [batch_size,1,seq,1] or [batch_size,1,1,seq]
             => [batch_size , ent_type_size , seq_len , seq_len] * [batch_size,1,seq,seq] or [batch_size,1,seq,seq]
             => [batch_size , ent_type_size , seq_len , seq_len] * [batch_size,ent_type_size,seq,seq]
            """
            # value * (1 - mask):原来mask那内有效的为1，无效为0.(1 - mask)有效为0，无效为1，无效的地方填充value
            return x * mask + value * (1 - mask)
    """
    add_mask_tril:将每个批次的每个实体预测矩阵保留上三角部分
    参数：
        logits:每个批次每个实体的预测矩阵
        mask:掩码矩阵
    return:
        只保留上三角的logits
    """
    def add_mask_tril(self, logits, mask):
        # 同步logits和mask格式
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        """1. 先将一个batch_size内将填充字符mask"""
        # logits.ndim:秩 => 维度
        # 先将列维度添加mask
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        # 再将行维度添加mask
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        """2. 保留上三角"""
        # 排除下三角
        # 1. 先创建全1矩阵形状和logits相同。2. 再裁切三角，要保留下三角数值为1
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        # 3. 将下三角全部减去一个极大的数
        logits = logits - mask * 1e12
        return logits

    """
    参数:
        input_ids:输入语料在tokenizer里对应的id列表tensor           [batch_size , max_seq_len]
        attention_mask:mask掩码信息(1,1,..,0,0)列表tensor         [batch_size , max_seq_len]
        token_type_ids:dataloader里保存     [batch_size , max_seq_len]
        labels:实体标签                     [batch_size , ent_type_size , max_seq_len , max_seq_len]
    return:
        out:
    """
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # 同步设备端
        self.device = input_ids.device
        # encoder（bert_model）参数不进行梯度更新
        with torch.no_grad():
            context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # 拿出context_outputs(BaseModelOutput)中的last_hidden_state(最后一层隐层状态)   [batch,max_seq_len,768]
        last_hidden_state = context_outputs.last_hidden_state
        # 将last_hidden_state传给dense_1层  outputs:[batch_size , seq_len , 128(self.inner_dim * 2)]
        outputs = self.dense_1(last_hidden_state)
        # 拆分奇偶列     qw/kw:  [batch_size , seq_len , 64(self.inner_dim)]
        # 从0,1开始间隔为2  [:,:,]  [...,2]
        qw, kw = outputs[...,::2], outputs[..., 1::2]
        if self.RoPE:
            # pos:[1 , seq_len , 64(self.inner_dim)]
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            # 提取其中sin、cos编码值从[1,seq_len,32](pos[..., 1::2]/pos[..., ::2])扩容到 [1,seq_len,64]
            """.repeat_interleave(2, dim=-1)：将-1(最后一维)维度的所有数值都重复2遍"""
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            # qw2:[batch_size , seq_len , 32 , 2]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            # qw2:[batch_size , seq_len , 64]   最后一维奇偶穿插
            qw2 = torch.reshape(qw2, qw.shape)
            # qw注入相对位置信息    变换矩阵Rm  qw:[batch,seq,64(self.inner_dim)]
            """qw一个batch_size[batch,seq,64]内都注入同一套位置编码信息[1,seq_len,64]"""
            qw = qw * cos_pos + qw2 * sin_pos
            # kw如法炮制
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            # kw注入相对位置信息    变换矩阵Rn  kw:[batch,seq,64(self.inner_dim)]
            kw = kw * cos_pos + kw2 * sin_pos
        # 爱因斯坦求和公式
        # logits:[batch , seq_len , seq_len]
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        # bias:[batch , self.ent_type_size * 2 , seq_len]
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        '''
        [batch , 1 , seq(m) , seq(n)] + 
        [batch , even_state(奇数) ent_type_size * 2 / 2 , 1 , seq] + 
        [batch , odd_state(偶数) ent_type_size * 2 / 2 , 1 , seq]
        广播机制=> [batch,class(even_state+odd_state),seq,state]+[batch,even_state,seq,state]+[batch,odd_state,seq,state]
                => [batch , event_type , seq , seq]
        '''
        #           [batch , 1 , seq , state] + [batch , even_state , 1 , state] + [batch , odd_state , 1 , state]
        # logits:[batch_size , ent_type_size , seq_len , seq_len]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # 添加下三角掩码 => 得出上三角矩阵    logits:[batch_size , self.ent_type_size , seq_len , seq_len]
        logits = self.add_mask_tril(logits, mask=attention_mask)
        out = (logits,)
        # 如果模型还接收了label参数，则为模型训练，则需要计算并返回logits和loss
        if labels is not None:
            loss = loss_fun(labels , logits)
            # out:{tuple}   loss , logits
            out = (loss,) + out
        return out

if __name__ == '__main__':
    from process import get_dataloader, load_json_corpus
    from ner_dataset import NerDataset

    opt = ArgsParse().get_parser()
    local = os.path.join(os.path.dirname(__file__),opt.local_model_dir, opt.bert_model)
    bert = BertModel.from_pretrained(local)
    tokenizer = BertTokenizerFast.from_pretrained(local)

    # 创建模型对象
    egp = EffiGlobalPointer(bert, opt.categories_size, opt.head_size)
    # print(egp)
    # 模型训练语料
    # 加载语料
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'corpus/nested_corpus'))
    corpus_data = load_json_corpus(corpus_dir)
    # 语料数据集
    dataset = NerDataset(corpus_data)

    dataloader = get_dataloader(opt,dataset, tokenizer)

    for train_data in dataloader:
        input_ids = train_data['input_ids']
        attention_mask = train_data['attention_mask']
        token_type_ids = train_data['token_type_ids']
        labels = train_data['labels']
        print(labels.numpy())
        logits = egp(input_ids, attention_mask, token_type_ids , labels)[0]
        print(logits.shape)
        break

