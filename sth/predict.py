import os
import json
import torch
import numpy as np
from tqdm import tqdm
from config import ArgsParse
from model_utils import load_ner_model, custom_local_bert_tokenizer
from regex_ import reGex

def inference(opt, text, model, tokenizer):
    # 获取文本的offset_mapping
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    new_span, entities = [], []
    # 提取token索引
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    # 编码后文本token
    model_input = tokenizer(text, return_tensors='pt')
    input_ids = model_input["input_ids"].to(opt.device)
    token_type_ids = model_input["token_type_ids"].to(opt.device)
    attention_mask = model_input["attention_mask"].to(opt.device)
    # 模型推理
    with torch.no_grad():
        scores = model(input_ids, attention_mask, token_type_ids)[0]
        scores = scores.squeeze(0).cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"ent_name": text[new_span[start][0] : new_span[end][-1] + 1],"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": opt.categories_rev[l]})

    return {"text": text, "entities": entities}

from process import convet_corpus , corpus_split, get_dataloader, NerDataset

def predict_main(text):


    opt = ArgsParse().get_parser()

    # 加载模型和tokenizer
    model, max_length = load_ner_model(opt)
    model.to(opt.device)
    tokenizer = custom_local_bert_tokenizer(opt, max_length)

    # 从语料中随机抽取样本进行推理
    all_ = []
    # 加载语料
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.corpus_dir , opt.corpus_load_file))
    corpus_data = convet_corpus(opt.categories , opt.corpus_load_file )


    for d in tqdm(text):
        pred_dict = inference(opt, d, model, tokenizer)
        re_dict = reGex(d)
        pred_dict['entities'] += re_dict

        pred_dict_entities = pred_dict['entities']
        pred_dict['entities'] = sorted(pred_dict_entities, key=lambda x: x['start_idx'])

        all_.append(pred_dict)
        # all_.append(inference(opt, d["text"], model, tokenizer))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), './outputs/ner_test_4.json'))
    json.dump(
        all_,
        open(output_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )
    # print(all_)
    return all_




if __name__ == '__main__':

    #
    # opt = ArgsParse().get_parser()
    #
    # # 加载模型和tokenizer
    # model, max_length = load_ner_model(opt)
    # model.to(opt.device)
    # tokenizer = custom_local_bert_tokenizer(opt, max_length)
    #
    # # 从语料中随机抽取样本进行推理
    # all_ = []
    # # 加载语料
    # corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.corpus_dir , opt.corpus_load_file))
    # corpus_data = convet_corpus(opt.categories , opt.corpus_load_file )
    #
    # # 拆分语料
    # _, test_corpus = corpus_split(corpus_data, split_rate=0.998)
    # for d in tqdm(test_corpus):
    #     pred_dict = inference(opt, d["text"], model, tokenizer)
    #     re_dict = reGex(d["text"])
    #     pred_dict['entities'] += re_dict
    #     all_.append(pred_dict)
    #     # all_.append(inference(opt, d["text"], model, tokenizer))
    # output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), './outputs/ner_test_2.json'))
    # json.dump(
    #     all_,
    #     open(output_file, 'w', encoding='utf-8'),
    #     indent=4,
    #     ensure_ascii=False
    # )
    # # print(all_)
    a = '任职资格\t1、房地产经营与管理相关专业；2、3年以上房产企业开发报建经理级及以上岗位工作经验；3、具有长期、稳定、良好的社会（政府）自愿和较强人际沟通能力；4、熟悉房地产项目审批的程序及环节；5、具备适应岗位的文字、语言表达能力及工程图纸审核技能；6、具有很强的对外公关能力，以及团队管理能力。\n'
    out = predict_main([a])
    print(out)
