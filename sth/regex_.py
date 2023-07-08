import os
import re as ree
import json
# import regex as ree


def reGex(text):

    # 提取经验
    def experience(text):
        all_list = []

        # print(text)
        pattern = "[有|具|0-9|丰富|对][\u4e00-\u9fa5\w]+经验"
        # print(pattern)
        # re.search 在文本中搜索匹配的内容
        matches = ree.finditer(pattern,text)
        for match in matches:
            # print(f'match:',match)
            index_e = [match.start(), match.end()]
            token_e = text[match.start():match.end()]
            all_list.append({"ent_name":token_e , "start_idx":index_e[0] , "end_idx":index_e[1] , "type":"经验"})
        return all_list

    # 提取学历
    def record(text):
        all_list = []
        # 提取文本字符串

        # print(text)
        # 定义一个正则表达式
        pattern = r'(高中|本科|专科|研究生|博士)'
        # 进行匹配
        matches = ree.finditer(pattern,text)
        # print(f'matches:',matches)
        # 输出匹配结果
        for match in matches:
            # print(f'match:',match)
            index_r = [match.start(), match.end()]
            token_r = text[match.start():match.end()]
            all_list.append({"ent_name": token_r, "start_idx": index_r[0], "end_idx": index_r[1], "type": "学历"})
        return all_list


    # 提取职称
    def professional(text):
        all_list = []
        pattern = "[有|具][\u4e00-\u9fa5\w]+职称"
        # print(pattern)
        # re.search 在文本中搜索匹配的内容
        matches = ree.finditer(pattern,text)

        for match in matches:
            # print(f'match:',match)
            index_p = [match.start(), match.end()]
            token_p = text[match.start():match.end()]
            all_list.append({"ent_name": token_p, "start_idx": index_p[0], "end_idx": index_p[1], "type": "职称"})
        return all_list


    # 提取证书
    def certificate(text):
        all_list = []
        pattern1 = "[有|具][\u4e00-\u9fa5\w]+证书"
        pattern2 = r'(英语CET-4|英语CET-6|英语(4|6|8)级|英语(四|六|八)级|证券从业资格证|法律职业资格证)'
        # print(pattern)
        # re.search 在文本中搜索匹配的内容
        matches1 = ree.finditer(pattern1,text)
        matches2 = ree.finditer(pattern2,text)

        for match1 in matches1:
            # print(f'match:',match)
            index_c1 = [match1.start(), match1.end()]
            token_c1 = text[match1.start():match1.end()]
            all_list.append({"ent_name": token_c1, "start_idx": index_c1[0], "end_idx": index_c1[1], "type": "证书"})

        for match2 in matches2:
            # print(f'match:',match)
            index_c2 = [match2.start(), match2.end()]
            token_c2 = text[match2.start():match2.end()]
            all_list.append({"ent_name": token_c2, "start_idx": index_c2[0], "end_idx": index_c2[1], "type": "证书"})

        return all_list

    experience_list = experience(text)
    record_list = record(text)
    professional_list = professional(text)
    certificate_list = certificate(text)
    return experience_list + record_list + professional_list + certificate_list


if __name__ == '__main__':
    from config import ArgsParse

    # 初始化参数解析器
    opt = ArgsParse().get_parser()

    load = opt.pred_corpus
    re_list = reGex(load)

    print(re_list)







