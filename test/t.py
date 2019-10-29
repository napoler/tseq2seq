#encoding=utf-8
from __future__ import unicode_literals
import sys
sys.path.append("../")
from tseq2seq import *
from AutoBUlidVocabulary import *
# Ts2s()
# seq_data = [['b柯基犬是个子吗', '柯基犬'], ['哈士奇喜欢吃肉', '哈士奇'], ['猫咪喜欢吃鱼', '猫咪'], ['毛驴喜欢吃草', '毛驴'], ['牛逼的人', '人'], ['老牛拉车', '老牛'],['他将摘要抽取的指导准则设计为以下两个维度','摘要']]
# ts2s=Ts2s(n_step = 30,n_hidden = 64,epoch=5,path='./')
# # seq_data= rand_seq()
# ts2s.bulid_data(seq_data)
# ts2s.train()
# t=seq_data[:1]




# # print('*'*50,n)
# print(t[0][1])
# print('生成标题 ->', ts2s.translate(t[0][0]))





vocab=GVocab(path='./')


def remove_symbols(sentence):
    """
    Remove numbers and symbols from ASCII
    删除英文标点
    """
    import string
    del_estr = string.punctuation + string.digits  # ASCII 标点符号，数字
    replace = " "*len(del_estr)
    tran_tab = str.maketrans(del_estr, replace)
    sentence = sentence.translate(tran_tab)
    return sentence
def text_list(text):
    """
    使用空格转化为数组
    """
    return text.split()

def cmn_to_data():
    """
    构建训练数据
    """
# ========读取原始数据========
    with open('cmn.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')
    data = data[:100]
    # print(data[-5:])
    # 分割英文数据和中文数据
    en_data = [line.split('\t')[0] for line in data]
    ch_data = ['' + line.split('\t')[1] + '' for line in data]
    new_en_data=[]
    for line in en_data:
        line = remove_symbols(line)
        new_en_data.append(text_list(line))
    new_ch_data=[]
    for line in ch_data:
        new_ch_data.append(vocab.text_list(line))

    # print('英文数据:\n', new_en_data[:10])
    # print('\n中文数据:\n', new_ch_data[:5])

    seq_data=[]
    for i,line in enumerate(new_en_data):
        it=[]
        it.append(line)
        it.append(new_ch_data[i])
        seq_data.append(it)
    # print(seq_data[:10])
    return seq_data




seq_data=cmn_to_data()
Ts2s()
# seq_data = [['b柯基犬是个子吗', '柯基犬'], ['哈士奇喜欢吃肉', '哈士奇'], ['猫咪喜欢吃鱼', '猫咪'], ['毛驴喜欢吃草', '毛驴'], ['牛逼的人', '人'], ['老牛拉车', '老牛'],['他将摘要抽取的指导准则设计为以下两个维度','摘要']]
ts2s=Ts2s(n_step = 30,n_hidden = 64,epoch=500,path='./')
# seq_data= rand_seq()
ts2s.bulid_data(seq_data)
ts2s.train()
t=seq_data[:1]



# print('*'*50,n)
print(t[0][1])
print('生成标题 ->', ts2s.translate(t[0][0]))