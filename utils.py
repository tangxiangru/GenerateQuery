# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cPickle
import codecs
import jieba
import gensim
import os
import random

"""
DATA UTILS,数据处理模块，提供各种数据处理的api
"""

PAD = '<PAD>'
UNK = '<UNK>'
START = '<START>'
END = '<END>'

################################################################################
#                                 DATA UTILS                                   #
################################################################################
def strB2Q(ustring):
    # 半角转全角
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif 32 <= inside_code <= 126:
            inside_code += 65248
        rstring += unichr(inside_code)
    return rstring

def strQ2B(ustring):
    # 全角转半角
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def create_dict(item_list):
    # Create a dictionary of items from several list of list of items.
    dic = {}
    assert type(item_list) in (list,tuple)
    for items in item_list:
        for item in items:
            dic[item]=dic.get(item,0)+1
    # Make sure thar <PAD> have a id 0
    dic[PAD]=1e10
    # Add <UNK>,<START>,<END> to the dictionary
    for i,item in enumerate([UNK,START,END]):
        dic[item]=dic[PAD]-i-1
    return dic

def create_mapping(dic1,dic2,vocab_size1,vocab_size2):
    """
    Create a mapping (item to ID / ID to item) from two dictionary.
    """
    assert type(dic1) is dict and type(dic2) is dict
    assert vocab_size1 == -1 or vocab_size1 > 0
    assert vocab_size2 == -1 or vocab_size2 > 0
    sorted_items1 = [k for k, v in sorted(dic1.items(), key=lambda x: (-x[1], x[0]))]
    sorted_items2 = [k for k, v in sorted(dic2.items(), key=lambda x: (-x[1], x[0]))]
    if vocab_size1 > 0:
        sorted_items1 = sorted_items1[:vocab_size1]
    if vocab_size2 > 0:
        sorted_items2 = sorted_items2[:vocab_size2]
    sorted_items = sorted_items2
    target_vocab_size = len(sorted_items)
    dic3 = dict([(item,1) for item in sorted_items2])
    for item in sorted_items1:
        if item not in dic3:
            sorted_items.append(item)
    id2item = {i: v for i, v in enumerate(sorted_items)}
    item2id = {v: k for k, v in id2item.items()}
    return item2id, id2item,target_vocab_size

def load_pre_trained_emb(pre_trained_emb_path,binary=False,encoding='utf8'):
    """
    Load pre_trained word embeddings.
    'binary' is a boolean indicating whether the data is in binary word2vec format.
    """
    pre_trained_words = {}
    try:
        # pre_trained = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_emb_path,binary=binary,encoding=encoding)
        pre_trained = gensim.models.KeyedVectors.load(pre_trained_emb_path)
        for word in pre_trained.vocab:
            pre_trained_words[word] = pre_trained[word]
    except:
        # load pre_trained word embedding from text format(not fit word2vec format)
        # print "load pre_trained word embedding from text format(not fit word2vec format)"
        with codecs.open(pre_trained_emb_path,'r',encoding=encoding) as fin:
            for line in fin:
                we = line.split()
                if len(we) > 2:
                    w,e = we[0],np.array(we[1:],dtype=np.float32)
                    pre_trained_words[w]=e
    return pre_trained_words

################################################################################
#                                 Load DATA                                   #
################################################################################

def read_data(filename,sep_level='word',is_B2Q=True):
    """
    Read data from original file.
    Reture query(word_level or char_level) list and keywords list.
    """
    assert sep_level in ['word','char']
    query_list,keywords_list = [],[]
    with codecs.open(filename,'r','utf8') as fin:
        for line in fin:
            if is_B2Q:
                line = strB2Q(line) #全部转化为全角字符
            else:
                line = strQ2B(line) #全部转化为半角字符
            if len(line.strip().split('\t'))==3:
                id,query,keywords = line.strip().split('\t')
            else:
                query, keywords = line.strip().split('\t')
            if sep_level == 'word':
                query = [word for word in jieba.cut(query)]
            else:
                query = [char for char in query]
            keywords = keywords.split()
            query_list.append(query)
            keywords_list.append(keywords)
    assert len(query_list) == len(keywords_list)
    return query_list,keywords_list

def load_mappings(query_list,keywords_list,mappings_path=None,source_vocab_size=-1,target_vocab_size=-1):
    # Load or create mappings
    if mappings_path is not None and os.path.exists(mappings_path):
        print 'loadding mapping.pkl'
        item2id, id2item,target_vocab_size = cPickle.load(open(mappings_path,'rb'))
    else:
        query_dic = create_dict(query_list)
        keywords_dic = create_dict(keywords_list)
        item2id, id2item,target_vocab_size  = create_mapping(query_dic,keywords_dic,source_vocab_size,target_vocab_size)
        # if mappings_path is not None:
        #     cPickle.dump([item2id,id2item,target_vocab_size ],open(mappings_path,'wb'),cPickle.HIGHEST_PROTOCOL)
    return item2id,id2item,target_vocab_size

def load_emb(item2id,pre_trained_emb_path,emb_dim,binary=False,encoding='utf8'):
    # Load initialized embeddings
    pre_trained_words = load_pre_trained_emb(pre_trained_emb_path,binary=binary,encoding=encoding)
    emb_size = len(item2id)
    emb = np.zeros(shape=[emb_size,emb_dim],dtype=np.float32)
    for item,id in item2id.items():
        if item in pre_trained_words:
            emb[id] = pre_trained_words[item]
        elif strQ2B(item) in pre_trained_words:  #查找半角字符
            emb[id] = pre_trained_words[strQ2B(item)]  #查找全角字符
        elif strB2Q(item) in pre_trained_words:
            emb[id] = pre_trained_words[strB2Q(item)]
        else:
            emb[id] = np.random.uniform(low=0.1,high=0.1,size=[emb_dim]).astype(np.float32)
    emb[item2id[PAD]] = np.zeros(shape=[emb_dim],dtype=np.float32)
    return emb

def prepare_dataset(query_list,keywords_list,item2id):
    # Prepare_dataset, and convert thest datas to discrete representation.
    x,y = [],[]
    for query in query_list:
        sentence=[]
        for item in query:
            sentence.append(item2id.get(item,item2id[UNK]))
        x.append(sentence)
    for keywords in keywords_list:
        sentence=[item2id[START]]
        for item in keywords:
            sentence.append(item2id.get(item,item2id[UNK]))
        sentence.append(item2id[END])
        y.append(sentence)
    return x,y

################################################################################
#                              Batch Manager                                   #
################################################################################

class BatchManager(object):
    """
    Mini-Batch Manager Class with padding.
    """
    def __init__(self,data,batch_size,shuffle=False):
        self.data=data
        self.batch_size=batch_size
        self.num_batch=len(data)//batch_size
        self.shuffle = shuffle

    def _pad_data(self,data):
        query_batch,keywords_batch=zip(*data)
        query_max_length = max([len(query) for query in query_batch])
        keywords_max_length = max([len(keywords) for keywords in keywords_batch])
        pad_query_batch = []
        pad_keywords_batch = []
        pad_query_length = []
        pad_keywords_length =[]
        for query in query_batch:
            query_paddings = [0]*(query_max_length-len(query))
            pad_query_batch.append(query+query_paddings)
            pad_query_length.append(len(query))
        for keywords in keywords_batch:
            keywords_paddings = [0]*(keywords_max_length-len(keywords))
            pad_keywords_batch.append(keywords+keywords_paddings)
            pad_keywords_length.append(len(keywords)-1)
        return [pad_query_batch,pad_keywords_batch,pad_query_length,pad_keywords_length]

    def iter_batch(self):
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(self.num_batch):
            batch_data=self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield self._pad_data(batch_data)
        if not self.shuffle:
            batch_data=self.data[self.num_batch*self.batch_size:]
            yield self._pad_data(batch_data)

################################################################################
#                              Evaluate                                        #
################################################################################

def evaluate_scores(querys,predicts,goldens,id2item,out_dir):
    """Compute Presicion,Recall,F1"""
    assert len(predicts)==len(goldens)
    predict_num=0 #预测为正例数目
    golden_num=0  #全部正例数目
    tp = 0        #预测为正确的正例数目
    fout=codecs.open(os.path.join(out_dir,'results.txt'),'w','utf-8')
    for query,predict,golden in zip(querys,predicts,goldens):
        fout.write('query:  '+''.join([id2item[x] for x in query])+'\n')
        fout.write('golden: '+' '.join([id2item[x] for x in golden])+'\n')
        fout.write('predict:'+' '.join([id2item[x] for x in predict])+'\n\n')
        predict = set(predict)
        predict_num += len(predict)
        golden_num += len(golden)
        for item in predict:
            if item in golden:
                tp+=1
    fout.close()
    presicion = tp*100.0/(predict_num+1e-10)
    recall = tp*100.0/(golden_num+1e-10)
    f1 = presicion*recall*2/(presicion+recall+1e-10)
    return presicion,recall,f1

def evaluate_scores2(querys,predicts,goldens,goldens_unk,id2item,out_dir):
    """Compute Presicion,Recall,F1"""
    assert len(predicts)==len(goldens)
    predict_num=0 #预测为正例数目
    golden_num=0  #全部正例数目
    tp = 0        #预测为正确的正例数目
    fout=codecs.open(os.path.join(out_dir,'results.txt'),'w','utf-8')
    for query,predict,golden,golden_unk in zip(querys,predicts,goldens,goldens_unk):
        fout.write('query:  '+''.join([id2item[x] for x in query])+'\n')
        fout.write('golden: '+' '.join(golden)+'\n')
        fout.write('golden(unk): '+' '.join([id2item[x] for x in golden_unk])+'\n')
        predict = [id2item[x] for x in predict if x!=1]
        fout.write('predict: '+' '.join(predict)+'\n\n')

        predict = set(predict)
        predict_num += len(predict)
        golden_num += len(golden)
        for item in predict:
            if item in golden:
                tp+=1
    fout.close()
    presicion = tp*100.0/(predict_num+1e-10)
    recall = tp*100.0/(golden_num+1e-10)
    f1 = presicion*recall*2/(presicion+recall+1e-10)
    return presicion,recall,f1


if __name__ == '__main__':
    query_list,keywords_list=read_data('data/train.txt')
    item2id,id2item,target_vocab_size = create_mapping(create_dict(query_list),create_dict(keywords_list),-1,-1)

    x,y=prepare_dataset(query_list,keywords_list,item2id)
    assert len(x)==len(y)
    for i in xrange(10):
        print x[i],' '.join([id2item[item] for item in x[i]])
        print y[i],' '.join([id2item[item] for item in y[i]])
        print "********************"




