# 新建测试文件以测试符合新数据集的BLEU
# 在计算BLEU时 将模型生成的分散的API进行组合并测试 例如将String . substring组合为String.substring作为一个整体来评价

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import configs
from data_loader import APIDataset, load_dict
from helper import indexes2sent, sent2indexes
from models import RNNEncDec

api_dataset = pd.read_feather("data/api.feather")
api_index_dict = api_dataset.set_index('api')['index'].to_dict()
api_data_count_dict = api_dataset.set_index('index')['total_count'].to_dict()
vocab_size = len(api_dataset)


def test():
    timestamp = -1
    epoch = 50

    decode_mode = 'beamsearch'  # greedy, beamsearch
    top_k = 50

    conf = getattr(configs, 'config_RNNEncDec')()
    # 设置随机数种子
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1111)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    # 加载数据
    vocab_api = load_dict("data/vocab.apiseq.json")
    vocab_desc = load_dict("data/vocab.desc.json")

    # 加载模型
    model = RNNEncDec(conf)
    checkpoint = f"output/RNNEncDec/basic/{timestamp}/models/model_epo{epoch}.pkl"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    query = "get the file extension value of the content type"
    # query = "parse the uses licence node of this package , if any , and returns the license definition if theres one"
    desc, desc_lens = sent2indexes(query, vocab_desc, 50)
    desc = torch.tensor(desc, dtype=torch.long).unsqueeze(0)
    desc_lens = torch.tensor(desc_lens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        sample_words, sample_lens = model.sample(desc, desc_lens, top_k, decode_mode)

    for i in range(top_k):
        print(indexes2sent(sample_words[i], vocab_api))
    pass

if __name__ == '__main__':
    test()
