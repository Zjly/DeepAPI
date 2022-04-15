# 新建测试文件以测试符合新数据集的BLEU
# 在计算BLEU时 将模型生成的分散的API进行组合并测试 例如将String . substring组合为String.substring作为一个整体来评价
import math
import os
import random

import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import configs
from data_loader import APIDataset, load_dict
from helper import indexes2sent
from models import RNNEncDec


def test():
    """
    测试模型
    :return:
    """
    timestamp = 202204150037
    epoch = 100000
    decode_mode = 'beamsearch'  # greedy, beamsearch
    top_k = 10

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
    test_set = APIDataset("data/test.feather")
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    vocab_api = load_dict("data/vocab.apiseq.json")
    vocab_desc = load_dict("data/vocab.desc.json")

    # 加载模型
    model = RNNEncDec(conf)
    checkpoint = f"output/RNNEncDec/basic/{timestamp}/models/model_epo{epoch}.pkl"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    path = f"output/RNNEncDec/basic/{timestamp}/output/epo{epoch}"

    os.makedirs(path, exist_ok=True)

    file_path = path + f"/{decode_mode}_{top_k}.log"

    # 结果评估
    evaluate(model, test_loader, vocab_desc, vocab_api, decode_mode, top_k, file_path)


def evaluate(model, test_loader, vocab_desc, vocab_api, decode_mode, top_k, file_path):
    """
    对模型进行评估
    :return:
    """
    result = open(file_path, "w")
    device = next(model.parameters()).device

    # 评估指标
    bleu_list = []
    local_t = 0
    for desc, desc_lens, apiseq, apiseq_lens in tqdm(test_loader):
        desc_str = indexes2sent(desc[0].numpy(), vocab_desc)[0].replace("<s> ", "").replace(" </s>", "")
        desc, desc_lens = [tensor.to(device) for tensor in [desc, desc_lens]]

        with torch.no_grad():
            sample_words, sample_lens = model.sample(desc, desc_lens, top_k, decode_mode)

        pred_sents, _ = indexes2sent(sample_words, vocab_api)
        for i in range(len(pred_sents)):
            pred_sents[i] = pred_sents[i].replace(" </s>", "").replace(" . ", ".")
        pred_tokens = [sent.split(' ')[:] for sent in pred_sents]

        # 目标结果
        ref_str, _ = indexes2sent(apiseq[0].numpy(), vocab_api)
        ref_str = ref_str.replace("<s> ", "").replace(" </s>", "").replace(" . ", ".")
        ref_tokens = ref_str.split(' ')

        # 进行BLEU的评估
        bleu = calculate_bleu(pred_tokens, ref_tokens)
        bleu_list.append(bleu)

        # 输出结果
        local_t += 1

        result.write(f"Batch {local_t}" + "\n")
        result.write("Question: \t" + desc_str + "\n")
        result.write("Target: \t" + ref_str + "\n")
        for output_index, row in enumerate(pred_sents):
            result.write(f"Output {output_index + 1}: \t" + str(row) + "\n")
        result.write("BLEU: " + str(bleu) + "\n")
        result.write("\n")

    # 平均评估结果计算
    result.write(f"Total BLEU: {np.mean(bleu_list)}")

    return np.mean(bleu_list)


def calculate_bleu(output_ids, label_ids):
    """
    计算BLEU
    :param output_ids: 模型输出值
    :param label_ids: 标签
    :return:
    """
    if len(output_ids) == 0:
        return 0

    bleu_list = []
    for i in range(len(output_ids)):
        # 未生成任何api
        if len(output_ids[i]) == 0:
            bleu_list.append(0)
            continue

        # 计算bleu
        bleu_gram = sentence_bleu([output_ids[i]], label_ids,
                                  weights=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])

        # 如果标签的长度小于gram数 则仅统计最大值为标签长度的gram bleu
        bleu_len = min(len(bleu_gram), len(label_ids))
        bleu = 0
        for j in range(bleu_len):
            bleu += bleu_gram[j]
        bleu /= bleu_len

        # 计算bp
        bp = min(1, pow(math.e, (1 - len(label_ids) / len(output_ids[i]))))

        # 向bleu中加入bp惩罚
        bleu_list.append(bleu * bp)

    return np.max(bleu_list) * 100


if __name__ == '__main__':
    test()
