# 新建测试文件以测试符合新数据集的BLEU
# 在计算BLEU时 将模型生成的分散的API进行组合并测试 例如将String . substring组合为String.substring作为一个整体来评价
import math
import os
import random

import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import configs
from data_loader import APIDataset, load_dict
from helper import indexes2sent
from models import RNNEncDec

api_dataset = pd.read_feather("data/api.feather")
api_data_count_dict = api_dataset.set_index('api')['total_count'].to_dict()
vocab_size = len(api_dataset)


def test():
    """
    测试模型
    :return:
    """
    timestamp = 202204150037
    epoch = 935000
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
    levenshtein_distance_list = []
    popularity_list = []
    local_t = 0

    # 保存出现次数信息
    api_count_dict = {}

    for desc, desc_lens, apiseq, apiseq_lens in tqdm(test_loader):
        desc_str = indexes2sent(desc[0].numpy(), vocab_desc)[0].replace("<s> ", "").replace(" </s>", "")
        desc, desc_lens = [tensor.to(device) for tensor in [desc, desc_lens]]

        with torch.no_grad():
            sample_words, sample_lens = model.sample(desc, desc_lens, top_k, decode_mode)

        pred_sents, _ = indexes2sent(sample_words, vocab_api)
        for i in range(len(pred_sents)):
            pred_sents[i] = pred_sents[i].replace(" </s>", "").replace(" . ", ".").replace(".<init>", "()")
        pred_tokens = [sent.split(' ')[:] for sent in pred_sents]

        # 目标结果
        ref_str, _ = indexes2sent(apiseq[0].numpy(), vocab_api)
        ref_str = ref_str.replace("<s> ", "").replace(" </s>", "").replace(" . ", ".").replace(".<init>", "()")
        ref_tokens = ref_str.split(' ')

        # 加入次数统计
        for pred_token in pred_tokens:
            for id in pred_token:
                api_count_dict[id] = api_count_dict.setdefault(id, 0) + 1

        # 进行BLEU的评估
        bleu = calculate_bleu(pred_tokens, ref_tokens)
        bleu_list.append(bleu)

        # 计算levenshtein_distance
        levenshtein_distance = calculate_levenshtein_distance(pred_tokens)
        levenshtein_distance_list.append(levenshtein_distance)

        # 计算popularity
        popularity = calculate_popularity(pred_tokens)
        popularity_list.append(popularity)

        # 输出结果
        local_t += 1

        result.write(f"Batch {local_t}" + "\n")

        result.write("Question: \t" + desc_str + "\n")

        result.write("Target: \t" + ref_str + "\n")

        for output_index, row in enumerate(pred_sents):
            result.write(f"Output {output_index + 1}: \t" + str(row) + "\n")

        result.write(
            f"BLEU: {round(bleu, 2)}, levenshtein distance: {round(levenshtein_distance, 4)}, popularity: {round(popularity, 4)}\n\n")

    # 计算指标
    coverage = calculate_coverage(api_count_dict)
    information_entropy = calculate_information_entropy(api_count_dict)

    # 平均评估结果计算
    result.write(
        "------------------------------------------------------------------------------------------------------------------------\n")
    result.write(
        f"BLEU: {round(np.mean(bleu_list), 2)}, levenshtein distance: {round(float(np.mean(levenshtein_distance_list)), 4)}, coverage: {round(coverage, 4)}, information_entropy: {round(information_entropy, 4)}, popularity: {round(float(np.mean(popularity_list)), 4)}")


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


def calculate_levenshtein_distance(output_ids):
    """
    计算levenshtein距离
    对于推荐结果多样性的度量，衡量Top-K结果中API序列两两之间的相似度，表示单个推荐结果中列表的多样性。ILS越大，代表推荐结果的多样性越好
    :param output_ids:
    :return:
    """
    # 输出序列个数
    seq_length = len(output_ids)
    if seq_length <= 1:
        return 0

    # 两两之间计算levenshtein距离
    levenshtein_list = []
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            levenshtein_list.append(levenshtein(output_ids[i], output_ids[j]))

    return (2 / (seq_length * (seq_length - 1))) * np.sum(levenshtein_list)


def levenshtein(seq1, seq2):
    """
    levenshtein相似度度量
    :param seq1:
    :param seq2:
    :return:
    """
    len1 = len(seq1)
    len2 = len(seq2)

    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        dp[i, 0] = i

    for i in range(len2 + 1):
        dp[0, i] = i

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            t = 1
            if seq1[i - 1] == seq2[j - 1]:
                t = 0

            dp[i][j] = min(dp[i - 1, j - 1] + t, dp[i, j - 1] + 1, dp[i - 1, j] + 1)

    if max(len1, len2) == 0:
        return 0

    return dp[len1][len2] / max(len1, len2)


def calculate_coverage(count_dict):
    """
    计算覆盖率
    覆盖率：对于推荐结果多样性的度量，衡量推荐系统所推荐的API占所有API数的比例。覆盖率越高，代表推荐结果的多样性越好
    :param count_dict:
    :return:
    """
    return len(count_dict) / len(api_dataset)


def calculate_information_entropy(count_dict):
    """
    计算信息熵
    信息熵：对于推荐结果多样性的度量，衡量推荐系统中推荐结果信息的不确定性，即推荐结果的是否包含了热门以及冷门的API。信息熵越大，代表推荐结果的多样性越好
    :param count_dict:
    :return:
    """
    # 计算api总数
    sum_count = 0
    for key in count_dict:
        sum_count += count_dict[key]

    # 计算信息熵
    information_entropy = 0
    for key in count_dict:
        pi = count_dict[key] / sum_count
        information_entropy += pi * math.log(pi)

    return -information_entropy


def calculate_popularity(output_ids):
    """
    计算流行度
    计算推荐列表中api的热度。流行度越大，说明推荐算法倾向于推荐“热度”越大、越流行的api；反之，则越倾向于推荐比较冷门的api，越能反映出用户的兴趣。
    :param output_ids:
    :return:
    """
    popularity_list = []
    for output_id in output_ids:
        if len(output_id) == 0:
            popularity_list.append(0)
            continue

        # 计算流行度
        id_popularity = 0
        for api in output_id:
            if api in api_data_count_dict:
                id_popularity += math.log(api_data_count_dict[api])
        id_popularity = id_popularity / len(output_id)
        popularity_list.append(id_popularity)
    return np.mean(popularity_list)


if __name__ == '__main__':
    test()
