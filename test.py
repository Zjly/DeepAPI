# 新建测试文件以测试符合新数据集的BLEU
# 在计算BLEU时 将模型生成的分散的API进行组合并测试 例如将String . substring组合为String.substring作为一个整体来评价

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from helper import indexes2sent
from models import RNNEncDec

api_dataset = pd.read_feather("data/api.feather")
api_index_dict = api_dataset.set_index('api')['index'].to_dict()
api_data_count_dict = api_dataset.set_index('index')['total_count'].to_dict()
vocab_size = len(api_dataset)


def test():
    """
    测试模型
    :return:
    """
    # timestamp = 202207160059
    # epoch = 100

    timestamp = -1
    epoch = 1

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

    file_path = path + f"/{decode_mode}_{top_k}"

    # 结果评估
    evaluate(model, test_loader, vocab_desc, vocab_api, decode_mode, top_k, file_path)

    # top_k_list = [5, 10]
    # for top_k in top_k_list:
    #     file_path = path + f"/{decode_mode}_{top_k}"
    #     evaluate(model, test_loader, vocab_desc, vocab_api, decode_mode, top_k, file_path)


def evaluate(model, test_loader, vocab_desc, vocab_api, decode_mode, top_k, file_path):
    """
    对模型进行评估
    :return:
    """
    result = open(f"{file_path}.log", "w")
    device = next(model.parameters()).device

    # 评估指标
    output_id_list = []
    bleu_list = []
    meteor_list = []
    rouge_list = []
    levenshtein_distance_list = []
    jaro_winkler_list = []
    tail_list = []
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

        output_id = []
        for pred_t in pred_tokens:
            output = []
            for token in pred_t:
                if token in api_index_dict:
                    output.append(api_index_dict[token])
                else:
                    output.append(-1)
            output_id.append(output)

        label_id = []
        for token in ref_tokens:
            if token in api_index_dict:
                label_id.append(api_index_dict[token])
            else:
                label_id.append(-1)

        # 加入次数统计
        for pred_token in output_id:
            for id in pred_token:
                if id != "":
                    api_count_dict[id] = api_count_dict.setdefault(id, 0) + 1

        # 计算bleu
        bleu = Metric.calculate_bleu(output_id, label_id)
        bleu_list.append(bleu)

        # 计算meteor
        meteor = Metric.calculate_meteor(output_id, label_id)
        meteor_list.append(meteor)

        # 计算rouge
        rouge = Metric.calculate_rouge(output_id, label_id)
        rouge_list.append(rouge)

        # 计算levenshtein_distance
        levenshtein_distance = Metric.calculate_levenshtein_distance(output_id)
        levenshtein_distance_list.append(levenshtein_distance)

        # 计算jaro_winkler
        jaro_winkler = Metric.calculate_jaro_winkler(output_id)
        jaro_winkler_list.append(jaro_winkler)

        # 计算tail
        tail = Metric.calculate_tail(output_id)
        tail_list.append(tail)

        # 输出结果
        local_t += 1

        result.write(f"Batch {local_t}" + "\n")

        result.write("Question: \t" + desc_str + "\n")

        result.write("Target: \t" + ref_str + "\n")

        for output_index, row in enumerate(pred_sents):
            result.write(f"Output {output_index + 1}: \t" + str(row) + "\n")

        result.write(
            f"BLEU: {round(bleu, 4)}, meteor: {round(meteor, 4)}, rouge: {round(rouge, 4)}, levenshtein distance: {round(levenshtein_distance, 4)}, jaro winkler: {round(jaro_winkler, 4)}, tail: {round(tail, 4)}\n\n")

        # 加入output_id的输出
        output_id_list.append(output_id)

    # 保存输出id
    output_id_df = pd.DataFrame(columns=['output_id'])
    output_id_df['output_id'] = output_id_list
    output_id_df.to_feather(file_path + "-output.feather")

    # 保存计数字典
    api_count_dict = sorted(api_count_dict.items(), key=lambda d: d[0])
    api_count_df = pd.DataFrame(columns=['api', 'count'])
    api_list = []
    count_list = []
    for row in api_count_dict:
        api_list.append(row[0])
        count_list.append(row[1])
    api_count_df['api'] = api_list
    api_count_df['count'] = count_list
    api_count_df.to_feather(file_path + "-api_count.feather")

    # 计算指标
    coverage = Metric.calculate_coverage(api_count_dict)
    tail_coverage = Metric.calculate_tail_coverage(api_count_dict)

    # 平均评估结果计算
    result.write(
        "------------------------------------------------------------------------------------------------------------------------\n")
    result.write(
        f"BLEU: {round(float(np.mean(bleu_list)), 4)}, meteor: {round(float(np.mean(meteor_list)), 4)}, rouge: {round(float(np.mean(rouge_list)), 4)}, levenshtein distance: {round(float(np.mean(levenshtein_distance_list)), 4)}, jaro winkler: {round(float(np.mean(jaro_winkler_list)), 4)}, coverage: {round(coverage, 4)}, tail_coverage: {round(tail_coverage, 4)}, tail:{round(float(np.mean(tail_list)), 4)}")


class Metric:
    api_dataset = api_dataset
    api_data_count = api_dataset.set_index('index')['total_count'].to_dict()
    vocab_size = api_dataset.iloc[0]['index']
    rouge = Rouge()

    @staticmethod
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

        return np.max(bleu_list)

    @staticmethod
    def calculate_meteor(output_ids, label_ids):
        """
        计算METEORs
        :param output_ids: 模型输出值
        :param label_ids: 标签
        :return:
        """
        if len(output_ids) == 0:
            return 0

        meteor_list = []
        for i in range(len(output_ids)):
            # 未生成任何api
            if len(output_ids[i]) == 0:
                meteor_list.append(0)
                continue

            # 计算bleu
            score = single_meteor_score(list(str(id) for id in output_ids[i]), list(str(id) for id in label_ids))

            meteor_list.append(score)

        return np.max(meteor_list)

    @staticmethod
    def calculate_rouge(output_ids, label_ids):
        """
        计算ROUGE
        :param output_ids: 模型输出值
        :param label_ids: 标签
        :return:
        """
        if len(output_ids) == 0:
            return 0

        # 转化为由空格分隔的字符串
        output_ids = list(" ".join(str(id) for id in output) for output in output_ids)
        label_ids = " ".join(str(id) for id in label_ids)

        rouge_list = []
        for i in range(len(output_ids)):
            # 未生成任何api
            if len(output_ids[i]) == 0:
                rouge_list.append(0)
                continue

            # 计算rouge
            score = Metric.rouge.get_scores(output_ids[i], label_ids, avg=True)['rouge-l']['f']

            rouge_list.append(score)

        return np.max(rouge_list)

    @staticmethod
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
                levenshtein_list.append(Metric.levenshtein(output_ids[i], output_ids[j]))

        return (2 / (seq_length * (seq_length - 1))) * np.sum(levenshtein_list)

    @staticmethod
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

    @staticmethod
    def calculate_jaro_winkler(output_ids):
        """
        计算jaro_winkler
        对于推荐结果多样性的度量，衡量Top-K结果中API序列两两之间的相似度，表示单个推荐结果中列表的多样性。ILS越大，代表推荐结果的多样性越好
        :param output_ids:
        :return:
        """
        # 输出序列个数
        seq_length = len(output_ids)
        if seq_length <= 1:
            return 0

        # 两两之间计算levenshtein距离
        jaro_winkler_list = []
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                jaro_winkler_list.append(Metric.jaro_winkler(output_ids[i], output_ids[j]))

        return (2 / (seq_length * (seq_length - 1))) * np.sum(jaro_winkler_list)

    @staticmethod
    def jaro(seq1, seq2):
        """
        jaro相似度度量
        :param seq1:
        :param seq2:
        :return:
        """
        # 保证sequence1的长度比sequence2的长度短
        if len(seq1) > len(seq2):
            s = seq1
            seq1 = seq2
            seq2 = s

        len1 = len(seq1)
        len2 = len(seq2)

        # 匹配窗口大小
        window = max(len1, len2) // 2 - 1

        # 匹配的字符数量
        m = 0

        # 字符转换的次数
        t = 0

        # sequence2匹配转换的个数
        seq2_matched = [False] * len2

        for i in range(len1):
            # 直接匹配
            if seq1[i] == seq2[i]:
                m += 1
                seq2_matched[i] = True
                continue

            # 换位匹配
            for j in range(window):
                j_index = i - j - 1
                if j_index >= 0 and not seq2_matched[j_index] and seq1[i] == seq2[j_index]:
                    seq2_matched[j_index] = True
                    m += 1
                    t += 1
                    break

            for j in range(window):
                j_index = i + j + 1
                if j_index < len2 and not seq2_matched[j_index] and seq1[i] == seq2[j_index]:
                    seq2_matched[j_index] = True
                    m += 1
                    t += 1
                    break

        if m == 0:
            return 0

        return 1 / 3 * (m / len1 + m / len2 + (m - t // 2) / m)

    @staticmethod
    def jaro_winkler(seq1, seq2):
        """
        计算jaro_winkler相似度
        :param seq1:
        :param seq2:
        :return:
        """
        jaro_score = Metric.jaro(seq1, seq2)
        l = 0
        for i in range(min(4, len(seq1), len(seq2))):
            if seq1[i] == seq2[i]:
                l += 1

        p = 0.1
        jaro_winkler_score = jaro_score + l * p * (1 - jaro_score)
        return 1 - jaro_winkler_score

    @staticmethod
    def calculate_coverage(count_dict):
        """
        计算覆盖率
        覆盖率：对于推荐结果多样性的度量，衡量推荐系统所推荐的API占所有API数的比例。覆盖率越高，代表推荐结果的多样性越好
        :param count_dict:
        :return:
        """
        return len(count_dict) / len(Metric.api_dataset)

    @staticmethod
    def calculate_tail_coverage(count_dict):
        """
        计算尾部覆盖率
        将最受欢迎的前20%项目表示为头部项目，其他项目是构成尾部项目集的尾部项目
        :param count_dict:
        :return:
        """
        tail_begin_index = Metric.api_dataset.iloc[0]['index'] + len(Metric.api_dataset) * 0.2
        tail_count = 0
        for index, apis in enumerate(count_dict):
            if apis[0] > tail_begin_index:
                tail_count += 1

        return tail_count / (len(Metric.api_dataset) * 0.8)

    @staticmethod
    def calculate_tail(output_ids):
        """
        计算尾部
        将最受欢迎的前20%项目表示为头部项目，其他项目是构成尾部项目集的尾部项目
        :param count_dict:
        :return:
        """
        tail_begin_index = Metric.api_dataset.iloc[0]['index'] + len(Metric.api_dataset) * 0.2
        tail_list = []
        for output_id in output_ids:
            output_len = len(output_ids)
            tail_count = 0
            for id in output_id:
                if id > tail_begin_index:
                    tail_count += 1

            tail_list.append(tail_count / output_len)

        return np.mean(tail_list)


if __name__ == '__main__':
    test()
