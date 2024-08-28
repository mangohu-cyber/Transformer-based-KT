"""
file: skill_builder_data_corrected.csv
input: user_id, skill_id, correct
output: train.pt, eval.pt

"""

import pandas as pd
import torch
import random
from torch.nn.utils.rnn import pad_sequence


class DatasetProcessor(object):
    """Processor for handling data from CSV files."""

    def __init__(self):
        self.num_skill = None
        self.max_len = None

    def _read_csv(self, dataset_path):
        """Reads a CSV file."""
        # 初始化user序列
        user_sequences = []
        df = pd.read_csv(dataset_path)

        # 最大序列长度
        max_skill_num = 0
        # 每个用户的作答序列
        grouped = df.groupby('user_id')

        # 将每一个user的作答序列以tuple存储
        for _, group in grouped:
            skill_ids = group['skill_id'].tolist()
            skill_count = len(skill_ids)
            if skill_count <= 2:
                continue
            correctness = group['correct'].tolist()
            # 计算最大序列长度
            max_skill_num = max(max_skill_num, skill_count)
            user_sequences.append(([skill_count], skill_ids, correctness))

        # 打乱user_sequences
        random.shuffle(user_sequences)
        # 统计知识点总数,最大序列长度
        self.num_skill = df['skill_id'].nunique()
        self.max_len = max_skill_num
        print("Finish reading data...")
        print(f"Number of users: {len(user_sequences)}")
        print(f"Number of skills: {self.num_skill}")
        print(f"Max sequence Length: {self.max_len}")

        return user_sequences

    def get_examples(self, data_dir):
        """Gets the examples for the dataset."""
        return self._create_examples(self._read_csv(data_dir))

    def _create_examples(self, tuple_rows):
        """Creates examples for the training and dev sets."""
        inputs = []  # 输入
        target_id = []  # 目标知识点id
        target_correct = []  # 目标作答结果
        ids = []  # 样本id
        correct = []  # 真实作答结果
        seq_len = []  # 序列长度

        for i in range(len(tuple_rows)):
            inputs.append(torch.tensor([int(tuple_rows[i][1][j]) + int(tuple_rows[i][2][j]) * self.num_skill for j in
                                        range(len(tuple_rows[i][1]) - 1)]))
            target_id.append(torch.tensor(list(map(lambda k: int(k), tuple_rows[i][1][1:]))))
            target_correct.append(torch.tensor(list(map(lambda k: int(k), tuple_rows[i][2][1:]))))
            ids.append(torch.tensor(list(map(lambda k: int(k), tuple_rows[i][1][:-1]))))
            correct.append(torch.tensor(list(map(lambda k: int(k), tuple_rows[i][2][:-1]))))
            seq_len.append(torch.tensor([int(tuple_rows[i][0][0]) - 1]))  # sequence

        # 使用 pad_sequence 来将张量填充到相同的长度
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        target_id_padded = pad_sequence(target_id, batch_first=True, padding_value=0)
        target_correct_padded = pad_sequence(target_correct, batch_first=True, padding_value=0)
        ids_padded = pad_sequence(ids, batch_first=True, padding_value=0)
        correct_padded = pad_sequence(correct, batch_first=True, padding_value=0)
        seq_len_padded = pad_sequence(seq_len, batch_first=True, padding_value=0)

        return (inputs_padded, target_id_padded, target_correct_padded,
                ids_padded, correct_padded, seq_len_padded)


def save_as_pt(filename, inputs, target_id, target_correct, ids, correct, seq_len):
    """Saves the data as a .pt file."""
    print('Data shape:', inputs.shape, target_id.shape, target_correct.shape, ids.shape, correct.shape, seq_len.shape)
    print('%s records' % inputs.shape[0])

    data = {
        'inputs': inputs,
        'target_id': target_id,
        'target_correct': target_correct,
        'ids': ids,
        'correct': correct,
        'seq_len': seq_len
    }

    torch.save(data, filename)
    print(f"Data saved to {filename}")


def data_split_remove(filename):
    # 读取数据集
    df = pd.read_csv(filename, low_memory=False)

    # 对源数据集的：order_id、skill_id缺失行处理，单独user_id序列处理,和skill_id的独热编码
    df.dropna(subset=['order_id', 'user_id', 'skill_id', 'correct'], inplace=True)

    # 保留order_id、user_id、skill_id、correct属性列
    df = df[['order_id', 'user_id', 'skill_id', 'correct']]

    # 对order_id、user_id排序
    df = df.sort_values(['order_id', 'user_id'])
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()
    df['skill_id'], _ = pd.factorize(df['skill_id'])

    # 计算总行数
    total_rows = len(df)

    # 计算每个部分的行数
    train_rows = int(total_rows * 0.8)
    eval_rows = int(total_rows * 0.1)  # 验证集占10%

    # 分割数据集 train:eval:test=8:1:1
    train_df = df.iloc[:train_rows]
    remaining_df = df.iloc[train_rows:]
    eval_df = remaining_df.iloc[:eval_rows]
    test_df = remaining_df.iloc[eval_rows:]

    # 保存数据集
    train_df.to_csv('../1-data/data_processed/assist12/train.csv', index=False)
    eval_df.to_csv('../1-data/data_processed/assist12/eval.csv', index=False)
    test_df.to_csv('../1-data/data_processed/assist12/test.csv', index=False)

    print(f"Train set size: {len(train_df)}")
    print(f"Eval set size: {len(eval_df)}")
    print(f"Test set size: {len(test_df)}")


if __name__ == '__main__':
    data_split_remove(filename='../1-data/data_raw/assist09/2012-2013-data-with-predictions-4-final.csv')

    train_dir = '../1-data/data_processed/assist12/train.csv'
    test_dir = '../1-data/data_processed/assist12/test.csv'

    processor = DatasetProcessor()
    inputs, target_id, target_correct, ids, correct, seq_len = processor.get_examples(train_dir)
    save_as_pt('../1-data/data_processed/assist12/train.pt', inputs, target_id, target_correct, ids, correct, seq_len)
    inputs, target_id, target_correct, ids, correct, seq_len = processor.get_examples(train_dir)
    save_as_pt('../1-data/data_processed/assist12/eval.pt', inputs, target_id, target_correct, ids, correct, seq_len)
