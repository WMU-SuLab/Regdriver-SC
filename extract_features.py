import os
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import argparse

# 固定随机种子
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

# ======================
# 参数设置
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="DNABERT-2 特征提取脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练集文件路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试集文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--train_output_name", type=str, default="train.npy", help="训练集输出文件名")
    parser.add_argument("--test_output_name", type=str, default="test.npy", help="测试集输出文件名")
    return parser.parse_args()

# ======================
# 特征提取函数
# ======================
def extract_features(sequence, tokenizer, model):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None)
    sequences = df.iloc[:, 4].tolist()
    labels = df.iloc[:, 5].tolist()
    return sequences, labels

# ======================
# 主程序
# ======================
def main():
    args = parse_args()

    # 初始化模型和 tokenizer
    print("加载模型中...")
    model = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    # 读取数据
    train_seqs, _ = load_data(args.train_file)
    test_seqs, _ = load_data(args.test_file)

    # 提取特征
    print("提取训练集特征...")
    X_train = [extract_features(seq, tokenizer, model) for seq in tqdm(train_seqs, desc="Train")]
    print("提取测试集特征...")
    X_test = [extract_features(seq, tokenizer, model) for seq in tqdm(test_seqs, desc="Test")]

    # 转换为 numpy 数组
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, args.train_output_name), X_train)
    np.save(os.path.join(args.output_dir, args.test_output_name), X_test)

    print(f"特征已保存到：{args.output_dir}")

if __name__ == "__main__":
    main()
