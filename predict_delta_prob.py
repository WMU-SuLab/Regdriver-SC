#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# 固定随机种子（可选）
torch.manual_seed(12)
np.random.seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== 动态适配的 MLP（支持 BatchNorm） ======
class MLPBN(nn.Module):
    """
    结构：
      [ Linear(in, h1) -> (BN(h1)) -> ReLU -> Dropout ] x (len(hidden_layers))
      -> Linear(last_hidden, out_dim)
    """
    def __init__(self, input_dim, hidden_layers, output_dim=2,
                 use_bn=True, dropout_p=0.5, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_prob_class1(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p1.detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.array([])


def load_meta(file_path: str, has_header: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=0 if has_header else None, dtype=str)
    sub = df.iloc[:, :4].copy()
    sub.columns = ["chr", "start", "end", "regionID"]
    return sub


def align_on_key(meta_pre: pd.DataFrame, meta_post: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["chr", "start", "end", "regionID"]
    meta_pre["_key"] = meta_pre[key_cols].agg("|".join, axis=1)
    meta_post["_key"] = meta_post[key_cols].agg("|".join, axis=1)
    merged = pd.merge(
        meta_pre[["_key"] + key_cols],
        meta_post[["_key"]],
        on="_key",
        how="inner",
        validate="one_to_one"
    )
    merged.drop(columns=["_key"], inplace=True)
    return merged


def parse_args():
    ap = argparse.ArgumentParser(description="突变前/突变后 概率差值推理（自适配 checkpoint 结构）")
    ap.add_argument("--model_path", required=True, help="已训练 MLP 的 .pth 权重路径（state_dict）")
    ap.add_argument("--pre_meta", required=True, help="突变前 bed/tsv（含 chr start end regionID）")
    ap.add_argument("--post_meta", required=True, help="突变后 bed/tsv（含 chr start end regionID）")
    ap.add_argument("--pre_feat", required=True, help="突变前特征 .npy（与 pre_meta 对应）")
    ap.add_argument("--post_feat", required=True, help="突变后特征 .npy（与 post_meta 对应）")
    ap.add_argument("--output", required=True, help="输出文件路径（tsv）")
    ap.add_argument("--has_header", action="store_true", help="若 meta 文件有表头则加此标志")
    ap.add_argument("--batch_size", type=int, default=256, help="预测 batch 大小")
    ap.add_argument("--dropout_p", type=float, default=0.5, help="dropout 概率（用于重建结构）")
    ap.add_argument("--scaler", default=None, help="训练阶段保存的 StandardScaler.pkl；未提供则跳过标准化")
    ap.add_argument("--input_dim", type=int, default=None, help="输入维度覆盖（通常自动从特征推断）")
    return ap.parse_args()


# —— 从 checkpoint 的 state_dict 自动推断网络结构 ——
def infer_arch_from_state_dict(sd: dict):
    """
    返回：
      - linear_indices: [i0, i1, i2, ...]  # 顺序上的 Linear 模块在 nn.Sequential 里的索引
      - bn_after_linear: True/False        # 是否在线性层后紧跟 BatchNorm
      - linear_out_dims: [h1, h2, ..., out_dim]
    规则：
      - 把所有含 ".weight" 的模块索引取出来，再排除掉 batchnorm（其下还会有 running_mean）
      - 线性层索引升序即顺序
      - 对应的 weight shape 的第 0 维即该层的 out_features
      - 如果在某个 linear 紧跟的索引上出现 running_mean/var，则视为有 BN
    """
    # 找出所有形如 net.<idx>.weight 的键
    weight_keys = [k for k in sd.keys() if re.match(r"^net\.\d+\.weight$", k)]
    # 标记哪些 idx 是 batchnorm（BN 会有 running_mean）
    bn_idx = set(int(re.findall(r"\d+", k)[0]) for k in sd.keys() if re.match(r"^net\.\d+\.running_mean$", k))
    # 线性层索引 = 有 weight 但不在 bn_idx 中
    lin_idx = sorted([int(re.findall(r"\d+", k)[0]) for k in weight_keys
                      if int(re.findall(r"\d+", k)[0]) not in bn_idx])

    if not lin_idx:
        raise ValueError("无法从 state_dict 推断线性层索引，请检查 checkpoint。")

    # 推断是否“线性后紧跟 BN”：只要发现 lin_idx 中某个 i 的后面（i+1）在 bn_idx，就认为使用了BN
    use_bn = any((i + 1) in bn_idx for i in lin_idx)

    # 收集每个线性层的 out_features
    out_dims = []
    for i in lin_idx:
        w = sd[f"net.{i}.weight"]
        if w.ndim != 2:
            raise ValueError(f"意外的 weight 维度：net.{i}.weight shape={tuple(w.shape)}")
        out_dims.append(w.shape[0])

    return lin_idx, use_bn, out_dims


def main():
    args = parse_args()

    # 1) 读取 meta 并对齐
    meta_pre = load_meta(args.pre_meta, has_header=args.has_header)
    meta_post = load_meta(args.post_meta, has_header=args.has_header)
    aligned = align_on_key(meta_pre, meta_post)

    # 2) 读取特征
    X_pre = np.load(args.pre_feat)
    X_post = np.load(args.post_feat)
    if args.input_dim is not None:
        input_dim = int(args.input_dim)
    else:
        input_dim = X_pre.shape[1]
    if X_post.shape[1] != input_dim:
        raise ValueError(f"pre/post 特征维度不一致: pre={X_pre.shape}, post={X_post.shape}")

    # 3) 标准化（可选：推理时应与训练保持一致）
    if args.scaler and os.path.isfile(args.scaler):
        scaler: StandardScaler = joblib.load(args.scaler)
        X_pre = scaler.transform(X_pre)
        X_post = scaler.transform(X_post)
        print(f"已加载并应用 StandardScaler: {args.scaler}")
    else:
        print("未提供 scaler，跳过标准化（可能与训练分布不一致）。")

    # 4) 加载 checkpoint（先不建模，先读 state_dict 来“反推结构”）
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]  # 兼容某些保存格式

    # 5) 解析结构
    lin_idx, use_bn, linear_out_dims = infer_arch_from_state_dict(state)
    # 最后一层是分类头，其输出就是 out_dim；前面的都是隐藏层
    if len(linear_out_dims) < 1:
        raise ValueError("未能解析出线性层维度。")
    hidden_layers = linear_out_dims[:-1]
    output_dim = linear_out_dims[-1]

    print(f"[解析 checkpoint] hidden_layers={hidden_layers}, output_dim={output_dim}, use_bn={use_bn}")

    # 6) 按解析的结构重建模型
    model = MLPBN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        use_bn=use_bn,
        dropout_p=args.dropout_p
    ).to(device)

    # 7) 加载权重（严格匹配）
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # 打印出可能的缓冲名或非关键参数名，通常不会影响前向
        print(f"[警告] 加载 state_dict 时存在不完全匹配：\n  missing={missing}\n  unexpected={unexpected}")

    model.eval()

    # 8) 预测
    prob_pre = predict_prob_class1(model, X_pre, batch_size=args.batch_size)
    prob_post = predict_prob_class1(model, X_post, batch_size=args.batch_size)
    if len(prob_pre) != len(prob_post):
        raise ValueError(f"突变前/后的样本数量不一致：pre={len(prob_pre)}, post={len(prob_post)}")

    abs_delta = np.abs(prob_post - prob_pre)

    # 9) 组织输出
    out = aligned.copy()
    out["abs_delta_prob"] = abs_delta
    out = out[["chr", "start", "end", "regionID", "abs_delta_prob"]]

    # 10) 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"✅ 完成：已输出 {len(out)} 行结果到 {args.output}")


if __name__ == "__main__":
    main()
