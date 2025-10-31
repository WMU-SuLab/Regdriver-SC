import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib  # pip install joblib

# 固定随机种子（可选）
torch.manual_seed(12)
np.random.seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== 与训练时一致的 MLP 结构 ======
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 256, 128], output_dim=2, activation=nn.ReLU, dropout_p=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), activation(), nn.Dropout(dropout_p)]
            prev_dim = h
        layers += [nn.Linear(prev_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_prob_class1(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """返回每条样本的正类（class=1）概率"""
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
    """
    读取bed/tsv，返回仅包含元信息列：
    假定前4列分别为：chr, start, end, regionID（元件id）
    如果你的列顺序不同，请在这里调整。
    """
    df = pd.read_csv(file_path, sep="\t", header=0 if has_header else None, dtype=str)
    # 仅取前四列并重命名
    sub = df.iloc[:, :4].copy()
    sub.columns = ["chr", "start", "end", "regionID"]
    return sub


def align_on_key(meta_pre: pd.DataFrame, meta_post: pd.DataFrame) -> pd.DataFrame:
    """
    按 (chr, start, end, regionID) 进行内连接对齐，避免两文件顺序不一致。
    """
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
    ap = argparse.ArgumentParser(description="突变前/突变后 概率差值推理（输出绝对值）")
    ap.add_argument("--model_path", required=True, help="已训练 MLP 的 .pth 权重路径")
    ap.add_argument("--pre_meta", required=True, help="突变前bed/tsv（包含 chr start end regionID 以及其他列）")
    ap.add_argument("--post_meta", required=True, help="突变后bed/tsv（包含 chr start end regionID 以及其他列）")
    ap.add_argument("--pre_feat", required=True, help="突变前特征 .npy（与 pre_meta 对应）")
    ap.add_argument("--post_feat", required=True, help="突变后特征 .npy（与 post_meta 对应）")
    ap.add_argument("--output", required=True, help="输出文件路径（tsv）")
    ap.add_argument("--has_header", action="store_true", help="若 meta 文件有表头则加此标志")
    ap.add_argument("--batch_size", type=int, default=256, help="预测batch大小")
    ap.add_argument("--hidden_layers", default="256,256,128", help="隐藏层，如 256,256,128")
    ap.add_argument("--dropout_p", type=float, default=0.5, help="dropout 概率")
    ap.add_argument("--scaler", default=None, help="训练阶段保存的 StandardScaler.pkl；未提供则跳过标准化")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) 读取元信息并对齐（仅保留 chr start end regionID）
    meta_pre = load_meta(args.pre_meta, has_header=args.has_header)
    meta_post = load_meta(args.post_meta, has_header=args.has_header)

    # 2) 读取特征并检查维度
    X_pre = np.load(args.pre_feat)
    X_post = np.load(args.post_feat)

    # 尝试按 key 对齐行（更稳健）；若两文件是严格同顺序，也可以直接跳过这一步
    aligned = align_on_key(meta_pre, meta_post)  # 只输出公共交集
    # 根据 aligned 的顺序重新索引特征：
    # 这里要求 meta_pre/post 与特征在读取时一一对应（行号一致）
    # 如果你的特征与行顺序不一致，需要在保存特征时带上索引再对齐（此处假设一致）
    # 因此我们用对齐后的索引去过滤 meta_* 并重建行顺序
    key_cols = ["chr", "start", "end", "regionID"]
    meta_pre["_key"] = meta_pre[key_cols].agg("|".join, axis=1)
    meta_post["_key"] = meta_post[key_cols].agg("|".join, axis=1)
    aligned["_key"] = aligned[key_cols].agg("|".join, axis=1)
    # 获取在原meta中的位置索引
    pre_idx = meta_pre.set_index("_key").loc[aligned["_key"]].index
    post_idx = meta_post.set_index("_key").loc[aligned["_key"]].index
    # 将索引转为行号（由于上面用的是索引键，直接按顺序取特征即可）
    # 假设 meta 与特征顺序一致，这里直接用 numpy 的切片等价于“按aligned顺序重排”
    # 若需严格映射，请在保存特征时记录并使用原行号。

    # 3) 标准化（推理阶段应使用训练时的 scaler）
    if args.scaler and os.path.isfile(args.scaler):
        scaler: StandardScaler = joblib.load(args.scaler)
        X_pre = scaler.transform(X_pre)
        X_post = scaler.transform(X_post)
        print(f"已加载并应用 StandardScaler: {args.scaler}")
    else:
        print("未提供 scaler，跳过标准化（可能与训练分布不一致）。")

    # 4) 初始化并加载模型
    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
    input_dim = X_pre.shape[1]
    model = MLP(input_dim=input_dim, hidden_layers=hidden_layers, dropout_p=args.dropout_p).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) 预测突变前/后的正类概率
    prob_pre = predict_prob_class1(model, X_pre, batch_size=args.batch_size)
    prob_post = predict_prob_class1(model, X_post, batch_size=args.batch_size)

    if len(prob_pre) != len(prob_post):
        raise ValueError(f"突变前/后的样本数量不一致：pre={len(prob_pre)}, post={len(prob_post)}")

    # 6) 计算绝对差值
    abs_delta = np.abs(prob_post - prob_pre)

    # 7) 组织输出（仅 chr start end regionID + abs_delta_prob）
    out = aligned.copy()
    out["abs_delta_prob"] = abs_delta
    out = out[["chr", "start", "end", "regionID", "abs_delta_prob"]]

    # 8) 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"✅ 完成：已输出 {len(out)} 行结果到 {args.output}")


if __name__ == "__main__":
    main()
