#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified pipeline (no intermediate files on disk):

1) From --regions_bed (chrom start end binID score) and --mutations_maf, build mutation-level
   rank/phred purely in memory, then intersect with --regions_bed1 (merged bed) to compute
   per-bin mutation_impact_score = mean(phred), filling 0 for bins without overlaps (all in memory).
2) Run Binomial-only inference; use the in-memory mutation_impact_score table as functional scores.

Outputs: only final inference results under --out_dir (no intermediate TSVs).
"""

import os
import sys
import math
import logging
import pickle
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pybedtools import BedTool

from scipy.stats import binomtest, pearsonr
from sklearn.metrics import r2_score, explained_variance_score
from statsmodels.sandbox.stats.multicomp import multipletests  # BH-FDR

logger = logging.getLogger("INFER")

# --------------------------------------------------
# sklearn 旧版本兼容（在反序列化前注入别名）
# --------------------------------------------------
def _install_sklearn_compat_shims():
    import sys as _sys, types as _types
    try:
        import sklearn.preprocessing._data as _data
    except Exception:
        return
    shim = _types.ModuleType("sklearn.preprocessing.data")
    shim.__dict__.update(_data.__dict__)
    _sys.modules["sklearn.preprocessing.data"] = shim


# =========================
# I/O utils (inlined)
# =========================
def read_model(path: str):
    _install_sklearn_compat_shims()
    with open(path, "rb") as f:
        return pickle.load(f)

def read_feature(path: str, use_features=None) -> pd.DataFrame:
    """Read X table (TSV/HDF5). Must contain 'binID' index."""
    if path.lower().endswith((".h5", ".hdf5")):
        X = pd.read_hdf(path, "X")
        if use_features is not None:
            X = X.loc[:, use_features]
    else:
        if use_features is not None:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID",
                            usecols=["binID"] + list(use_features))
        else:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID")
    assert len(X.index) == len(X.index.unique()), "binID in feature table is not unique."
    if X.isnull().values.any():
        logger.warning("NA values found in features; filling with 0.")
        X = X.fillna(0)
    return X

def read_response(path: str) -> pd.DataFrame:
    """Read y (TSV). Needs columns: binID, length, nMut, nSample, N"""
    y = pd.read_csv(
        path, sep="\t", header=0, index_col="binID",
        usecols=["binID", "length", "nMut", "nSample", "N"]
    )
    assert len(y.index) == len(y.index.unique()), "binID in response table is not unique."
    return y

# =========================
# Step A: build mutation_impact_score in memory (no files)
# =========================
def read_regions_scored(path_bed: str) -> pd.DataFrame:
    """--regions_bed: chrom start end binID score"""
    return pd.read_csv(
        path_bed, sep="\t", header=None,
        names=["chrom","start","end","binID","score"],
        dtype={"chrom":str,"start":int,"end":int,"binID":str,"score":float},
    )

def read_region_merged(path_bed: str) -> pd.DataFrame:
    """--regions_bed1: chrom start end binID"""
    return pd.read_csv(
        path_bed, sep="\t", header=None,
        names=["chrom","start","end","binID"],
        dtype={"chrom":str,"start":int,"end":int,"binID":str},
    )

def read_mutations(path_maf: str) -> pd.DataFrame:
    """--mutations_maf: chrom start end ref alt donor"""
    return pd.read_csv(
        path_maf, sep="\t", header=None,
        names=["chrom","start","end","ref","alt","donor"],
        dtype={"chrom":str,"start":int,"end":int,"ref":str,"alt":str,"donor":str},
    )

def intersect_mut_with_regions(muts_df: pd.DataFrame, regs_df: pd.DataFrame) -> pd.DataFrame:
    """Return columns: chrom start end ref alt donor binID score"""
    mut_bed = BedTool.from_dataframe(muts_df[["chrom","start","end","ref","alt","donor"]])
    reg_bed = BedTool.from_dataframe(regs_df[["chrom","start","end","binID","score"]])
    inter = mut_bed.intersect(reg_bed, wa=True, wb=True)
    inter_df = inter.to_dataframe(
        names=["chrom","start","end","ref","alt","donor","chrom_r","start_r","end_r","binID","score"]
    )
    inter_df = inter_df[["chrom","start","end","ref","alt","donor","binID","score"]].copy()
    inter_df["score"] = inter_df["score"].astype(float)
    return inter_df

def make_mut_with_scores_df(inter_df: pd.DataFrame) -> pd.DataFrame:
    """
    From intersections (with score), build mutation-level table in memory:
      sort by score desc; rank starts at 1; phred = -10*log10(rank/total).
    Return columns: chrom start end ref alt donor score rank phred  (no binID)
    """
    if inter_df.empty:
        return pd.DataFrame(columns=["chrom","start","end","ref","alt","donor","score","rank","phred"])
    df = inter_df.sort_values(
        by=["score","chrom","start","end"],
        ascending=[False,True,True,True],
        kind="mergesort",
    ).reset_index(drop=True)
    total = df.shape[0]
    df["rank"] = df.index + 1
    df["phred"] = df["rank"].apply(lambda r: -10.0 * math.log10(r/total))
    return df[["chrom","start","end","ref","alt","donor","score","rank","phred"]]

def build_region_mutation_impact_score_df(
    mut_with_scores_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    prefixes: Optional[Sequence[str]] = ("silencer","enhancer","promoter"),
) -> pd.DataFrame:
    """
    Intersect mut_with_scores (A) with merged (B) in memory and compute mean phred per binID,
    then left-join against all bins and fill 0 for missing.
    Return: DataFrame indexed by binID with a single numeric column 'mutation_impact_score'.
    """
    a_cols = ["chrom","start","end","ref","alt","donor","score","rank","phred"]
    a_bed = BedTool.from_dataframe(mut_with_scores_df[a_cols])
    b_cols = ["chrom","start","end","binID"]
    b_bed = BedTool.from_dataframe(merged_df[b_cols])
    inter = a_bed.intersect(b_bed, wa=True, wb=True)
    inter_cols = a_cols + ["chrom_r","start_r","end_r","binID"]
    inter_df = inter.to_dataframe(names=inter_cols)

    if inter_df.empty:
        out = merged_df[["binID"]].drop_duplicates().copy()
        out["mutation_impact_score"] = 0.0
        if prefixes:
            out = out[out["binID"].astype(str).str.startswith(tuple(prefixes))]
        out = out.set_index("binID")
        return out[["mutation_impact_score"]]

    inter_df["phred"] = inter_df["phred"].astype(float)
    agg = (
        inter_df.groupby("binID", as_index=False)["phred"]
        .mean()
        .rename(columns={"phred":"mutation_impact_score"})
    )
    all_bins = merged_df[["binID"]].drop_duplicates()
    out = all_bins.merge(agg, on="binID", how="left")
    out["mutation_impact_score"] = out["mutation_impact_score"].fillna(0.0)
    if prefixes:
        out = out[out["binID"].astype(str).str.startswith(tuple(prefixes))]
    out = out.set_index("binID")
    return out[["mutation_impact_score"]]

# =========================
# Step B: Binomial-only inference
# =========================
def report_metrics(yhat: np.ndarray, y: np.ndarray):
    r2 = r2_score(y, yhat)
    var_exp = explained_variance_score(y, yhat)
    r = float(pearsonr(yhat, y)[0]) if len(yhat) > 1 else np.nan
    logger.info(f"Test-set metrics: R2={r2:.3f}, VarExplained={var_exp:.3f}, Pearson r={r:.3f}")

def bh_fdr(pvals: np.ndarray):
    return multipletests(pvals, method="fdr_bh")[1]

def binomial_burden_test(count, pred, offset):
    p = np.clip(pred / offset, 1e-12, 1 - 1e-12)
    # 使用 binomtest（非弃用）
    pvals = np.array([binomtest(int(x), n=int(n), p=float(pi), alternative="greater").pvalue
                      for x, n, pi in zip(count, offset, p)])
    return pvals

# -------- 关键新增：推理阶段的特征对齐管道 --------
def prepare_X_for_model(X_df: pd.DataFrame, model_dict: dict) -> np.ndarray:
    """
    训练->推理 的对齐：
      1) 按训练时 scaler 拟合的完整列集 feature_names 对齐顺序；缺列补 0
      2) 用存下来的 RobustScaler.transform 缩放
      3) 再切到训练时用于 GLM 的 use_features
    返回：(n_samples, n_used_features)
    """
    all_feats = [str(f) for f in model_dict["feature_names"]]
    use_feats = model_dict.get("use_features")
    if use_feats is None or len(use_feats) == 0:
        use_feats = all_feats
    use_feats = [str(f) for f in use_feats]

    # 补缺列并重排
    Xw = X_df.copy()
    for f in all_feats:
        if f not in Xw.columns:
            Xw[f] = 0.0
    Xw = Xw.loc[:, all_feats]

    # 缩放
    scaler = model_dict["scaler"]
    X_scaled = scaler.transform(Xw.values)

    # 切 use_features（保持训练顺序）
    feat_index = {f: i for i, f in enumerate(all_feats)}
    idx = [feat_index[f] for f in use_feats]
    X_used = X_scaled[:, idx]

    # 形状自检：GLM 训练时手动加了常数项，所以 params 长度 = used_features + 1
    n_params = len(np.asarray(model_dict["model"].params))
    if X_used.shape[1] + 1 != n_params:
        raise RuntimeError(
            f"特征维度与 GLM 参数不匹配：X_used.shape[1]+1 = {X_used.shape[1]+1}, "
            f"len(params) = {n_params}. "
            f"检查 use_features 与保存的模型是否一致。"
        )
    return X_used

def predict_with_glm_binomial(X_used: np.ndarray, y_df: pd.DataFrame, model_dict: dict) -> np.ndarray:
    # 训练时加了常数项，这里同样加
    Xc = np.c_[X_used, np.ones(X_used.shape[0])]
    n_params = len(np.asarray(model_dict["model"].params))
    if Xc.shape[1] != n_params:
        raise RuntimeError(f"预测矩阵列数({Xc.shape[1]})与模型参数数({n_params})不一致。")
    prob = np.array(model_dict["model"].predict(Xc))
    pred = prob * (y_df.length.values * y_df.N.values)
    return pred

def run_inference_with_mutation_impact_score(
    model_path: str,
    X_path: str,
    y_path: str,
    fs_df: Optional[pd.DataFrame],
    use_gmean: bool,
    project_name: str,
    out_dir: str,
):
    model = read_model(model_path)
    if model.get("model_name") != "Binomial":
        logger.error(f"Loaded model is not Binomial. Got: {model.get('model_name')}")
        sys.exit(1)

    # 注意：这里不再提前切 model["feature_names"]，由 prepare_X_for_model 统一处理
    X_df = read_feature(X_path)
    y = read_response(y_path)

    use_bins = np.intersect1d(X_df.index.values, y.index.values)
    X_df = X_df.loc[use_bins, :]
    y = y.loc[use_bins, :]

    # 统一到训练列集 -> 缩放 -> 切 use_features
    X_used = prepare_X_for_model(X_df, model)

    # 预测 & 评估
    y["nPred"] = predict_with_glm_binomial(X_used, y, model)
    report_metrics(y["nPred"].values, y["nMut"].values)

    # 原始负担检验
    count = np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut
    offset = y.length * y.N + 1
    y["raw_p"] = binomial_burden_test(count, y["nPred"].values, offset.values)
    y["raw_q"] = bh_fdr(y["raw_p"].values)

    # 功能注释加权
    y = functional_adjustment_binomial(y, fs_df, use_gmean)

    # 仅保存最终结果
    os.makedirs(out_dir, exist_ok=True)
    y = y.sort_values(y.columns[-2], ascending=True)
    outp = os.path.join(out_dir, f"{project_name}.result.tsv")
    y.to_csv(outp, sep="\t")
    logger.info(f"Saved final result to: {outp}")

# =========================
# 功能加权（维持你的原实现）
# =========================
def functional_adjustment_binomial(
    y: pd.DataFrame,
    fs_df: Optional[pd.DataFrame],
    use_gmean: bool = True,
):
    if fs_df is None or fs_df.shape[1] == 0:
        return y

    if fs_df.index.name != "binID":
        fs_df = fs_df.copy()
        fs_df.index.name = "binID"

    fs_df = fs_df.select_dtypes(include=[np.number])
    if fs_df.shape[1] == 0:
        return y

    cutoff_fixed = 0.5
    threshold = -10.0 * np.log10(cutoff_fixed)
    logger.info(f"Functional adjustment: fixed cutoff=0.5; threshold={threshold:.4f}")

    y = y.join(fs_df, how="left")
    y.fillna({col: 0.0 for col in fs_df.columns}, inplace=True)

    ct = 0
    avg_weight = np.zeros(y.shape[0])

    for score in fs_df.columns:
        wcol = f"{score}_weight"
        ncol = f"{score}_nMut"
        pcol = f"{score}_p"
        qcol = f"{score}_q"

        y[wcol] = y[score] / threshold
        y.loc[y.raw_q > 1, wcol] = 999
        y[wcol].fillna(888, inplace=True)

        y[ncol] = y[wcol] * (np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut)

        mask = (y.raw_q <= 1)
        y[pcol] = y.raw_p
        if mask.any():
            count = y.loc[mask, ncol]
            offset = y.loc[mask, "length"] * y.loc[mask, "N"] + 1
            y.loc[mask, pcol] = binomial_burden_test(count, y.loc[mask, "nPred"], offset)
        y[qcol] = bh_fdr(y[pcol])

        avg_weight += y[wcol].fillna(0).values
        ct += 1

    if ct >= 2:
        y["avg_weight"] = avg_weight / ct
        y["avg_nMut"] = y["avg_weight"] * (np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut)
        mask = (y.raw_q <= 0.25)
        y["avg_p"] = y.raw_p
        if mask.any():
            count = y.loc[mask, "avg_nMut"]
            offset = y.loc[mask, "length"] * y.loc[mask, "N"] + 1
            y.loc[mask, "avg_p"] = binomial_burden_test(count, y.loc[mask, "nPred"], offset)
        y["avg_q"] = bh_fdr(y["avg_p"])

    return y

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%(m/%d/%Y %H:%M:%S)s",
    )

    p = argparse.ArgumentParser(
        description="Build region mutation_impact_score in memory (no temp files) and run Binomial inference using it as functional scores."
    )
    # Inference I/O
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--X_path", type=str, required=True)
    p.add_argument("--y_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--project_name", type=str, required=True)
    p.add_argument("--no_gmean", action="store_true")

    # Inputs to build mutation_impact_score
    p.add_argument("--regions_bed", type=str, required=True, help="chrom start end binID score")
    p.add_argument("--mutations_maf", type=str, required=True, help="chrom start end ref alt donor")
    p.add_argument("--regions_bed1", type=str, required=True, help="chrom start end binID (universe)")
    p.add_argument("--keep_prefix", type=str, default="silencer,enhancer,promoter",
                   help="Prefixes of binID to keep; 'ALL' for no filtering")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step A: build mutation_impact_score purely in memory
    regs_score = read_regions_scored(args.regions_bed)
    merged_df  = read_region_merged(args.regions_bed1)
    muts       = read_mutations(args.mutations_maf)

    inter = intersect_mut_with_regions(muts, regs_score)
    mut_with_scores_df = make_mut_with_scores_df(inter)

    prefixes = None if args.keep_prefix.strip().upper() == "ALL" else \
        tuple([p.strip() for p in args.keep_prefix.split(",") if p.strip()])

    mutation_impact_score_df = build_region_mutation_impact_score_df(mut_with_scores_df, merged_df, prefixes)

    # Step B: inference with mutation_impact_score (in-memory)
    run_inference_with_mutation_impact_score(
        model_path=args.model_path,
        X_path=args.X_path,
        y_path=args.y_path,
        fs_df=mutation_impact_score_df,
        use_gmean=(not args.no_gmean),
        project_name=args.project_name,
        out_dir=args.out_dir,
    )
