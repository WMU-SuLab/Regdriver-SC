#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Making inferences about driver (Binomial-only, single-file)

Supports:
1) Predict nMut with Binomial GLM
2) Binomial burden test
3) Functional-score–adjusted test (Binomial, fixed cutoff = 0.5 for all scores)
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from scipy.stats import binom_test, pearsonr
from sklearn.metrics import r2_score, explained_variance_score
from statsmodels.sandbox.stats.multicomp import multipletests  # BH-FDR

logger = logging.getLogger("INFER")


# =========================
# I/O utils (inlined)
# =========================
def read_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_feature(path: str, use_features=None) -> pd.DataFrame:
    """Read X table (TSV/HDF5). Must contain 'binID' index."""
    if path.lower().endswith((".h5", ".hdf5")):
        X = pd.read_hdf(path, "X")
        if use_features is not None:
            X = X.loc[:, use_features]
    else:
        # TSV or compressed TSV
        if use_features is not None:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID", usecols=["binID"] + list(use_features))
        else:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID")
    # sanity checks & NA handling
    assert len(X.index) == len(X.index.unique()), "binID in feature table is not unique."
    na_cnt = X.isnull().sum()
    if na_cnt.sum() > 0:
        bad = na_cnt[na_cnt > 0].index.tolist()
        logger.warning(f"NA values found in features [{', '.join(bad)}]; filling NA with 0.")
        X = X.fillna(0)
    logger.info(f"Loaded {X.shape[1]} features for {X.shape[0]} bins from {path}")
    return X


def read_response(path: str) -> pd.DataFrame:
    """Read y (TSV). Needs columns: binID, length, nMut, nSample, N"""
    y = pd.read_csv(
        path, sep="\t", header=0, index_col="binID",
        usecols=["binID", "length", "nMut", "nSample", "N"]
    )
    assert len(y.index) == len(y.index.unique()), "binID in response table is not unique."
    return y


def read_fs(path: str, fs_cut: dict | None = None) -> pd.DataFrame:
    """Read functional score table (TSV).
    If fs_cut is None -> read all columns except 'binID'.
    Otherwise read only the specified score columns.
    """
    if fs_cut is None:
        fs = pd.read_csv(path, sep="\t", header=0, index_col="binID")
        # drop any non-numeric columns (safety), keep score-like columns
        for col in list(fs.columns):
            if not np.issubdtype(fs[col].dtype, np.number):
                fs.drop(columns=[col], inplace=True)
    else:
        cols = ["binID"] + list(fs_cut.keys())
        fs = pd.read_csv(path, sep="\t", header=0, index_col="binID", usecols=cols)
    assert len(fs.index) == len(fs.index.unique()), "binID in functional score table is not unique."
    return fs


def save_result(y: pd.DataFrame, project_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # sort by penultimate column for compatibility with original pipeline
    y = y.sort_values(y.columns[-2], ascending=True)
    outp = os.path.join(out_dir, f"{project_name}.result.tsv")
    y.to_csv(outp, sep="\t")
    logger.info(f"Saved results to: {outp}")


# =========================
# Core inference (Binomial)
# =========================
def scale_data_with_model_scaler(X: np.ndarray, scaler):
    """Use scaler stored in the trained model (no fitting here)."""
    return scaler.transform(X)


def report_metrics(yhat: np.ndarray, y: np.ndarray):
    r2 = r2_score(y, yhat)
    var_exp = explained_variance_score(y, yhat)
    r = float(pearsonr(yhat, y)[0]) if len(yhat) > 1 else np.nan
    logger.info(f"Test-set metrics: R2={r2:.3f}, VarExplained={var_exp:.3f}, Pearson r={r:.3f}")


def predict_with_glm_binomial(X: np.ndarray, y_df: pd.DataFrame, model_dict: dict) -> np.ndarray:
    """
    pred = P(success) * trials = predict_proba * (length * N)
    model_dict['model'] is a statsmodels GLMResults object (Binomial).
    """
    Xc = np.c_[X, np.ones(X.shape[0])]                # add constant column
    prob = np.array(model_dict["model"].predict(Xc))  # P(success)
    pred = prob * (y_df.length.values * y_df.N.values)
    return pred


def binomial_burden_test(count, pred, offset):
    """
    One-sided binomial test (greater). n=offset, p=pred/offset.
    """
    p = np.clip(pred / offset, 1e-12, 1 - 1e-12)
    pvals = np.array([
        binom_test(x, n=int(n), p=float(pi), alternative="greater")
        for x, n, pi in zip(count, offset, p)
    ])
    return pvals


def bh_fdr(pvals: np.ndarray):
    """Benjamini–Hochberg FDR"""
    return multipletests(pvals, method="fdr_bh")[1]


def functional_adjustment_binomial(y: pd.DataFrame, fs_path: str | None, use_gmean=True):
    """
    Functional adjustment with fixed cutoff = 0.5 for ALL scores in fs_path.

    For each score column S in fs file:
      - cutoff = 0.5 (fixed)
      - threshold = -10*log10(0.5) ≈ 3.0103
      - weight = S / threshold
      - recompute weighted count for near-significant elements (raw_q <= 1)
      - re-run binomial burden test → S_p, BH-FDR → S_q

    If >= 2 scores exist, compute an 'avg_weight' channel (raw_q <= 0.25).
    """
    if fs_path is None:
        return y

    # 读取全部功能分数列（自动过滤非数值）
    fs = read_fs(fs_path, fs_cut=None)
    if fs.shape[1] == 0:
        logger.warning("Functional score file has no numeric score columns; skip functional adjustment.")
        return y

    # 固定 cutoff = 0.5
    cutoff_fixed = 0.5
    threshold = -10.0 * np.log10(cutoff_fixed)  # ~= 3.0103
    logger.info(f"Functional adjustment uses a fixed cutoff = {cutoff_fixed} for ALL scores.")

    # 合并
    y = y.join(fs)

    ct = 0
    avg_weight = np.zeros(y.shape[0])

    for score in fs.columns:
        logger.info(f"Functional score: {score} (fixed cutoff=0.5)")
        wcol = f"{score}_weight"
        y[wcol] = y[score] / threshold
        y.loc[y.raw_q > 1, wcol] = 999
        y[wcol].fillna(888, inplace=True)

        ncol = f"{score}_nMut"
        y[ncol] = y[wcol] * (np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut)

        # recompute only for near-sig (raw_q <= 1)
        mask = (y.raw_q <= 1)
        pcol = f"{score}_p"
        qcol = f"{score}_q"
        y[pcol] = y.raw_p
        if mask.any():
            count = y.loc[mask, ncol]
            offset = y.loc[mask, "length"] * y.loc[mask, "N"] + 1
            y.loc[mask, pcol] = binomial_burden_test(count, y.loc[mask, "nPred"], offset)
        y[qcol] = bh_fdr(y[pcol])

        avg_weight += y[wcol].fillna(0).values
        ct += 1

    if ct >= 2:
        logger.info("Using average weights across scores (fixed cutoff=0.5)")
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


def make_inference(
    model_path: str,
    X_path: str,
    y_path: str,
    fs_path: str | None = None,
    test_method: str = "binomial",
    use_gmean: bool = True,
    project_name: str | None = None,
    out_dir: str = "./output",
):
    """Main wrapper (Binomial-only, fixed functional cutoff)."""
    if test_method.lower() != "binomial":
        logger.error(f"Only 'binomial' is supported. Got: {test_method}")
        sys.exit(1)

    # load model
    model = read_model(model_path)
    if model.get("model_name") != "Binomial":
        logger.error(f"Loaded model is not Binomial. Got: {model.get('model_name')}")
        sys.exit(1)

    if project_name is None:
        project_name = model["project_name"]

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Outputs → {out_dir} (prefix: {project_name})")

    # load data
    # keep the same feature order as in training, then select used features
    X_df = read_feature(X_path, use_features=None)
    X_df = X_df.loc[:, model["feature_names"]]
    y = read_response(y_path)

    # align by intersection
    use_bins = np.intersect1d(X_df.index.values, y.index.values)
    X = X_df.loc[use_bins, :].values
    y = y.loc[use_bins, :]

    # scale → select features (as in training)
    X = scale_data_with_model_scaler(X, model["scaler"])
    X = X[:, np.isin(model["feature_names"], model["use_features"])]

    # predict expected mutation count
    y["nPred"] = predict_with_glm_binomial(X, y, model)

    # metrics
    report_metrics(y["nPred"].values, y["nMut"].values)

    # burden test (Binomial)
    count = np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut
    offset = y.length * y.N + 1
    y["raw_p"] = binomial_burden_test(count, y["nPred"].values, offset.values)
    y["raw_q"] = bh_fdr(y["raw_p"].values)

    # functional adjustment with fixed cutoff=0.5
    y = functional_adjustment_binomial(y, fs_path, use_gmean)

    # save
    save_result(y, project_name, out_dir)
    logger.info("Done.")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Inference (Binomial-only, single-file; functional cutoff fixed at 0.5)")
    p.add_argument("--model_path", type=str, required=True, help="Path to trained Binomial model .pkl")
    p.add_argument("--X_path", type=str, required=True, help="Path to features (TSV/HDF5)")
    p.add_argument("--y_path", type=str, required=True, help="Path to response (TSV)")
    p.add_argument("--fs_path", type=str, default=None, help="(Optional) path to functional scores TSV")
    p.add_argument("--test_method", type=str, default="binomial", help="Must be 'binomial'")
    p.add_argument("--no_gmean", action="store_true", help="Use raw nMut instead of sqrt(nMut*nSample)")
    p.add_argument("--project_name", type=str, default=None, help="Output prefix; defaults to model.project_name")
    p.add_argument("--out_dir", type=str, default="./output", help="Output directory")

    args = p.parse_args()

    make_inference(
        model_path=args.model_path,
        X_path=args.X_path,
        y_path=args.y_path,
        fs_path=args.fs_path,
        test_method=args.test_method,
        use_gmean=(not args.no_gmean),
        project_name=args.project_name,
        out_dir=args.out_dir,
    )
