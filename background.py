#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Background mutation rate model (BMR) — Binomial only
"""

import os
import sys
import logging
import pickle
import pkg_resources  # 若不使用 read_param，可删除
import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import logit

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from sklearn.metrics import r2_score, explained_variance_score

import statsmodels.api as sm  # 必需：用于 Binomial GLM

logger = logging.getLogger("MODEL")



# =========================
# 内联：dataIO 中的函数
# =========================
def read_feature(path, use_features=None):
    """读取特征表 X（TSV/HDF5/XGBoost buffer）
    - 要求存在列 'binID'（作为索引）
    - 如果提供 use_features，则只读取/筛选这些特征
    """
    if path.lower().endswith((".h5", ".hdf5")):
        X = pd.read_hdf(path, "X")
        if use_features is not None:
            X = X.loc[:, use_features]
    elif path.lower().endswith(".buffer"):
        if not _HAS_XGB:
            raise ImportError("读取 .buffer 需要 xgboost，请安装 xgboost 或改用 TSV/HDF5。")
        X = xgb.DMatrix(path)
    else:
        if use_features is not None:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID", usecols=["binID"] + use_features)
        else:
            X = pd.read_csv(path, sep="\t", header=0, index_col="binID")

    if isinstance(X, pd.DataFrame):
        assert len(X.index.values) == len(X.index.unique()), "binID in feature table is not unique."
        na_count = X.isnull().sum()
        if na_count.sum() > 0:
            na_names = na_count.index.values[np.where(na_count > 0)]
            logger.warning("NA values found in features [{}]".format(", ".join(na_names)))
            logger.warning("Fill NA with 0")
            X.fillna(0, inplace=True)
        logger.info("Successfully load {} features for {} bins".format(X.shape[1], X.shape[0]))
        return X
    else:
        # xgb.DMatrix
        logger.info("Successfully load {} features for {} bins".format(X.num_col(), X.num_row()))
        return X


def read_response(path):
    """读取响应表 y（TSV）
    需要列：binID, length, nMut, nSample, N
    """
    y = pd.read_csv(
        path,
        sep="\t",
        header=0,
        index_col="binID",
        usecols=["binID", "length", "nMut", "nSample", "N"],
    )
    assert len(y.index.values) == len(y.index.unique()), "binID in response table is not unique."
    return y


def read_fi(path, cutoff=0.5):
    """读取特征重要性（TSV，列：name, importance），并按阈值过滤"""
    if path is None:
        return None
    fi = pd.read_csv(path, sep="\t", header=0, index_col="name", usecols=["name", "importance"])
    assert len(fi.index.values) == len(fi.index.unique()), "Feature name in feature importance table is not unique."
    keep = (fi.importance >= cutoff).values
    return fi.index.values[keep].tolist()


def save_fi(fi_scores, feature_names, project_name, out_dir):
    """保存特征重要性"""
    os.makedirs(out_dir, exist_ok=True)
    res = pd.DataFrame({"name": feature_names, "importance": fi_scores}, columns=["name", "importance"])
    path = os.path.join(out_dir, project_name + ".feature_importance.tsv")
    logger.info(f"Save feature importance to: {path}")
    res.to_csv(path, sep="\t", index=False)
    return res


def read_param(path=None):
    """读取 xgboost 参数（与本 Binomial 版本无关，仅保留接口兼容）。"""
    path = path if path is not None else pkg_resources.resource_filename(__name__, "xgb_param.pkl")
    with open(path, "rb") as f:
        param = pickle.load(f)
    return param


def save_prediction(ypred, y, project_name, out_dir, model_name):
    """保存训练集预测"""
    os.makedirs(out_dir, exist_ok=True)
    y = y.copy()
    y["pred"] = ypred
    path = os.path.join(out_dir, f"{project_name}.{model_name}.model.train.pred.tsv")
    logger.info(f"Save training prediction to: {path}")
    y.to_csv(path, sep="\t")
    return


def read_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(model, project_name, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{project_name}.{model_name}.model.pkl")
    logger.info(f"Save model to: {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_result(y, project_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    y = y.sort_values(y.columns[-2], ascending=True)
    path = os.path.join(out_dir, f"{project_name}.result.tsv")
    logger.info(f"Save result to: {path}")
    y.to_csv(path, sep="\t")


# =========================
# BMR（Binomial-only）核心
# =========================
def run_bmr(
    model_name: str,
    X_path: str,
    y_path: str,
    fi_cut: float = 0.5,
    fi_path: str = None,
    kfold: int = 3,  # 保留以便后续扩展
    param_path: str = None,  # 保留以便后续扩展
    project_name: str = None,
    out_dir: str = None,
    save_pred: bool = False,
):
    """BMR 入口（仅 Binomial）"""
    if model_name != "Binomial":
        logger.error(f"Only Binomial is supported in this simplified version. Got: {model_name}")
        sys.exit(1)

    # 读取（可选）特征重要性列表
    use_features = read_fi(fi_path, fi_cut)
    run_feature_select = False if use_features else True

    # 读入特征与响应
    X_df = read_feature(X_path, use_features)  # DataFrame
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("This Binomial pipeline expects features in a pandas DataFrame (TSV/HDF5).")
    y = read_response(y_path)  # 需包含列：nMut, length, N
    feature_names = X_df.columns.values

    # 下采样 0 突变元素（极端失衡时才启用）
    pct_zero = y[y.nMut == 0].shape[0] / y.shape[0] * 100
    pct_req = 0.1
    if pct_zero > 99:
        logger.info("{:.2f}% elements have zero mutation. Downsampling to {}%".format(pct_zero, pct_req * 100))
        ct_req = int(y.shape[0] * pct_req)
        y_nonzero = y[y.nMut > 0]
        y_zero = y[y.nMut == 0].sample(n=ct_req, replace=False)
        y = pd.concat([y_nonzero, y_zero])
        del y_nonzero, y_zero

    # 仅保留 length >= 100 且在 X ∩ y 的元素
    use_bins = np.intersect1d(X_df.index.values, y.loc[y.length >= 100, :].index.values)
    logger.info("Use {} bins in model training".format(use_bins.shape[0]))
    X = X_df.loc[use_bins, :].values  # 转为 numpy
    y = y.loc[use_bins, :]

    # Robust 缩放
    X, scaler = scale_data(X)

    # 若未提供 use_features，则进行特征选择（LassoCV + RandomizedLasso/退化方案）
    if run_feature_select:
        alpha = run_lasso(X, y)
        fi_scores = run_rndlasso(X, y, alpha, feature_names)
        fi = save_fi(fi_scores, feature_names, project_name, out_dir)
        keep = (fi.importance >= fi_cut).values
        use_features = fi.name.values[keep]
        # 注意：上面 fi_scores 是按 feature_names 对齐的，因此这里根据 name 过滤列
        X = X[:, np.isin(feature_names, use_features)]

    # 训练 Binomial GLM
    model = run_glm_binomial(X, y)

    # 预测为：拟合概率 * 试验次数（length * N）
    yhat = (model.fittedvalues * y.length * y.N).values

    # 评价与(可选)保存预测
    report_metrics(yhat, y.nMut.values)
    if save_pred:
        save_prediction(yhat, y, project_name, out_dir, "Binomial")

    # 分散度检测
    pval, theta = dispersion_test(yhat, y.nMut.values)

    # 清理模型缓存
    with warnings.catch_warnings():
        model.remove_data()

    # 保存模型
    model_info = {
        "model_name": "Binomial",
        "model": model,
        "scaler": scaler,
        "pval_dispersion": pval,
        "theta": theta,
        "feature_names": feature_names,
        "use_features": use_features,
        "project_name": project_name,
    }
    save_model(model_info, project_name, out_dir, "Binomial")
    logger.info("Job done!")


def scale_data(X, scaler=None):
    """鲁棒缩放（对异常值不敏感）"""
    if scaler is not None:
        return scaler.transform(X), scaler
    scaler = RobustScaler(copy=False)
    scaler.fit(X)
    return scaler.transform(X), scaler


def run_lasso(X, y, max_iter=3000, cv=5, n_threads=1):
    """使用 LassoCV 选择正则化强度 alpha"""
    logger.info("Implementing LassoCV with {} iter. and {}-fold CV".format(max_iter, cv))
    y_logit = logit((y.nMut + 0.5) / (y.length * y.N))
    use_ix = np.random.choice(y_logit.shape[0], min(300000, y_logit.shape[0]), replace=True)
    Xsub = X[use_ix, :]
    ysub = y_logit[use_ix]
    reg = LassoCV(max_iter=max_iter, cv=cv, copy_X=False, n_jobs=n_threads)
    lassocv = reg.fit(Xsub, ysub)
    logger.info("LassoCV alpha = {}".format(lassocv.alpha_))
    return lassocv.alpha_


def run_rndlasso(X, y, alpha, feature_names, n_resampling=500, sample_fraction=0.1, n_threads=1):
    """Randomized Lasso 获取特征重要性
    - 若环境无 RandomizedLasso，则回退到基于 Lasso 权重的简单重要性（|coef|）。
    """
    y_logit = logit((y.nMut + 0.5) / (y.length * y.N))

    if _HAS_RNDLASSO:
        logger.info(
            "Implementing Randomized Lasso with alpha={}, n_resampling={} and sample_fraction={}".format(
                alpha, n_resampling, sample_fraction
            )
        )
        rnd = RandomizedLasso(
            alpha=alpha,
            n_resampling=n_resampling,
            sample_fraction=sample_fraction,
            selection_threshold=1e-3,
            max_iter=3000,
            normalize=False,
            n_jobs=n_threads,
        )
        rnd.fit(X, y_logit)
        fi_scores = rnd.scores_
    else:
        logger.warning("RandomizedLasso 不可用，回退为 Lasso(coef_) 的绝对值作为重要性。")
        base = LassoCV(max_iter=3000, cv=5, copy_X=False)
        base.fit(X, y_logit)
        coefs = getattr(base, "coef_", np.zeros(len(feature_names)))
        fi_scores = np.abs(coefs)

    # 与 feature_names 对齐的一维数组
    if fi_scores is None or len(fi_scores) != len(feature_names):
        logger.warning("特征重要性长度异常，回退为均匀重要性。")
        fi_scores = np.ones(len(feature_names), dtype=float)
    return fi_scores


def run_glm_binomial(X, y):
    """训练 Binomial GLM（statsmodels）"""
    logger.info("Building binomial GLM")
    X = np.c_[X, np.ones(X.shape[0])]  # 手动加常数项
    y_binom = np.zeros((y.shape[0], 2), dtype=np.int_)
    y_binom[:, 0] = y.nMut
    y_binom[:, 1] = y.length * y.N - y.nMut
    glm = sm.GLM(y_binom, X, family=sm.families.Binomial())
    model = glm.fit()
    return model


def report_metrics(yhat, y):
    """基本训练集指标"""
    r2 = r2_score(y, yhat)
    var_exp = explained_variance_score(y, yhat)
    r = stats.pearsonr(yhat, y)[0]
    logger.info(
        "Model metrics for training set: r2={:.2f}, Variance explained={:.2f}, Pearson'r={:.2f}".format(
            r2, var_exp, r
        )
    )


def dispersion_test(yhat, y, k=100):
    """基于回归的分散度检验（Poisson 假设下的过度离散）"""
    theta = 0.0
    pval = 0.0
    for i in range(k):
        y_sub, yhat_sub = resample(y, yhat, random_state=i)
        aux = (np.power((y_sub - yhat_sub), 2) - yhat_sub) / yhat_sub  # Poisson 回归下的辅助回归量
        mod = sm.OLS(aux, yhat_sub)
        res = mod.fit()
        theta += res.params[0]
        pval += res.pvalues[0]
    theta /= k
    pval /= k
    return pval, theta


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

    parser = argparse.ArgumentParser(description="BMR (Binomial-only) — single-file version")
    parser.add_argument("--model_name", type=str, default="Binomial")
    parser.add_argument("--X_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--fi_cut", type=float, default=0.5)
    parser.add_argument("--fi_path", type=str, default=None)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--save_pred", action="store_true")
    args = parser.parse_args()

    run_bmr(
        model_name=args.model_name,
        X_path=args.X_path,
        y_path=args.y_path,
        fi_cut=args.fi_cut,
        fi_path=args.fi_path,
        project_name=args.project_name,
        out_dir=args.out_dir,
        save_pred=args.save_pred,
    )
