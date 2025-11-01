#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Background mutation rate model (BMR)

Two types of BMR model are supported:
1) Randomized Lasso + GLM   (fallback to LassoCV(coef_) if RandomizedLasso is unavailable)
2) Gradient Boosting Machine (XGBoost)

示例:
python background.py \
  --X_path /path/data/merge_features/blood_all.bed \
  --y_path /path/data/response/blood.tsv \
  --out_dir /path/output/background_Binomial \
  --project_name blood
"""

import os
import sys
import logging
import warnings
import pickle
import pkg_resources
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import r2_score, explained_variance_score

# 尝试导入 RandomizedLasso（旧版 sklearn）；不可用时回退
try:
    from sklearn.linear_model import RandomizedLasso  # type: ignore
    _HAS_RNDLASSO = True
except Exception:
    RandomizedLasso = None  # type: ignore
    _HAS_RNDLASSO = False

# 第三方库告警压制（仅导入时）
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import xgboost as xgb
    import statsmodels.api as sm

logger = logging.getLogger("MODEL")

# =========================
# 内联 dataIO：I/O 函数
# =========================
def read_feature(path, use_features=None):
    """读取特征表 X（TSV/HDF5/XGBoost buffer）
    - 需有 'binID' 列为索引（TSV/HDF）
    - 如果提供 use_features，则只读取/筛选这些特征
    返回：pd.DataFrame 或 xgb.DMatrix（本脚本仅在 GLM 流程中使用 DataFrame）
    """
    if path.lower().endswith(('.h5', '.hdf5')):
        X = pd.read_hdf(path, 'X')
        if use_features is not None:
            X = X.loc[:, use_features]
    elif path.lower().endswith('.buffer'):
        X = xgb.DMatrix(path)
    else:
        if use_features is not None:
            X = pd.read_csv(path, sep='\t', header=0, index_col='binID',
                            usecols=['binID'] + list(use_features))
        else:
            X = pd.read_csv(path, sep='\t', header=0, index_col='binID')

    if isinstance(X, pd.DataFrame):
        assert len(X.index.values) == len(X.index.unique()), "binID in feature table is not unique."
        na_count = X.isnull().sum()
        if na_count.sum() > 0:
            na_names = X.columns.values[np.where(na_count > 0)]
            logger.warning("NA values found in features [{}]".format(', '.join(map(str, na_names))))
            logger.warning("Fill NA with 0")
            X.fillna(0, inplace=True)
        logger.info("Successfully load {} features for {} bins".format(X.shape[1], X.shape[0]))
        return X
    else:
        logger.info("Successfully load {} features for {} bins".format(X.num_col(), X.num_row()))
        return X

def read_response(path):
    """读取响应表 y（TSV）
    需要列：binID, length, nMut, nSample, N
    """
    y = pd.read_csv(path, sep='\t', header=0, index_col='binID',
                    usecols=['binID', 'length', 'nMut', 'nSample', 'N'])
    assert len(y.index.values) == len(y.index.unique()), "binID in response table is not unique."
    return y

def read_fi(path, cutoff=0.5):
    """读取特征重要性（TSV: name, importance），按 cutoff 过滤；path 为 None 则返回 None"""
    if path is None:
        return None
    fi = pd.read_csv(path, sep='\t', header=0, index_col='name', usecols=['name', 'importance'])
    assert len(fi.index.values) == len(fi.index.unique()), "Feature name in feature importance table is not unique."
    keep = (fi.importance >= cutoff).values
    return fi.index.values[keep].tolist()

def save_fi(fi_scores, feature_names, project_name, out_dir):
    """保存特征重要性表"""
    os.makedirs(out_dir, exist_ok=True)
    res = pd.DataFrame({'name': feature_names, 'importance': fi_scores}, columns=['name', 'importance'])
    path = os.path.join(out_dir, project_name + '.feature_importance.tsv')
    logger.info(f"Save feature importance to: {path}")
    res.to_csv(path, sep='\t', index=False)
    return res

def read_param(path=None):
    """读取 xgboost 参数（与本 Binomial 版本无关，仅保留接口兼容）。"""
    path = path if path is not None else pkg_resources.resource_filename(__name__, 'xgb_param.pkl')
    with open(path, 'rb') as f:
        param = pickle.load(f)
    return param

def save_prediction(ypred, y, project_name, out_dir, model_name):
    """保存训练集预测"""
    os.makedirs(out_dir, exist_ok=True)
    y = y.copy()
    y['pred'] = ypred
    path = os.path.join(out_dir, f'{project_name}.{model_name}.model.train.pred.tsv')
    logger.info(f"Save training prediction to: {path}")
    y.to_csv(path, sep='\t')
    return

def save_model(model, project_name, out_dir, model_name):
    """保存模型 pkl（最高协议）"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{project_name}.{model_name}.model.pkl')
    logger.info(f"Save model to: {path}")
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_result(y, project_name, out_dir):
    """保存结果表（按倒数第二列排序，与原项目保持一致）"""
    os.makedirs(out_dir, exist_ok=True)
    y = y.sort_values(y.columns[-2], ascending=True)
    path = os.path.join(out_dir, f'{project_name}.result.tsv')
    logger.info(f"Save result to: {path}")
    y.to_csv(path, sep='\t')

# =========================
# BMR（GLM/GBM）
# =========================
def run_bmr(model_name: str,
            X_path: str,
            y_path: str,
            fi_cut: float = 0.5,
            fi_path: str = None,
            kfold: int = 3,
            param_path: str = None,
            project_name: str = "DriverPower1",
            out_dir: str = "./output",
            save_pred: bool = False):
    """BMR 训练入口（本脚本聚焦 Binomial；GBM 保留接口一致性）"""

    # 读取（可选）特征重要性列表
    use_features = read_fi(fi_path, fi_cut)
    run_feature_select = False if use_features else True

    # 读取特征与响应
    X_df = read_feature(X_path, use_features)  # DataFrame, index=binID
    if not isinstance(X_df, pd.DataFrame):
        logger.error("This Binomial pipeline expects features in a pandas DataFrame (TSV/HDF5).")
        sys.exit(1)
    y = read_response(y_path)                  # 需含: binID, length, nMut, nSample, N
    feature_names = X_df.columns.values.astype(str)

    # 极端失衡：下采样 nMut=0
    pct_zero = y[y.nMut == 0].shape[0] / y.shape[0] * 100.0
    pct_req = 0.1
    if pct_zero > 99:
        logger.info("{:.2f}% elements have zero mutation. Downsampling to {}%".format(pct_zero, pct_req * 100))
        ct_req = int(y.shape[0] * pct_req)
        y_nonzero = y[y.nMut > 0]
        y_zero = y[y.nMut == 0].sample(n=ct_req, replace=False)
        y = pd.concat([y_nonzero, y_zero])
        del y_nonzero, y_zero

    # 只用 length>=100 且 X∩y 的 bin
    use_bins = np.intersect1d(X_df.index.values, y.loc[y.length >= 100, :].index.values)
    logger.info("Use {} bins in model training".format(use_bins.shape[0]))
    X = X_df.loc[use_bins, :].values  # -> np.array
    y = y.loc[use_bins, :]

    if model_name in ("Binomial",):
        # Robust 缩放
        X, scaler = scale_data(X)

        # 特征选择（若未提供 FI）
        use_features_final = use_features
        if run_feature_select:
            alpha = run_lasso(X, y)
            fi_scores = run_rndlasso(X, y, alpha, feature_names)
            fi_tbl = save_fi(fi_scores, feature_names, project_name, out_dir)
            keep = (fi_tbl.importance >= fi_cut).values
            use_features_final = fi_tbl.name.values[keep].astype(str)
            if use_features_final.size == 0:
                logger.warning("No features passed fi_cut; keep ALL features for GLM.")
                use_features_final = feature_names
            else:
                X = X[:, np.isin(feature_names, use_features_final)]

        # 训练 GLM
        model = run_glm(X, y, model_name)
        # 训练集预测
        yhat = (model.fittedvalues * y.length * y.N).values

        # 训练集指标
        report_metrics(yhat, y.nMut.values)
        if save_pred:
            save_prediction(yhat, y, project_name, out_dir, model_name)

        # 分散度（Poisson 辅助回归）
        pval, theta = dispersion_test(yhat, y.nMut.values)

        # 清理缓存，减小 pkl
        if hasattr(model, "remove_data"):
            with warnings.catch_warnings():
                model.remove_data()

        # 保存模型信息
        model_info = {
            "model_name": model_name,
            "model": model,
            "scaler": scaler,
            "pval_dispersion": pval,
            "theta": theta,
            "feature_names": feature_names,                # 训练期完整列集（scaler 基于它拟合）
            "use_features": np.asarray(use_features_final),# 真正进入 GLM 的列集（顺序即训练顺序）
            "project_name": project_name,
        }

    elif model_name == "GBM":
        # GBM 分支保留（未在本需求中使用）
        offset = np.array(np.log(y.length + 1 / y.N) + np.log(y.N))
        ks = KFold(n_splits=kfold)
        param = read_param(param_path)
        yhat = np.zeros(y.shape[0])
        k = 1
        fi_scores_all = pd.DataFrame(np.nan,
                                     columns=['fold' + str(i) for i in range(1, kfold + 1)],
                                     index=feature_names)
        model = dict()
        X_dict, X_idx = dict(), dict()
        for valid, train in ks.split(range(X.shape[0])):
            logger.info('Split data fold {}/{}'.format(k, kfold))
            X_dict[k] = xgb.DMatrix(data=X[train, :],
                                    label=y.nMut.values[train],
                                    feature_names=feature_names)
            X_idx[k] = train
            X_dict[k].set_base_margin(offset[train])
            k += 1
        del X

        for k in range(1, kfold + 1):
            logger.info('Training GBM fold {}/{}'.format(k, kfold))
            k_valid = k + 1 if k < kfold else 1
            model[k] = run_gbm(X_dict[k], X_dict[k_valid], param)
            yhat[X_idx[k_valid]] = model[k].predict(X_dict[k_valid])
            fi_scores_all['fold' + str(k)] = pd.Series(model[k].get_score(importance_type='gain'))

        fi_scores_all.fillna(0, inplace=True)
        fi_scores = fi_scores_all.mean(axis=1).values
        save_fi(fi_scores, fi_scores_all.index.values, project_name, out_dir)

        report_metrics(yhat, y.nMut.values)
        if save_pred:
            save_prediction(yhat, y, project_name, out_dir, model_name)

        pval, theta = dispersion_test(yhat, y.nMut.values)
        model_info = {
            "model_name": model_name,
            "model": model,
            "pval_dispersion": pval,
            "theta": theta,
            "kfold": kfold,
            "params": param,
            "feature_names": feature_names,
            "project_name": project_name,
            "model_dir": out_dir
        }

    else:
        logger.error('Unknown background model: {}. Please use Binomial or GBM'.format(model_name))
        sys.exit(1)

    save_model(model_info, project_name, out_dir, model_name)
    logger.info('Job done!')


# =========================
# 训练用子函数
# =========================
def scale_data(X, scaler=None):
    """鲁棒缩放（对异常值不敏感）"""
    if scaler is not None:
        return scaler.transform(X), scaler
    scaler = RobustScaler(copy=False)
    scaler.fit(X)
    return scaler.transform(X), scaler

def run_lasso(X, y, max_iter=3000, cv=5, n_threads=1):
    """使用 LassoCV 选择正则化强度 alpha（至多 30 万 subsample）"""
    logger.info("Implementing LassoCV with {} iter. and {}-fold CV".format(max_iter, cv))
    y_logit = logit((y.nMut + 0.5) / (y.length * y.N))
    n = y_logit.shape[0]
    use_ix = np.random.choice(n, min(300000, n), replace=True)
    Xsub = X[use_ix, :]
    ysub = y_logit[use_ix]
    reg = LassoCV(max_iter=max_iter, cv=cv, copy_X=False, n_jobs=n_threads)
    lassocv = reg.fit(Xsub, ysub)
    logger.info("LassoCV alpha = {}".format(lassocv.alpha_))
    return lassocv.alpha_

def run_rndlasso(X, y, alpha, feature_names, n_resampling=500, sample_fraction=0.1, n_threads=1):
    """RandomizedLasso（若可用）；否则回退为 LassoCV(coef_) 绝对值"""
    y_logit = logit((y.nMut + 0.5) / (y.length * y.N))
    if _HAS_RNDLASSO:
        logger.info("Implementing RandomizedLasso: alpha=%s, n_resampling=%s, sample_fraction=%s",
                    alpha, n_resampling, sample_fraction)
        reg = RandomizedLasso(alpha=alpha,
                              n_resampling=n_resampling,
                              sample_fraction=sample_fraction,
                              selection_threshold=1e-3,
                              max_iter=3000,
                              normalize=False,
                              n_jobs=n_threads)
        rndlasso = reg.fit(X, y_logit)
        fi_scores = getattr(rndlasso, "scores_", None)
        if fi_scores is not None:
            return fi_scores
        logger.warning("RandomizedLasso returned no scores_; fallback to LassoCV(coef_).")

    logger.warning("RandomizedLasso not available; using LassoCV(coef_) abs as importance.")
    base = LassoCV(max_iter=3000, cv=5, copy_X=False)
    base.fit(X, y_logit)
    coefs = getattr(base, "coef_", np.zeros(len(feature_names)))
    return np.abs(coefs)

def run_glm(X, y, model_name):
    """训练 GLM（Binomial）"""
    # 手动加常数列
    X = np.c_[X, np.ones(X.shape[0])]
    if model_name == "Binomial":
        logger.info("Building binomial GLM")
        y_binom = np.zeros((y.shape[0], 2), dtype=np.int_)
        y_binom[:, 0] = y.nMut
        y_binom[:, 1] = y.length * y.N - y.nMut
        glm = sm.GLM(y_binom, X, family=sm.families.Binomial())
    else:
        logger.error("Only Binomial GLM is implemented in this script.")
        sys.exit(1)
    model = glm.fit()
    return model

def run_gbm(dtrain, dvalid, param):
    n_round = param.get("num_boost_round", 5000)
    early_stop = param.get("early_stopping_rounds", 5)
    verbose_eval = param.get("verbose_eval", 100)
    watchlist = [(dvalid, "eval")]
    bst = xgb.train(params=param,
                    dtrain=dtrain,
                    num_boost_round=n_round,
                    evals=watchlist,
                    early_stopping_rounds=early_stop,
                    verbose_eval=verbose_eval)
    return bst

def report_metrics(yhat, y):
    r2 = r2_score(y, yhat)
    var_exp = explained_variance_score(y, yhat)
    r = stats.pearsonr(yhat, y)[0] if len(yhat) > 1 else np.nan
    logger.info("Model metrics for training set: r2={:.2f}, Variance explained={:.2f}, Pearson'r={:.2f}"
                .format(r2, var_exp, r))

def dispersion_test(yhat, y, k=100):
    """基于回归的分散度检验（Poisson 辅助回归，重复抽样）"""
    theta = 0.0
    pval = 0.0
    for i in range(k):
        y_sub, yhat_sub = resample(y, yhat, random_state=i)
        aux = (np.power((y_sub - yhat_sub), 2) - yhat_sub) / yhat_sub
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

    parser = argparse.ArgumentParser(description="BMR training (GLM or GBM)")
    parser.add_argument("--model_name", type=str, default="Binomial",
                        help="Binomial | GBM (default: Binomial)")
    parser.add_argument("--X_path", type=str, required=True, help="Training feature table (TSV/HDF5)")
    parser.add_argument("--y_path", type=str, required=True, help="Training response table (TSV)")
    parser.add_argument("--fi_cut", type=float, default=0.5, help="Feature-importance cutoff")
    parser.add_argument("--fi_path", type=str, default=None, help="Existing FI TSV (optional)")
    parser.add_argument("--kfold", type=int, default=3, help="KFold for GBM (default: 3)")
    parser.add_argument("--param_path", type=str, default=None, help="XGBoost param pkl (GBM only)")
    parser.add_argument("--project_name", type=str, required=True, help="Project name")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_pred", action="store_true", help="Save training predictions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_bmr(model_name=args.model_name,
            X_path=args.X_path,
            y_path=args.y_path,
            fi_cut=args.fi_cut,
            fi_path=args.fi_path,
            kfold=args.kfold,
            param_path=args.param_path,
            project_name=args.project_name,
            out_dir=args.out_dir,
            save_pred=args.save_pred)
