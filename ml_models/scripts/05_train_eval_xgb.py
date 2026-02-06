#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    choose_threshold_max_bacc,
    compute_binary_metrics,
    compute_roc,
    ensure_dir,
    get_env_info,
    require_parquet_engine,
    write_json,
    write_text,
)


def load_npz(path: Path) -> Dict[str, Any]:
    """Load a .npz file and return a dict of arrays."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_compositions(datasets_dir: Path, *, split: str) -> pd.DataFrame:
    """Load composition metadata for a split (sample_id, composition_raw)."""
    base = datasets_dir / f"pn_{split}.parquet"
    if not base.exists():
        raise FileNotFoundError(f"Missing dataset parquet: {base} (run 01_extract_jsonl_to_table.py)")
    return pd.read_parquet(base)[["sample_id", "composition_raw"]]


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute negative/positive ratio for class imbalance."""
    y = np.asarray(y).astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def main() -> None:
    """Train/validate XGBoost on train_tune, then evaluate on final valid + hem_only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3408)
    ap.add_argument("--inner_split_seed", type=int, default=3408)
    ap.add_argument("--tune_frac", type=float, default=0.10)
    ap.add_argument("--early_stopping_rounds", type=int, default=100)

    ap.add_argument("--max_depth", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--learning_rate", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--subsample", type=float, nargs="+", default=[0.8, 1.0])
    ap.add_argument("--colsample_bytree", type=float, nargs="+", default=[0.8, 1.0])
    ap.add_argument("--min_child_weight", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--reg_lambda", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--reg_alpha", type=float, nargs="+", default=[0.0])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0])
    ap.add_argument("--n_estimators", type=int, default=5000)

    ap.add_argument("--features_dir", default="ml_models/artifacts/features")
    ap.add_argument("--datasets_dir", default="ml_models/artifacts/datasets")
    ap.add_argument("--out_root", default="ml_models/runs/xgb")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_parquet_engine()

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))

    try:
        import xgboost as xgb  # noqa: F401
        from xgboost import XGBClassifier
    except Exception as e:
        raise SystemExit(
            "xgboost is required for this script. Install it in your environment (e.g., conda-forge py-xgboost)."
        ) from e

    features_dir = Path(args.features_dir)
    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.out_root) / f"seed={args.seed}"
    ensure_dir(out_dir)

    run_files = {
        "config": out_dir / "config.json",
        "threshold": out_dir / "threshold.json",
        "metrics_valid": out_dir / "metrics_valid.json",
        "roc_valid": out_dir / "roc_valid.npz",
        "pred_valid": out_dir / "predictions_valid.parquet",
        "metrics_hem": out_dir / "metrics_valid_hem_only.json",
        "pred_hem": out_dir / "predictions_valid_hem_only.parquet",
        "env": out_dir / "env.txt",
    }
    for p in run_files.values():
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (use --overwrite)")

    train_npz = load_npz(features_dir / "features_v1_train.npz")
    valid_npz = load_npz(features_dir / "features_v1_valid.npz")
    hem_npz = load_npz(features_dir / "features_v1_valid_hem_only.npz")

    X_train = train_npz["X"].astype(np.float32)
    y_train = train_npz["y"].astype(int)
    X_valid = valid_npz["X"].astype(np.float32)
    y_valid = valid_npz["y"].astype(int)
    X_hem = hem_npz["X"].astype(np.float32)
    y_hem = hem_npz["y"].astype(int)

    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_train,
        y_train,
        test_size=float(args.tune_frac),
        random_state=int(args.inner_split_seed),
        stratify=y_train,
    )

    spw_fit = float(compute_scale_pos_weight(y_fit))

    results: List[Dict[str, Any]] = []
    best: Optional[Tuple[Tuple[float, float], Dict[str, Any]]] = None
    fit_sig = inspect.signature(XGBClassifier.fit)
    early_stopping_via_fit = "early_stopping_rounds" in fit_sig.parameters

    # Grid search (small) over a few CPU-friendly params.
    for md in args.max_depth:
        for lr in args.learning_rate:
            for ss in args.subsample:
                for cs in args.colsample_bytree:
                    for mcw in args.min_child_weight:
                        for rlambda in args.reg_lambda:
                            for ralpha in args.reg_alpha:
                                for gamma in args.gamma:
                                    model_kwargs: Dict[str, Any] = {
                                        "n_estimators": int(args.n_estimators),
                                        "max_depth": int(md),
                                        "learning_rate": float(lr),
                                        "subsample": float(ss),
                                        "colsample_bytree": float(cs),
                                        "min_child_weight": float(mcw),
                                        "reg_lambda": float(rlambda),
                                        "reg_alpha": float(ralpha),
                                        "gamma": float(gamma),
                                        "objective": "binary:logistic",
                                        "eval_metric": "auc",
                                        "tree_method": "hist",
                                        "random_state": int(args.seed),
                                        "n_jobs": n_jobs,
                                        "scale_pos_weight": spw_fit,
                                    }
                                    fit_kwargs: Dict[str, Any] = {"eval_set": [(X_tune, y_tune)], "verbose": False}
                                    if early_stopping_via_fit:
                                        fit_kwargs["early_stopping_rounds"] = int(args.early_stopping_rounds)
                                    else:
                                        # Newer xgboost (e.g., 3.x) wires early stopping via constructor kwargs.
                                        model_kwargs["early_stopping_rounds"] = int(args.early_stopping_rounds)

                                    model = XGBClassifier(**model_kwargs)
                                    model.fit(X_fit, y_fit, **fit_kwargs)

                                    y_score_tune = model.predict_proba(X_tune)[:, 1]
                                    thr = choose_threshold_max_bacc(y_tune, y_score_tune)
                                    metrics_tune = compute_binary_metrics(y_tune, y_score_tune, threshold=thr["threshold"])

                                    best_iteration = getattr(model, "best_iteration", None)
                                    n_estimators_best = (
                                        int(best_iteration + 1) if best_iteration is not None else int(args.n_estimators)
                                    )

                                    info = {
                                        "max_depth": int(md),
                                        "learning_rate": float(lr),
                                        "subsample": float(ss),
                                        "colsample_bytree": float(cs),
                                        "min_child_weight": float(mcw),
                                        "reg_lambda": float(rlambda),
                                        "reg_alpha": float(ralpha),
                                        "gamma": float(gamma),
                                        "scale_pos_weight": spw_fit,
                                        "early_stopping_via": "fit" if early_stopping_via_fit else "constructor",
                                        "best_n_estimators": n_estimators_best,
                                        "threshold": float(thr["threshold"]),
                                        "metrics_tune": metrics_tune,
                                    }
                                    results.append(info)

                                    key = (
                                        metrics_tune.get("balanced_accuracy", float("nan")),
                                        metrics_tune.get("roc_auc", float("nan")),
                                    )
                                    if best is None:
                                        best = (key, info)
                                    else:
                                        prev_key, _ = best
                                        if (not math.isnan(key[0]) and math.isnan(prev_key[0])) or (key[0] > prev_key[0]):
                                            best = (key, info)
                                        elif key[0] == prev_key[0] and key[1] > prev_key[1]:
                                            best = (key, info)

                                    print(
                                        f"[tune] depth={md} lr={lr:g} ss={ss:g} cs={cs:g} "
                                        f"mcw={mcw:g} l2={rlambda:g} l1={ralpha:g} gamma={gamma:g} "
                                        f"bAcc={metrics_tune['balanced_accuracy']:.4f} AUC={metrics_tune['roc_auc']:.4f} "
                                        f"thr={thr['threshold']:.6g} best_n={n_estimators_best}"
                                    )

    assert best is not None
    _, best_info = best
    best_thr = float(best_info["threshold"])
    best_n_estimators = int(best_info["best_n_estimators"])

    # Retrain on full train with the best number of estimators (from early stopping on tune).
    spw_full = float(compute_scale_pos_weight(y_train))
    final_model = XGBClassifier(
        n_estimators=best_n_estimators,
        max_depth=int(best_info["max_depth"]),
        learning_rate=float(best_info["learning_rate"]),
        subsample=float(best_info["subsample"]),
        colsample_bytree=float(best_info["colsample_bytree"]),
        min_child_weight=float(best_info["min_child_weight"]),
        reg_lambda=float(best_info["reg_lambda"]),
        reg_alpha=float(best_info["reg_alpha"]),
        gamma=float(best_info["gamma"]),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=int(args.seed),
        n_jobs=n_jobs,
        scale_pos_weight=spw_full,
    )
    final_model.fit(X_train, y_train, verbose=False)

    y_score_valid = final_model.predict_proba(X_valid)[:, 1]
    metrics_valid = compute_binary_metrics(y_valid, y_score_valid, threshold=best_thr)
    roc = compute_roc(y_valid, y_score_valid)
    if roc["thresholds"] is not None:
        np.savez_compressed(run_files["roc_valid"], fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"])
    else:
        np.savez_compressed(run_files["roc_valid"], fpr=np.array([]), tpr=np.array([]), thresholds=np.array([]))

    y_score_hem = final_model.predict_proba(X_hem)[:, 1]
    metrics_hem = compute_binary_metrics(y_hem, y_score_hem, threshold=best_thr)

    valid_comp = load_compositions(datasets_dir, split="valid")
    hem_comp = load_compositions(datasets_dir, split="valid_hem_only")

    valid_pred = pd.DataFrame(
        {
            "sample_id": valid_npz["sample_id"].astype(str),
            "y_true": y_valid.astype(int),
            "y_score": y_score_valid.astype(float),
            "y_pred": (y_score_valid >= float(best_thr)).astype(int),
        }
    ).merge(valid_comp, on="sample_id", how="left")
    valid_pred.to_parquet(run_files["pred_valid"], index=False)

    hem_pred = pd.DataFrame(
        {
            "sample_id": hem_npz["sample_id"].astype(str),
            "y_true": y_hem.astype(int),
            "y_score": y_score_hem.astype(float),
            "y_pred": (y_score_hem >= float(best_thr)).astype(int),
        }
    ).merge(hem_comp, on="sample_id", how="left")
    hem_pred.to_parquet(run_files["pred_hem"], index=False)

    env_info = get_env_info()
    write_text(run_files["env"], json.dumps(env_info, indent=2, sort_keys=True) + "\n")

    config = {
        "model": "xgb",
        "seed": int(args.seed),
        "inner_split_seed": int(args.inner_split_seed),
        "tune_frac": float(args.tune_frac),
        "grid": {
            "max_depth": [int(x) for x in args.max_depth],
            "learning_rate": [float(x) for x in args.learning_rate],
            "subsample": [float(x) for x in args.subsample],
            "colsample_bytree": [float(x) for x in args.colsample_bytree],
            "min_child_weight": [float(x) for x in args.min_child_weight],
            "reg_lambda": [float(x) for x in args.reg_lambda],
            "reg_alpha": [float(x) for x in args.reg_alpha],
            "gamma": [float(x) for x in args.gamma],
        },
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "early_stopping_via": "fit" if early_stopping_via_fit else "constructor",
        "n_estimators_search": int(args.n_estimators),
        "scale_pos_weight(train_fit)": spw_fit,
        "scale_pos_weight(train_full)": spw_full,
        "best_params": best_info,
        "best_threshold": float(best_thr),
        "features_train": str(features_dir / "features_v1_train.npz"),
        "features_valid": str(features_dir / "features_v1_valid.npz"),
        "features_valid_hem_only": str(features_dir / "features_v1_valid_hem_only.npz"),
        "datasets_dir": str(datasets_dir),
        "tune_search": results,
        "env": env_info,
    }
    write_json(run_files["config"], config)
    write_json(run_files["threshold"], {"threshold": float(best_thr), "selected_on": "train_tune", "metric": "bAcc"})
    write_json(run_files["metrics_valid"], metrics_valid)
    write_json(run_files["metrics_hem"], metrics_hem)

    print(f"[ok] wrote run dir: {out_dir}")


if __name__ == "__main__":
    main()
