#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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


def main() -> None:
    """Train/validate LinearSVC on train_tune, then evaluate on final valid + hem_only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3408)
    ap.add_argument("--inner_split_seed", type=int, default=3408)
    ap.add_argument("--tune_frac", type=float, default=0.10)
    ap.add_argument(
        "--c_values",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        help="LinearSVC C grid (log-ish spacing).",
    )
    ap.add_argument("--max_iter", type=int, default=10000)

    ap.add_argument("--features_dir", default="ml_models/artifacts/features")
    ap.add_argument("--datasets_dir", default="ml_models/artifacts/datasets")
    ap.add_argument("--out_root", default="ml_models/runs/svm_linearsvc")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_parquet_engine()

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

    # Internal tune split (train only).
    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_train,
        y_train,
        test_size=float(args.tune_frac),
        random_state=int(args.inner_split_seed),
        stratify=y_train,
    )

    results: List[Dict[str, Any]] = []
    best = None
    for C in args.c_values:
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=float(C),
                class_weight="balanced",
                max_iter=int(args.max_iter),
                dual=False,
                random_state=int(args.seed),
            ),
        )
        model.fit(X_fit, y_fit)
        y_score_tune = model.decision_function(X_tune)
        thr = choose_threshold_max_bacc(y_tune, y_score_tune)
        metrics_tune = compute_binary_metrics(y_tune, y_score_tune, threshold=thr["threshold"])
        results.append({"C": float(C), "threshold": float(thr["threshold"]), "metrics_tune": metrics_tune})

        key = (metrics_tune.get("balanced_accuracy", float("nan")), metrics_tune.get("roc_auc", float("nan")))
        if best is None:
            best = (key, float(C), float(thr["threshold"]))
        else:
            # Compare bAcc first, then ROC-AUC.
            prev_key, _, _ = best
            if (not math.isnan(key[0]) and math.isnan(prev_key[0])) or (key[0] > prev_key[0]):
                best = (key, float(C), float(thr["threshold"]))
            elif key[0] == prev_key[0] and key[1] > prev_key[1]:
                best = (key, float(C), float(thr["threshold"]))

        print(
            f"[tune] C={C:g} bAcc={metrics_tune['balanced_accuracy']:.4f} "
            f"AUC={metrics_tune['roc_auc']:.4f} thr={thr['threshold']:.6g}"
        )

    assert best is not None
    (_, best_c, best_thr) = best

    # Retrain final model on full train.
    final_model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=float(best_c),
            class_weight="balanced",
            max_iter=int(args.max_iter),
            dual=False,
            random_state=int(args.seed),
        ),
    )
    final_model.fit(X_train, y_train)

    # Evaluate on final test(valid).
    y_score_valid = final_model.decision_function(X_valid)
    metrics_valid = compute_binary_metrics(y_valid, y_score_valid, threshold=float(best_thr))
    roc = compute_roc(y_valid, y_score_valid)
    if roc["thresholds"] is not None:
        np.savez_compressed(run_files["roc_valid"], fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"])
    else:
        np.savez_compressed(run_files["roc_valid"], fpr=np.array([]), tpr=np.array([]), thresholds=np.array([]))

    # Evaluate HEM-only (recall only is meaningful if all-P).
    y_score_hem = final_model.decision_function(X_hem)
    metrics_hem = compute_binary_metrics(y_hem, y_score_hem, threshold=float(best_thr))

    # Save predictions with composition.
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
        "model": "svm_linearsvc",
        "seed": int(args.seed),
        "inner_split_seed": int(args.inner_split_seed),
        "tune_frac": float(args.tune_frac),
        "c_values": [float(x) for x in args.c_values],
        "max_iter": int(args.max_iter),
        "best_c": float(best_c),
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
