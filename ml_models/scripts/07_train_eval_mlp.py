#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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


def pick_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def set_all_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if int(np.sum(y_true == 1)) == 0 or int(np.sum(y_true == 0)) == 0:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class TrainResult:
    lr: float
    best_epoch: int
    best_auc: float
    threshold: float
    metrics_tune: Dict[str, Any]
    model_state: Dict[str, Any]


def build_mlp(in_dim: int, hidden_sizes: List[int], *, dropout: float) -> Any:
    import torch
    from torch import nn

    layers: List[nn.Module] = []
    prev = int(in_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.ReLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(h)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def count_parameters(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _make_loaders(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    batch_size: int,
    eval_batch_size: int,
    device: str,
    num_workers: int,
) -> Tuple[Any, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pin = device == "cuda"
    fit_ds = TensorDataset(torch.from_numpy(X_fit).float(), torch.from_numpy(y_fit).long())
    tune_ds = TensorDataset(torch.from_numpy(X_tune).float(), torch.from_numpy(y_tune).long())

    train_loader = DataLoader(
        fit_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
    )
    tune_loader = DataLoader(
        tune_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
    )
    return train_loader, tune_loader


def _safe_std(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    return std


def train_one_lr(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    in_dim: int,
    hidden_sizes: List[int],
    dropout: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    eval_batch_size: int,
    device: str,
    seed: int,
    num_workers: int,
) -> TrainResult:
    import torch
    from torch import nn

    set_all_seeds(seed)

    train_loader, tune_loader = _make_loaders(
        X_fit,
        y_fit,
        X_tune,
        y_tune,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        device=device,
        num_workers=num_workers,
    )

    mean = torch.from_numpy(X_fit.mean(axis=0).astype(np.float32)).to(device)
    std = torch.from_numpy(_safe_std(X_fit)).to(device)

    n_pos = int(np.sum(y_fit == 1))
    n_neg = int(np.sum(y_fit == 0))
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    model = build_mlp(in_dim, hidden_sizes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    best_state: Optional[Dict[str, Any]] = None
    best_auc: float = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            xb = (xb - mean) / std
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Evaluate on tune for early stopping.
        model.eval()
        y_true_list: List[np.ndarray] = []
        y_score_list: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in tune_loader:
                xb = xb.to(device)
                xb = (xb - mean) / std
                logits = model(xb).squeeze(-1)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                y_true_list.append(yb.numpy().astype(int))
                y_score_list.append(prob.astype(float))
        y_true_tune = np.concatenate(y_true_list)
        y_score_tune = np.concatenate(y_score_list)
        auc = safe_auc(y_true_tune, y_score_tune)

        if not math.isnan(auc) and auc > best_auc + 1e-6:
            best_auc = float(auc)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[epoch {epoch:03d}] lr={lr:g} tune_auc={auc:.4f} best_auc={best_auc:.4f} no_improve={epochs_no_improve}")
        if epochs_no_improve >= int(patience):
            break

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = int(max_epochs)
        best_auc = float("nan")

    # Tune-time threshold selection for bAcc.
    model.load_state_dict(best_state)
    model.eval()
    y_true_list = []
    y_score_list = []
    with torch.no_grad():
        for xb, yb in tune_loader:
            xb = xb.to(device)
            xb = (xb - mean) / std
            logits = model(xb).squeeze(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y_true_list.append(yb.numpy().astype(int))
            y_score_list.append(prob.astype(float))
    y_true_tune = np.concatenate(y_true_list)
    y_score_tune = np.concatenate(y_score_list)
    thr = choose_threshold_max_bacc(y_true_tune, y_score_tune)
    metrics_tune = compute_binary_metrics(y_true_tune, y_score_tune, threshold=float(thr["threshold"]))

    return TrainResult(
        lr=float(lr),
        best_epoch=int(best_epoch),
        best_auc=float(best_auc),
        threshold=float(thr["threshold"]),
        metrics_tune=metrics_tune,
        model_state=best_state,
    )


def retrain_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    in_dim: int,
    hidden_sizes: List[int],
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    seed: int,
    num_workers: int,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_all_seeds(seed)

    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    std_t = torch.from_numpy(std.astype(np.float32)).to(device)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    model = build_mlp(in_dim, hidden_sizes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    pin = device == "cuda"
    ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            xb = (xb - mean_t) / std_t
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"[retrain] epoch {epoch:03d}/{epochs}")

    return model


def score_dataset(
    model: Any,
    X: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    eval_batch_size: int,
    num_workers: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    pin = device == "cuda"
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(
        ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
    )

    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    std_t = torch.from_numpy(std.astype(np.float32)).to(device)

    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            xb = (xb - mean_t) / std_t
            logits = model(xb).squeeze(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            scores.append(prob)
    return np.concatenate(scores)


def main() -> None:
    """MLP baseline (composition descriptors) with 3 capacity points: small/medium/large."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3408)
    ap.add_argument("--inner_split_seed", type=int, default=3408)
    ap.add_argument("--tune_frac", type=float, default=0.10)

    ap.add_argument("--arch", choices=["small", "medium", "large", "custom"], default="small")
    ap.add_argument("--hidden_sizes", type=int, nargs="+", default=None, help="Only used when --arch=custom.")
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr_grid", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=8192)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    ap.add_argument("--features_dir", default="ml_models/artifacts/features")
    ap.add_argument("--datasets_dir", default="ml_models/artifacts/datasets")
    ap.add_argument("--feature_set", default="v1", help="Uses features_<set>_{train,valid,valid_hem_only}.npz")
    ap.add_argument("--out_root", default=None, help="Default: ml_models/runs/mlp_<arch>")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_retrain_full", action="store_true")
    args = ap.parse_args()

    require_parquet_engine()

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit("torch is required for MLP baseline. Install torch in your environment.") from e

    device = pick_device(str(args.device))
    features_dir = Path(args.features_dir)
    datasets_dir = Path(args.datasets_dir)
    feature_set = str(args.feature_set).strip()
    out_root = str(args.out_root) if args.out_root is not None else f"ml_models/runs/mlp_{args.arch}"
    out_dir = Path(out_root) / f"seed={args.seed}"
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

    train_npz = load_npz(features_dir / f"features_{feature_set}_train.npz")
    valid_npz = load_npz(features_dir / f"features_{feature_set}_valid.npz")
    hem_npz = load_npz(features_dir / f"features_{feature_set}_valid_hem_only.npz")

    X_train = train_npz["X"].astype(np.float32)
    y_train = train_npz["y"].astype(int)
    X_valid = valid_npz["X"].astype(np.float32)
    y_valid = valid_npz["y"].astype(int)
    X_hem = hem_npz["X"].astype(np.float32)
    y_hem = hem_npz["y"].astype(int)

    in_dim = int(X_train.shape[1])

    if args.arch == "small":
        hidden_sizes = [512, 256, 128]
    elif args.arch == "medium":
        hidden_sizes = [1024, 1024, 1024]
    elif args.arch == "large":
        hidden_sizes = [4096, 4096, 4096]
    else:
        if not args.hidden_sizes:
            raise SystemExit("--arch=custom requires --hidden_sizes")
        hidden_sizes = [int(x) for x in args.hidden_sizes]

    # Internal tune split (train only).
    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_train,
        y_train,
        test_size=float(args.tune_frac),
        random_state=int(args.inner_split_seed),
        stratify=y_train,
    )

    mean = X_fit.mean(axis=0).astype(np.float32)
    std = _safe_std(X_fit)

    # Pick LR by tune bAcc (primary) then tune AUC.
    best: Optional[Tuple[Tuple[float, float], TrainResult]] = None
    tune_search: List[Dict[str, Any]] = []
    for lr in args.lr_grid:
        res = train_one_lr(
            X_fit,
            y_fit,
            X_tune,
            y_tune,
            in_dim=in_dim,
            hidden_sizes=hidden_sizes,
            dropout=float(args.dropout),
            lr=float(lr),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            eval_batch_size=int(args.eval_batch_size),
            device=device,
            seed=int(args.seed),
            num_workers=int(args.num_workers),
        )
        tune_search.append(
            {
                "lr": res.lr,
                "best_epoch": res.best_epoch,
                "best_auc": res.best_auc,
                "threshold": res.threshold,
                "metrics_tune": res.metrics_tune,
            }
        )

        key = (res.metrics_tune.get("balanced_accuracy", float("nan")), res.metrics_tune.get("roc_auc", float("nan")))
        if best is None:
            best = (key, res)
        else:
            prev_key, _ = best
            if (not math.isnan(key[0]) and math.isnan(prev_key[0])) or (key[0] > prev_key[0]):
                best = (key, res)
            elif key[0] == prev_key[0] and key[1] > prev_key[1]:
                best = (key, res)

        print(
            f"[tune] lr={res.lr:g} best_epoch={res.best_epoch} "
            f"bAcc={res.metrics_tune['balanced_accuracy']:.4f} AUC={res.metrics_tune['roc_auc']:.4f} "
            f"thr={res.threshold:.6g}"
        )

    assert best is not None
    _, best_res = best

    # Train final model: either reuse fit-trained state, or retrain on full train for best_epoch.
    if args.no_retrain_full:
        import torch

        model = build_mlp(in_dim, hidden_sizes, dropout=float(args.dropout))
        model.load_state_dict(best_res.model_state)
        model = model.to(device)
    else:
        model = retrain_full(
            X_train,
            y_train,
            in_dim=in_dim,
            hidden_sizes=hidden_sizes,
            dropout=float(args.dropout),
            lr=float(best_res.lr),
            epochs=int(best_res.best_epoch),
            batch_size=int(args.batch_size),
            mean=mean,
            std=std,
            device=device,
            seed=int(args.seed),
            num_workers=int(args.num_workers),
        )

    y_score_valid = score_dataset(
        model,
        X_valid,
        mean=mean,
        std=std,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    metrics_valid = compute_binary_metrics(y_valid, y_score_valid, threshold=float(best_res.threshold))
    roc = compute_roc(y_valid, y_score_valid)
    if roc["thresholds"] is not None:
        np.savez_compressed(run_files["roc_valid"], fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"])
    else:
        np.savez_compressed(run_files["roc_valid"], fpr=np.array([]), tpr=np.array([]), thresholds=np.array([]))

    y_score_hem = score_dataset(
        model,
        X_hem,
        mean=mean,
        std=std,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    metrics_hem = compute_binary_metrics(y_hem, y_score_hem, threshold=float(best_res.threshold))

    valid_comp = load_compositions(datasets_dir, split="valid")
    hem_comp = load_compositions(datasets_dir, split="valid_hem_only")

    valid_pred = pd.DataFrame(
        {
            "sample_id": valid_npz["sample_id"].astype(str),
            "y_true": y_valid.astype(int),
            "y_score": y_score_valid.astype(float),
            "y_pred": (y_score_valid >= float(best_res.threshold)).astype(int),
        }
    ).merge(valid_comp, on="sample_id", how="left")
    valid_pred.to_parquet(run_files["pred_valid"], index=False)

    hem_pred = pd.DataFrame(
        {
            "sample_id": hem_npz["sample_id"].astype(str),
            "y_true": y_hem.astype(int),
            "y_score": y_score_hem.astype(float),
            "y_pred": (y_score_hem >= float(best_res.threshold)).astype(int),
        }
    ).merge(hem_comp, on="sample_id", how="left")
    hem_pred.to_parquet(run_files["pred_hem"], index=False)

    env_info = get_env_info()
    write_text(run_files["env"], json.dumps(env_info, indent=2, sort_keys=True) + "\n")

    config = {
        "model": f"mlp_{args.arch}",
        "seed": int(args.seed),
        "inner_split_seed": int(args.inner_split_seed),
        "tune_frac": float(args.tune_frac),
        "arch": str(args.arch),
        "hidden_sizes": [int(x) for x in hidden_sizes],
        "dropout": float(args.dropout),
        "parameter_count": int(count_parameters(model)),
        "device": device,
        "lr_grid": [float(x) for x in args.lr_grid],
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "feature_set": feature_set,
        "features_train": str(features_dir / f"features_{feature_set}_train.npz"),
        "features_valid": str(features_dir / f"features_{feature_set}_valid.npz"),
        "features_valid_hem_only": str(features_dir / f"features_{feature_set}_valid_hem_only.npz"),
        "datasets_dir": str(datasets_dir),
        "tune_search": tune_search,
        "best_lr": float(best_res.lr),
        "best_epoch": int(best_res.best_epoch),
        "best_threshold": float(best_res.threshold),
        "retrain_full": bool(not args.no_retrain_full),
        "env": env_info,
    }
    write_json(run_files["config"], config)
    write_json(run_files["threshold"], {"threshold": float(best_res.threshold), "selected_on": "train_tune", "metric": "bAcc"})
    write_json(run_files["metrics_valid"], metrics_valid)
    write_json(run_files["metrics_hem"], metrics_hem)

    print(f"[ok] wrote run dir: {out_dir}")


if __name__ == "__main__":
    main()
