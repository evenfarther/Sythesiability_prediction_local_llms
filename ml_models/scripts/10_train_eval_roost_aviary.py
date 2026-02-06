#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
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
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_compositions(datasets_dir: Path, *, split: str) -> pd.DataFrame:
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
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("[warn] --device cuda requested, but torch.cuda.is_available()==False. Falling back to cpu.")
                return "cpu"
        except Exception:
            return "cpu"
    return device


def set_all_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_aviary_imports() -> None:
    """aviary-models imports optional deps (wandb, tensorboard) unconditionally.

    In this repo we installed `aviary-models` with `--no-deps` for stability, so we
    provide minimal no-op stubs to allow importing Roost without adding extra deps.
    """
    import sys
    import types

    if "wandb" not in sys.modules:
        try:
            import wandb  # noqa: F401
        except ModuleNotFoundError:
            m = types.ModuleType("wandb")

            def _noop(*_args: Any, **_kwargs: Any) -> None:
                return None

            m.init = _noop  # type: ignore[attr-defined]
            m.finish = _noop  # type: ignore[attr-defined]
            m.log = _noop  # type: ignore[attr-defined]
            m.config = {}  # type: ignore[attr-defined]
            sys.modules["wandb"] = m

    if "torch.utils.tensorboard" not in sys.modules:
        tb = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:  # noqa: D401
            """No-op SummaryWriter stub."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def close(self) -> None:
                return None

        tb.SummaryWriter = SummaryWriter  # type: ignore[attr-defined]
        sys.modules["torch.utils.tensorboard"] = tb


_PAIR_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    cached = _PAIR_CACHE.get(n)
    if cached is not None:
        return cached
    idx = np.arange(n, dtype=np.int64)
    self_idx = np.repeat(idx, n)
    nbr_idx = np.tile(idx, n)
    _PAIR_CACHE[n] = (self_idx, nbr_idx)
    return self_idx, nbr_idx


class RoostFromFrac118Dataset:
    """Roost inputs reconstructed from features_v1's 118-dim fraction vector.

    This avoids re-parsing compositions with pymatgen during training.
    """

    def __init__(
        self,
        *,
        sample_id: np.ndarray,
        frac118: np.ndarray,
        y: np.ndarray,
        frac_eps: float = 0.0,
    ) -> None:
        if frac118.ndim != 2 or frac118.shape[1] != 118:
            raise ValueError(f"Expected frac118 shape (N,118); got {frac118.shape}")
        if len(sample_id) != frac118.shape[0] or len(y) != frac118.shape[0]:
            raise ValueError("sample_id/frac118/y length mismatch")
        self.sample_id = sample_id.astype(str)
        self.frac118 = frac118.astype(np.float32)
        self.y = y.astype(int)
        self.frac_eps = float(frac_eps)

    def __len__(self) -> int:
        return int(self.frac118.shape[0])

    def __getitem__(self, idx: int):
        import torch

        frac = self.frac118[int(idx)]
        if self.frac_eps > 0:
            nz = np.flatnonzero(frac > self.frac_eps)
        else:
            nz = np.flatnonzero(frac > 0)
        if nz.size == 0:
            raise ValueError(f"Empty composition (all-zero frac vector) at idx={idx} ({self.sample_id[idx]})")

        weights = frac[nz].astype(np.float32)
        wsum = float(weights.sum())
        if not (wsum > 0):
            raise ValueError(f"Invalid weights sum at idx={idx}: sum={wsum}")
        weights = (weights / wsum).reshape((-1, 1))

        elem_z = (nz + 1).astype(np.int64)
        self_idx, nbr_idx = _pair_indices(int(elem_z.shape[0]))

        elem_weights = torch.from_numpy(weights)  # float32, (n_elems, 1)
        elem_fea = torch.from_numpy(elem_z).long()  # (n_elems,)
        self_idx_t = torch.from_numpy(self_idx).long()  # (n_elems*n_elems,)
        nbr_idx_t = torch.from_numpy(nbr_idx).long()

        y = int(self.y[int(idx)])
        target = torch.tensor([y], dtype=torch.long)

        sid = str(self.sample_id[int(idx)])
        # Roost's collate expects exactly 2 identifiers.
        return (elem_weights, elem_fea, self_idx_t, nbr_idx_t), [target], sid, sid


@dataclass(frozen=True)
class TrainBest:
    best_epoch: int
    best_threshold: float
    best_metrics_tune: Dict[str, Any]
    model_state: Dict[str, Any]
    history: List[Dict[str, Any]]


def _compute_class_weights(y_fit: np.ndarray) -> Optional[np.ndarray]:
    y_fit = np.asarray(y_fit).astype(int)
    n_pos = int(np.sum(y_fit == 1))
    n_neg = int(np.sum(y_fit == 0))
    n_total = int(y_fit.shape[0])
    if n_pos == 0 or n_neg == 0:
        return None
    # sklearn-style balanced weights: n_total / (n_classes * count_c)
    w0 = n_total / (2.0 * n_neg)
    w1 = n_total / (2.0 * n_pos)
    return np.array([w0, w1], dtype=np.float32)


def _to_device(inputs: Tuple[Any, ...], device: str):
    return tuple(x.to(device) for x in inputs)


def predict_scores(model: Any, loader: Any, *, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ys: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    ids: List[str] = []

    with torch.no_grad():
        for (inputs, targets, material_ids, _dup) in loader:
            (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx) = inputs
            (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx) = _to_device(
                (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx), device
            )

            logits = model(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)[0]
            prob_pos = torch.softmax(logits, dim=1)[:, 1]

            y = targets[0].squeeze(1).cpu().numpy().astype(int)
            ys.append(y)
            scores.append(prob_pos.detach().cpu().numpy().astype(np.float32))
            ids.extend([str(x) for x in material_ids])

    return np.concatenate(ys), np.concatenate(scores), np.asarray(ids, dtype=str)


def train_roost_one_seed(
    *,
    frac_fit: np.ndarray,
    y_fit: np.ndarray,
    frac_tune: np.ndarray,
    y_tune: np.ndarray,
    sample_id_fit: np.ndarray,
    sample_id_tune: np.ndarray,
    seed: int,
    device: str,
    elem_embedding: str,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    patience: Optional[int],
    lr: float,
    weight_decay: float,
    num_workers: int,
    frac_eps: float,
) -> TrainBest:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    _ensure_aviary_imports()
    from aviary.roost.data import collate_batch  # noqa: WPS433
    from aviary.roost.model import Roost  # noqa: WPS433

    set_all_seeds(int(seed))

    ds_fit = RoostFromFrac118Dataset(sample_id=sample_id_fit, frac118=frac_fit, y=y_fit, frac_eps=frac_eps)
    ds_tune = RoostFromFrac118Dataset(sample_id=sample_id_tune, frac118=frac_tune, y=y_tune, frac_eps=frac_eps)

    pin = device == "cuda"
    fit_loader = DataLoader(
        ds_fit,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    tune_loader = DataLoader(
        ds_tune,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin,
        collate_fn=collate_batch,
    )

    task_dict = {"pn": "classification"}
    model = Roost(
        task_dict=task_dict,
        robust=False,
        n_targets=[2],
        elem_embedding=str(elem_embedding),
        device=str(device),
    )
    model = model.to(device)

    class_weights = _compute_class_weights(y_fit)
    if class_weights is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best: Optional[TrainBest] = None
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for (inputs, targets, _material_ids, _dup) in fit_loader:
            (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx) = inputs
            (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx) = _to_device(
                (elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx), device
            )
            yb = targets[0].squeeze(1).to(device)

            logits = model(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)[0]
            loss = loss_fn(logits, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = int(yb.shape[0])
            total_seen += bs
            total_loss += float(loss.detach().cpu().item()) * bs

        mean_loss = total_loss / max(1, total_seen)

        y_true_t, y_score_t, _ids_t = predict_scores(model, tune_loader, device=device)
        thr = choose_threshold_max_bacc(y_true_t, y_score_t)
        metrics_tune = compute_binary_metrics(y_true_t, y_score_t, threshold=float(thr["threshold"]))

        row = {
            "epoch": int(epoch),
            "train_loss": float(mean_loss),
            "tune_threshold": float(thr["threshold"]),
            "tune_balanced_accuracy": float(metrics_tune.get("balanced_accuracy", float("nan"))),
            "tune_roc_auc": float(metrics_tune.get("roc_auc", float("nan"))),
            "tune_tpr": float(metrics_tune.get("tpr_recall", float("nan"))),
            "tune_tnr": float(metrics_tune.get("tnr_specificity", float("nan"))),
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] loss={mean_loss:.6g} "
            f"tune_bAcc={row['tune_balanced_accuracy']:.4f} "
            f"tune_AUC={row['tune_roc_auc']:.4f} "
            f"thr={row['tune_threshold']:.6g}"
        )

        score = float(row["tune_balanced_accuracy"])
        if best is None or (not np.isnan(score) and score > float(best.best_metrics_tune.get("balanced_accuracy", -1e9))):
            best = TrainBest(
                best_epoch=int(epoch),
                best_threshold=float(thr["threshold"]),
                best_metrics_tune=metrics_tune,
                model_state=copy.deepcopy(model.state_dict()),
                history=copy.deepcopy(history),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= int(patience):
                print(f"[early-stop] no improvement for {patience} epochs (best_epoch={best.best_epoch})")
                break

    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3408)
    ap.add_argument("--inner_split_seed", type=int, default=3408)
    ap.add_argument("--tune_frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--elem_embedding", default="matscholar200")
    ap.add_argument("--frac_eps", type=float, default=0.0)

    ap.add_argument("--features_dir", default="ml_models/artifacts/features")
    ap.add_argument("--datasets_dir", default="ml_models/artifacts/datasets")
    ap.add_argument("--out_root", default="ml_models/runs/roost_aviary")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Run on a tiny subset for quick validation.")
    args = ap.parse_args()

    require_parquet_engine()

    device = pick_device(str(args.device))
    ensure_dir(Path(args.out_root))

    out_dir = Path(args.out_root) / f"seed={int(args.seed)}"
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
        "model": out_dir / "model.pt",
        "history": out_dir / "training_history.json",
    }
    for p in run_files.values():
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (use --overwrite)")

    features_dir = Path(args.features_dir)
    datasets_dir = Path(args.datasets_dir)

    train_npz = load_npz(features_dir / "features_v1_train.npz")
    valid_npz = load_npz(features_dir / "features_v1_valid.npz")
    hem_npz = load_npz(features_dir / "features_v1_valid_hem_only.npz")

    X_train = train_npz["X"].astype(np.float32)
    y_train = train_npz["y"].astype(int)
    sid_train = train_npz["sample_id"].astype(str)

    X_valid = valid_npz["X"].astype(np.float32)
    y_valid = valid_npz["y"].astype(int)
    sid_valid = valid_npz["sample_id"].astype(str)

    X_hem = hem_npz["X"].astype(np.float32)
    y_hem = hem_npz["y"].astype(int)
    sid_hem = hem_npz["sample_id"].astype(str)

    if args.smoke:
        # Keep class balance by slicing both classes if possible.
        max_n = 2000
        pos_idx = np.where(y_train == 1)[0][: max_n // 2]
        neg_idx = np.where(y_train == 0)[0][: max_n // 2]
        keep = np.concatenate([pos_idx, neg_idx])
        X_train, y_train, sid_train = X_train[keep], y_train[keep], sid_train[keep]
        X_valid, y_valid, sid_valid = X_valid[:1000], y_valid[:1000], sid_valid[:1000]
        X_hem, y_hem, sid_hem = X_hem[:262], y_hem[:262], sid_hem[:262]
        print(f"[smoke] train={len(y_train)} valid={len(y_valid)} hem_only={len(y_hem)}")

    frac_train = X_train[:, :118].astype(np.float32)
    frac_valid = X_valid[:, :118].astype(np.float32)
    frac_hem = X_hem[:, :118].astype(np.float32)

    # Internal tune split (train only). valid is final test and is never used here.
    sid_fit, sid_tune, frac_fit, frac_tune, y_fit, y_tune = train_test_split(
        sid_train,
        frac_train,
        y_train,
        test_size=float(args.tune_frac),
        random_state=int(args.inner_split_seed),
        stratify=y_train,
    )

    best = train_roost_one_seed(
        frac_fit=frac_fit,
        y_fit=y_fit,
        frac_tune=frac_tune,
        y_tune=y_tune,
        sample_id_fit=sid_fit,
        sample_id_tune=sid_tune,
        seed=int(args.seed),
        device=str(device),
        elem_embedding=str(args.elem_embedding),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience) if args.patience > 0 else None,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        frac_eps=float(args.frac_eps),
    )

    # Rebuild model and load best state for evaluation.
    import torch

    _ensure_aviary_imports()
    from aviary.roost.data import collate_batch  # noqa: WPS433
    from aviary.roost.model import Roost  # noqa: WPS433
    from torch.utils.data import DataLoader

    task_dict = {"pn": "classification"}
    model = Roost(task_dict=task_dict, robust=False, n_targets=[2], elem_embedding=str(args.elem_embedding), device=str(device))
    model.load_state_dict(best.model_state)
    model = model.to(device).eval()

    pin = device == "cuda"
    ds_valid = RoostFromFrac118Dataset(sample_id=sid_valid, frac118=frac_valid, y=y_valid, frac_eps=float(args.frac_eps))
    ds_hem = RoostFromFrac118Dataset(sample_id=sid_hem, frac118=frac_hem, y=y_hem, frac_eps=float(args.frac_eps))
    valid_loader = DataLoader(
        ds_valid,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    hem_loader = DataLoader(
        ds_hem,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin,
        collate_fn=collate_batch,
    )

    best_thr = float(best.best_threshold)
    y_true_v, y_score_v, ids_v = predict_scores(model, valid_loader, device=str(device))
    metrics_valid = compute_binary_metrics(y_true_v, y_score_v, threshold=best_thr)
    roc = compute_roc(y_true_v, y_score_v)
    if roc["thresholds"] is not None:
        np.savez_compressed(run_files["roc_valid"], fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"])
    else:
        np.savez_compressed(run_files["roc_valid"], fpr=np.array([]), tpr=np.array([]), thresholds=np.array([]))

    y_true_h, y_score_h, ids_h = predict_scores(model, hem_loader, device=str(device))
    metrics_hem = compute_binary_metrics(y_true_h, y_score_h, threshold=best_thr)

    # Save predictions with composition.
    valid_comp = load_compositions(datasets_dir, split="valid")
    hem_comp = load_compositions(datasets_dir, split="valid_hem_only")

    valid_pred = pd.DataFrame(
        {
            "sample_id": ids_v.astype(str),
            "y_true": y_true_v.astype(int),
            "y_score": y_score_v.astype(float),
            "y_pred": (y_score_v >= best_thr).astype(int),
        }
    ).merge(valid_comp, on="sample_id", how="left")
    valid_pred.to_parquet(run_files["pred_valid"], index=False)

    hem_pred = pd.DataFrame(
        {
            "sample_id": ids_h.astype(str),
            "y_true": y_true_h.astype(int),
            "y_score": y_score_h.astype(float),
            "y_pred": (y_score_h >= best_thr).astype(int),
        }
    ).merge(hem_comp, on="sample_id", how="left")
    hem_pred.to_parquet(run_files["pred_hem"], index=False)

    env_info = get_env_info()
    env_info["aviary"] = "1.2.1"
    write_text(run_files["env"], json.dumps(env_info, indent=2, sort_keys=True) + "\n")

    config = {
        "model": "roost_aviary",
        "seed": int(args.seed),
        "inner_split_seed": int(args.inner_split_seed),
        "tune_frac": float(args.tune_frac),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "elem_embedding": str(args.elem_embedding),
        "frac_eps": float(args.frac_eps),
        "best_epoch": int(best.best_epoch),
        "best_threshold": float(best_thr),
        "tune_metrics_at_best": best.best_metrics_tune,
        "features_train": str(features_dir / "features_v1_train.npz"),
        "features_valid": str(features_dir / "features_v1_valid.npz"),
        "features_valid_hem_only": str(features_dir / "features_v1_valid_hem_only.npz"),
        "datasets_dir": str(datasets_dir),
        "env": env_info,
    }
    write_json(run_files["config"], config)
    write_json(run_files["threshold"], {"threshold": float(best_thr), "selected_on": "train_tune", "metric": "bAcc"})
    write_json(run_files["metrics_valid"], metrics_valid)
    write_json(run_files["metrics_hem"], metrics_hem)

    write_json(run_files["history"], {"history": best.history})
    torch.save({"state_dict": best.model_state, "config": config}, run_files["model"])

    print(f"[ok] wrote run dir: {out_dir}")


if __name__ == "__main__":
    main()

