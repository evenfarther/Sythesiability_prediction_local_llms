#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    # Note: upstream JSONs may contain NaN (python json allows it).
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def split_model_name(model_dir_name: str) -> Tuple[str, str]:
    """Map run dir name -> (method, variant)."""
    if model_dir_name == "xgb":
        return ("xgb", "default")
    if model_dir_name == "roost_aviary":
        return ("roost", "aviary")
    if model_dir_name.startswith("mlp_"):
        return ("mlp", model_dir_name.split("_", 1)[-1])
    if model_dir_name.startswith("svm_"):
        return ("svm", model_dir_name.split("_", 1)[-1])
    return ("other", model_dir_name)


def extract_summary_metrics(seed_summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Flatten mean/std metrics for CSV."""
    out: Dict[str, Optional[float]] = {}

    summary = seed_summary.get("summary", {}) or {}
    valid = summary.get("valid", {}) or {}
    hem_only = summary.get("hem_only", {}) or {}
    threshold = summary.get("threshold", {}) or {}

    def add(prefix: str, stats: Dict[str, Any]) -> None:
        out[f"{prefix}_mean"] = as_float(stats.get("mean"))
        out[f"{prefix}_std"] = as_float(stats.get("std"))

    add("valid_balanced_accuracy", valid.get("balanced_accuracy", {}) or {})
    add("valid_roc_auc", valid.get("roc_auc", {}) or {})
    add("valid_mcc", valid.get("mcc", {}) or {})
    add("valid_tpr_recall", valid.get("tpr_recall", {}) or {})
    add("valid_tnr_specificity", valid.get("tnr_specificity", {}) or {})
    add("valid_precision", valid.get("precision", {}) or {})

    add("hem_only_recall", hem_only.get("tpr_recall", {}) or {})
    add("threshold", threshold)

    return out


def verify_run_dir(model_dir: Path) -> List[str]:
    """Return a list of problems found; empty means OK."""
    problems: List[str] = []

    seed_summary_path = model_dir / "seed_summary.json"
    if not seed_summary_path.exists():
        problems.append(f"missing seed_summary.json: {seed_summary_path}")
        return problems

    try:
        seed_summary = read_json(seed_summary_path)
    except Exception as e:
        problems.append(f"failed to parse seed_summary.json: {seed_summary_path} ({e})")
        return problems

    seeds = seed_summary.get("seeds") or []
    n_seeds = seed_summary.get("n_seeds")
    if n_seeds is not None and isinstance(seeds, list) and len(seeds) != int(n_seeds):
        problems.append(f"n_seeds mismatch: seed_summary.json says {n_seeds}, seeds list has {len(seeds)}")

    required = ["config.json", "threshold.json", "metrics_valid.json", "metrics_valid_hem_only.json"]
    seed_dirs = sorted([p for p in model_dir.glob("seed=*") if p.is_dir()])
    if not seed_dirs:
        problems.append(f"no seed dirs under: {model_dir}")
        return problems

    for sd in seed_dirs:
        missing = [f for f in required if not (sd / f).exists()]
        if missing:
            problems.append(f"{sd}: missing {missing}")

    return problems


def write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="ml_models/runs")
    ap.add_argument("--out_json", default="ml_models/runs/total_summary.json")
    ap.add_argument("--out_csv", default="ml_models/runs/total_summary.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs_dir: {runs_dir}")

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    for p in (out_json, out_csv):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (use --overwrite)")

    model_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and (p / "seed_summary.json").exists()])
    if not model_dirs:
        raise SystemExit(f"No model run dirs with seed_summary.json found under: {runs_dir}")

    all_problems: Dict[str, List[str]] = {}
    models: Dict[str, Any] = {}

    for model_dir in model_dirs:
        model_name = model_dir.name

        problems: List[str] = []
        if not args.no_verify:
            problems = verify_run_dir(model_dir)
            if problems:
                all_problems[model_name] = problems

        seed_summary_path = model_dir / "seed_summary.json"
        seed_summary = read_json(seed_summary_path)

        method, variant = split_model_name(model_name)
        models[model_name] = {
            "method": method,
            "variant": variant,
            "seed_summary_path": str(seed_summary_path),
            "n_seeds": int(seed_summary.get("n_seeds")),
            "seeds": seed_summary.get("seeds"),
            "summary": seed_summary.get("summary"),
            "seed_runs": seed_summary.get("seed_runs"),
            "created_utc": seed_summary.get("created_utc"),
        }

    if all_problems:
        msg = ["Run verification failed; refusing to write total summary:"]
        for model_name, problems in sorted(all_problems.items()):
            msg.append(f"- {model_name}:")
            for p in problems:
                msg.append(f"  - {p}")
        raise SystemExit("\n".join(msg))

    out_obj = {
        "created_utc": utc_now_iso(),
        "runs_dir": str(runs_dir),
        "models": models,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # CSV: 1 row per model with mean/std metrics
    rows: List[Dict[str, Any]] = []
    max_seed_cols = 0
    for model_name in models.keys():
        seeds = models[model_name].get("seeds") or []
        if isinstance(seeds, list):
            max_seed_cols = max(max_seed_cols, len(seeds))

    for model_name in sorted(models.keys()):
        seed_summary = read_json(Path(models[model_name]["seed_summary_path"]))
        method, variant = split_model_name(model_name)
        seeds_list = seed_summary.get("seeds") or []
        # Use a non-numeric delimiter to avoid spreadsheet programs auto-casting to a large integer.
        seeds_str = "|".join(str(s) for s in seeds_list)
        row: Dict[str, Any] = {
            "model": model_name,
            "method": method,
            "variant": variant,
            "n_seeds": int(seed_summary.get("n_seeds")),
            "seeds": seeds_str,
            "best_c_counts": json.dumps((seed_summary.get("summary") or {}).get("best_c_counts") or {}, sort_keys=True),
        }
        # Also write explicit seed columns for spreadsheet friendliness.
        for i in range(1, max_seed_cols + 1):
            key = f"seed_{i}"
            row[key] = int(seeds_list[i - 1]) if i <= len(seeds_list) else None
        row.update(extract_summary_metrics(seed_summary))
        rows.append(row)

    fieldnames = [
        "model",
        "method",
        "variant",
        "n_seeds",
        "seeds",
        *[f"seed_{i}" for i in range(1, max_seed_cols + 1)],
        "valid_balanced_accuracy_mean",
        "valid_balanced_accuracy_std",
        "valid_roc_auc_mean",
        "valid_roc_auc_std",
        "valid_mcc_mean",
        "valid_mcc_std",
        "valid_tpr_recall_mean",
        "valid_tpr_recall_std",
        "valid_tnr_specificity_mean",
        "valid_tnr_specificity_std",
        "valid_precision_mean",
        "valid_precision_std",
        "hem_only_recall_mean",
        "hem_only_recall_std",
        "threshold_mean",
        "threshold_std",
        "best_c_counts",
    ]
    write_csv(out_csv, rows, fieldnames=fieldnames)

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
