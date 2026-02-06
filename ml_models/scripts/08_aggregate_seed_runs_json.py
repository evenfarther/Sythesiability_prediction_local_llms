#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    # Note: upstream metrics JSONs may contain NaN (Python json supports it by default).
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def mean_std(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"mean": None, "std": None}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    return {"mean": float(mean(xs)), "std": float(stdev(xs))}


def mcc_from_counts(*, tp: int, tn: int, fp: int, fn: int, n_pos: int, n_neg: int) -> Optional[float]:
    # MCC is only meaningful when both classes exist in y_true.
    if n_pos <= 0 or n_neg <= 0:
        return None
    den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    if den == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        default="ml_models/runs/svm_linearsvc",
        help="Directory containing seed=* subdirs.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <run_dir>/seed_summary.json).",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")

    out_path = Path(args.out) if args.out is not None else (run_dir / "seed_summary.json")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    seed_dirs = sorted([p for p in run_dir.glob("seed=*") if p.is_dir()], key=lambda p: int(p.name.split("=", 1)[-1]))
    if not seed_dirs:
        raise SystemExit(f"No seed dirs found under: {run_dir} (expected seed=*)")

    seed_runs: List[Dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed_str = seed_dir.name.split("=", 1)[-1]
        seed = int(seed_str)

        metrics_valid_path = seed_dir / "metrics_valid.json"
        metrics_hem_path = seed_dir / "metrics_valid_hem_only.json"
        thr_path = seed_dir / "threshold.json"
        cfg_path = seed_dir / "config.json"

        if not metrics_valid_path.exists():
            raise SystemExit(f"Missing: {metrics_valid_path}")
        if not metrics_hem_path.exists():
            raise SystemExit(f"Missing: {metrics_hem_path}")

        m_valid = read_json(metrics_valid_path)
        m_hem = read_json(metrics_hem_path)
        thr = read_json(thr_path).get("threshold") if thr_path.exists() else m_valid.get("threshold")
        cfg = read_json(cfg_path) if cfg_path.exists() else {}

        mcc = as_float(m_valid.get("mcc"))
        if mcc is None:
            mcc = mcc_from_counts(
                tp=int(m_valid.get("tp")),
                tn=int(m_valid.get("tn")),
                fp=int(m_valid.get("fp")),
                fn=int(m_valid.get("fn")),
                n_pos=int(m_valid.get("n_pos")),
                n_neg=int(m_valid.get("n_neg")),
            )

        seed_runs.append(
            {
                "seed": seed,
                "best_c": as_float(cfg.get("best_c")),
                "threshold": as_float(thr),
                "valid": {
                    "roc_auc": as_float(m_valid.get("roc_auc")),
                    "balanced_accuracy": as_float(m_valid.get("balanced_accuracy")),
                    "mcc": mcc,
                    "tpr_recall": as_float(m_valid.get("tpr_recall")),
                    "tnr_specificity": as_float(m_valid.get("tnr_specificity")),
                    "precision": as_float(m_valid.get("precision")),
                    "tp": int(m_valid.get("tp")),
                    "tn": int(m_valid.get("tn")),
                    "fp": int(m_valid.get("fp")),
                    "fn": int(m_valid.get("fn")),
                    "n_total": int(m_valid.get("n_total")),
                    "n_pos": int(m_valid.get("n_pos")),
                    "n_neg": int(m_valid.get("n_neg")),
                },
                "hem_only": {
                    "tpr_recall": as_float(m_hem.get("tpr_recall")),
                    "tp": int(m_hem.get("tp")),
                    "fn": int(m_hem.get("fn")),
                    "n_total": int(m_hem.get("n_total")),
                    "n_pos": int(m_hem.get("n_pos")),
                    "n_neg": int(m_hem.get("n_neg")),
                },
            }
    )

    # Aggregate stats (mean/std across seeds).
    valid_keys = ["roc_auc", "balanced_accuracy", "mcc", "tpr_recall", "tnr_specificity", "precision"]
    summary_valid: Dict[str, Any] = {}
    for k in valid_keys:
        summary_valid[k] = mean_std([as_float(r["valid"][k]) for r in seed_runs])

    best_c_counts: Dict[str, int] = {}
    for r in seed_runs:
        c = as_float(r.get("best_c"))
        if c is None:
            continue
        key = format(c, "g")
        best_c_counts[key] = best_c_counts.get(key, 0) + 1

    summary = {
        "threshold": mean_std([as_float(r["threshold"]) for r in seed_runs]),
        "best_c_counts": dict(sorted(best_c_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "valid": summary_valid,
        "hem_only": {"tpr_recall": mean_std([as_float(r["hem_only"]["tpr_recall"]) for r in seed_runs])},
    }

    # run_dir may be passed as '.'; Path('.').name is '' which breaks the "model" label.
    model_name = run_dir.name
    if not model_name:
        try:
            model_name = run_dir.resolve().name or str(run_dir.resolve())
        except Exception:
            model_name = str(run_dir)

    out_obj = {
        "model": model_name,
        "run_dir": str(run_dir),
        "created_utc": utc_now_iso(),
        "n_seeds": len(seed_runs),
        "seeds": [r["seed"] for r in seed_runs],
        "seed_runs": seed_runs,
        "summary": summary,
    }

    out_path.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
