#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import ensure_dir, write_text


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Aggregate run metrics into a Reviewer 1 baseline table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="ml_models/runs")
    ap.add_argument("--out_dir", default="ml_models/reports")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(runs_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for seed_dir in sorted(model_dir.glob("seed=*")):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name.split("=", 1)[-1]

            metrics_valid_path = seed_dir / "metrics_valid.json"
            metrics_hem_path = seed_dir / "metrics_valid_hem_only.json"
            thr_path = seed_dir / "threshold.json"
            if not metrics_valid_path.exists() or not metrics_hem_path.exists() or not thr_path.exists():
                continue

            m_valid = read_json(metrics_valid_path)
            m_hem = read_json(metrics_hem_path)
            thr = read_json(thr_path).get("threshold")

            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "valid_n_total": m_valid.get("n_total"),
                    "valid_n_pos": m_valid.get("n_pos"),
                    "valid_n_neg": m_valid.get("n_neg"),
                    "valid_threshold": thr,
                    "valid_roc_auc": m_valid.get("roc_auc"),
                    "valid_mcc": m_valid.get("mcc"),
                    "valid_tpr_recall": m_valid.get("tpr_recall"),
                    "valid_tnr_specificity": m_valid.get("tnr_specificity"),
                    "valid_precision": m_valid.get("precision"),
                    "valid_balanced_accuracy": m_valid.get("balanced_accuracy"),
                    "hem_only_n_total": m_hem.get("n_total"),
                    "hem_only_n_pos": m_hem.get("n_pos"),
                    "hem_only_n_neg": m_hem.get("n_neg"),
                    "hem_only_recall": m_hem.get("tpr_recall"),
                }
            )

    if not rows:
        raise SystemExit(f"No runs found under {runs_dir} (expected runs/<model>/seed=*/metrics_*.json)")

    df = pd.DataFrame.from_records(rows).sort_values(["model", "seed"])
    out_csv = out_dir / "reviewer1_major1_baselines_table.csv"
    df.to_csv(out_csv, index=False)

    notes = []
    notes.append("# Reviewer 1 Major Comment 1 — Baselines Notes")
    notes.append("")
    notes.append("- `valid_llm_pn.jsonl` is fixed as the final test split and is not used for tuning, threshold selection, or early stopping.")
    notes.append("- Thresholds are selected on `train_tune` split derived from `train_llm_pn.jsonl` using max balanced accuracy.")
    notes.append("- `valid_hem_only` currently has no negatives, so ROC-AUC/TNR are undefined and recall is emphasized.")
    notes.append("")
    out_md = out_dir / "reviewer1_major1_notes.md"
    write_text(out_md, "\n".join(notes) + "\n")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
