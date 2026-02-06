#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a *base model only* (no adapters) on PN validation using token logits.

This is a thin wrapper around `eval_pn_checkpoints.py` which:
- forces baseline evaluation ON
- forces adapter checkpoint evaluation OFF (ckpt_regex matches nothing)

Rationale:
We want an unambiguous "base_model" row for sanity checks (e.g., random-init vs pretrained).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PN base model (no adapters)")
    parser.add_argument("--base_model", type=str, required=True, help="HF id or local model directory")
    parser.add_argument("--val_jsonl", type=str, default="/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="/DATA/gpt-oss/cache_eval_pn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Dummy directory used only to namespace tokenization caches. Defaults to output_csv parent dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_csv if it exists.")

    args = parser.parse_args()

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not args.overwrite:
        print(f"[Skip] output_csv already exists (use --overwrite): {out_csv}")
        return

    checkpoints_dir = args.checkpoints_dir or str(out_csv.parent)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Ensure sibling import works no matter where this script is launched from.
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    from eval_pn_checkpoints import evaluate_checkpoints  # noqa: E402

    # Force: baseline only.
    evaluate_checkpoints(
        base_model=args.base_model,
        checkpoints_dir=checkpoints_dir,
        val_jsonl=args.val_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=args.device,
        output_csv=str(out_csv),
        threshold=args.threshold,
        ckpt_regex=r"^$",  # match nothing -> adapters are never evaluated
        include_baseline=True,
        resume=False,
    )


if __name__ == "__main__":
    main()

