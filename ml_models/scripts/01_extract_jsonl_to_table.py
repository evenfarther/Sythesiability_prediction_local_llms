#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VALID_HEM_ONLY_JSONL,
    DEFAULT_VALID_JSONL,
    ensure_dir,
    extract_composition_from_user_text,
    extract_label_from_messages,
    extract_user_text,
    read_jsonl,
    require_parquet_engine,
    split_name_to_sample_id,
    write_json,
)


def extract_split(jsonl_path: str, *, split: str) -> pd.DataFrame:
    """Extract all rows from a split JSONL into a table."""
    return extract_split_limited(jsonl_path, split=split, limit=None)


def extract_split_limited(jsonl_path: str, *, split: str, limit: int | None) -> pd.DataFrame:
    """Extract up to N rows (for debug) from a split JSONL into a table."""
    rows = []
    for line_no, obj in enumerate(read_jsonl(jsonl_path), start=1):
        if limit is not None and line_no > limit:
            break
        messages = obj.get("messages") or []
        label = extract_label_from_messages(messages)
        user_text = extract_user_text(messages)
        composition_raw = extract_composition_from_user_text(user_text)
        sample_id = split_name_to_sample_id(split, line_no)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "line_no": line_no,
                "composition_raw": composition_raw,
                "label": int(label),
                "user_text": user_text,
                "source_jsonl": str(jsonl_path),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    """CLI entrypoint for JSONL → parquet extraction."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--valid_jsonl", default=DEFAULT_VALID_JSONL)
    ap.add_argument("--valid_hem_only_jsonl", default=DEFAULT_VALID_HEM_ONLY_JSONL)
    ap.add_argument("--out_dir", default="ml_models/artifacts/datasets")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug option: only process the first N rows per split.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_parquet_engine()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    manifest = {
        "train_jsonl": args.train_jsonl,
        "valid_jsonl": args.valid_jsonl,
        "valid_hem_only_jsonl": args.valid_hem_only_jsonl,
        "outputs": {},
    }

    outputs = {
        "train": out_dir / "pn_train.parquet",
        "valid": out_dir / "pn_valid.parquet",
        "valid_hem_only": out_dir / "pn_valid_hem_only.parquet",
    }

    for split, out_path in outputs.items():
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

        jsonl_path = {
            "train": args.train_jsonl,
            "valid": args.valid_jsonl,
            "valid_hem_only": args.valid_hem_only_jsonl,
        }[split]

        df = extract_split_limited(jsonl_path, split=split, limit=args.limit)
        df.to_parquet(out_path, index=False)
        manifest["outputs"][split] = str(out_path)
        print(f"[ok] wrote {split}: {out_path} (rows={len(df)})")

    write_json(out_dir / "extract_manifest.json", manifest)


if __name__ == "__main__":
    main()
