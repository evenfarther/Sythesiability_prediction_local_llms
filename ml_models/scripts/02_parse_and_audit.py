#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VALID_HEM_ONLY_JSONL,
    DEFAULT_VALID_JSONL,
    ensure_dir,
    extract_composition_from_user_text,
    extract_label_from_messages,
    extract_user_text,
    feature_names_v1,
    get_env_info,
    parse_composition_to_features_v1,
    read_jsonl,
    require_parquet_engine,
    split_name_to_sample_id,
    write_json,
    write_text,
)


def load_or_extract_dataset(datasets_dir: Path, *, split: str, jsonl_path: str) -> pd.DataFrame:
    parquet_path = datasets_dir / f"pn_{split}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    # Fallback: extract directly from JSONL (allows running 02 without running 01).
    rows = []
    for line_no, obj in enumerate(read_jsonl(jsonl_path), start=1):
        messages = obj.get("messages") or []
        try:
            label = extract_label_from_messages(messages)
            user_text = extract_user_text(messages)
            composition_raw = extract_composition_from_user_text(user_text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSONL row {split}:{line_no} ({jsonl_path})") from e

        sample_id = split_name_to_sample_id(split, line_no)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "line_no": line_no,
                "composition_raw": composition_raw,
                "label": int(label),
                "user_text": str(user_text),
                "source_jsonl": str(jsonl_path),
            }
        )

    df = pd.DataFrame.from_records(rows)
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)
    return df


def parse_split(
    df: pd.DataFrame,
    *,
    split: str,
    r_entropy: float,
    audits_dir: Path,
    features_dir: Path,
    datasets_dir: Path,
    overwrite: bool,
    limit_rows: int | None,
) -> dict[str, Any]:
    """Parse compositions, build features_v1, and write per-split artifacts."""
    if limit_rows is not None:
        df = df.head(int(limit_rows)).copy()

    n_in = int(df.shape[0])

    bad_rows = []
    ok_rows = []

    feat_names = feature_names_v1(r_entropy)
    n_features = len(feat_names)
    if n_features != 122:
        raise ValueError(f"Unexpected v1 feature length: {n_features} (expected 122)")

    X = np.empty((n_in, n_features), dtype=np.float32)
    y = np.empty((n_in,), dtype=np.int8)

    j = 0
    for row in df.itertuples(index=False):
        composition_raw = getattr(row, "composition_raw")
        label = int(getattr(row, "label"))
        sample_id = getattr(row, "sample_id")
        try:
            parsed = parse_composition_to_features_v1(composition_raw, r_entropy=r_entropy)
            X[j, :118] = parsed.frac_vector
            X[j, 118:] = np.array(
                [
                    float(parsed.n_elements),
                    float(parsed.max_frac),
                    float(parsed.min_frac),
                    float(parsed.s_mix),
                ],
                dtype=np.float32,
            )
            y[j] = label
            ok_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "composition_raw": composition_raw,
                    "label": label,
                    "reduced_formula": parsed.reduced_formula,
                    "n_elements": parsed.n_elements,
                    "max_frac": parsed.max_frac,
                    "min_frac": parsed.min_frac,
                    "s_mix": parsed.s_mix,
                }
            )
            j += 1
        except Exception as e:
            bad_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "composition_raw": composition_raw,
                    "label": label,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    X = X[:j, :]
    y = y[:j]

    ok_df = pd.DataFrame.from_records(ok_rows)
    bad_df = pd.DataFrame.from_records(bad_rows)

    ensure_dir(datasets_dir)
    ensure_dir(features_dir)
    ensure_dir(audits_dir)

    parsed_path = datasets_dir / f"pn_{split}_parsed.parquet"
    bad_path = audits_dir / f"bad_rows_{split}.parquet"
    features_path = features_dir / f"features_v1_{split}.npz"

    for p in [parsed_path, bad_path, features_path]:
        if p.exists() and not overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (use --overwrite)")

    ok_df.to_parquet(parsed_path, index=False)
    if not bad_df.empty:
        bad_df.to_parquet(bad_path, index=False)

    np.savez_compressed(
        features_path,
        X=X,
        y=y,
        sample_id=ok_df["sample_id"].to_numpy(dtype=str),
        feature_names=np.array(feat_names, dtype=str),
        split=np.array([split], dtype=str),
        r_entropy=np.array([r_entropy], dtype=np.float32),
    )

    return {
        "split": split,
        "n_in": n_in,
        "n_ok": int(ok_df.shape[0]),
        "n_bad": int(bad_df.shape[0]),
        "parsed_parquet": str(parsed_path),
        "features_npz": str(features_path),
        "bad_rows_parquet": str(bad_path) if not bad_df.empty else None,
    }


def main() -> None:
    """CLI entrypoint for parsing, auditing, and feature building."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="ml_models/artifacts/datasets")
    ap.add_argument("--features_dir", default="ml_models/artifacts/features")
    ap.add_argument("--audits_dir", default="ml_models/artifacts/audits")
    ap.add_argument("--train_jsonl", default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--valid_jsonl", default=DEFAULT_VALID_JSONL)
    ap.add_argument("--valid_hem_only_jsonl", default=DEFAULT_VALID_HEM_ONLY_JSONL)
    ap.add_argument("--r_entropy", type=float, default=1.0)
    ap.add_argument(
        "--limit_rows",
        type=int,
        default=None,
        help="Debug option: only parse the first N rows per split (do NOT use for final results).",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_parquet_engine()

    datasets_dir = Path(args.datasets_dir)
    features_dir = Path(args.features_dir)
    audits_dir = Path(args.audits_dir)
    ensure_dir(datasets_dir)
    ensure_dir(features_dir)
    ensure_dir(audits_dir)

    env_info = get_env_info()
    write_json(audits_dir / "env.json", env_info)

    dfs = {
        "train": load_or_extract_dataset(datasets_dir, split="train", jsonl_path=args.train_jsonl),
        "valid": load_or_extract_dataset(datasets_dir, split="valid", jsonl_path=args.valid_jsonl),
        "valid_hem_only": load_or_extract_dataset(
            datasets_dir, split="valid_hem_only", jsonl_path=args.valid_hem_only_jsonl
        ),
    }

    summary = {
        "r_entropy": float(args.r_entropy),
        "splits": {},
        "outputs": {},
    }

    for split, df in dfs.items():
        info = parse_split(
            df,
            split=split,
            r_entropy=float(args.r_entropy),
            audits_dir=audits_dir,
            features_dir=features_dir,
            datasets_dir=datasets_dir,
            overwrite=args.overwrite,
            limit_rows=args.limit_rows,
        )
        summary["splits"][split] = info
        print(f"[ok] parsed {split}: in={info['n_in']} ok={info['n_ok']} bad={info['n_bad']}")

    # Leakage report: reduced_formula overlap between train and valid.
    train_parsed = pd.read_parquet(datasets_dir / "pn_train_parsed.parquet")
    valid_parsed = pd.read_parquet(datasets_dir / "pn_valid_parsed.parquet")
    hem_parsed = pd.read_parquet(datasets_dir / "pn_valid_hem_only_parsed.parquet")

    train_counts = train_parsed["reduced_formula"].value_counts()
    valid_counts = valid_parsed["reduced_formula"].value_counts()
    hem_counts = hem_parsed["reduced_formula"].value_counts()
    overlap = train_counts.index.intersection(valid_counts.index)
    leakage_df = pd.DataFrame(
        {
            "reduced_formula": overlap,
            "n_train": train_counts.loc[overlap].to_numpy(),
            "n_valid": valid_counts.loc[overlap].to_numpy(),
        }
    ).sort_values(["n_train", "n_valid"], ascending=False)

    leakage_path = audits_dir / "leakage_reduced_formula.csv"
    if leakage_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {leakage_path} (use --overwrite)")
    leakage_df.to_csv(leakage_path, index=False)
    summary["outputs"]["leakage_csv"] = str(leakage_path)

    # Leakage report: reduced_formula overlap between train and valid_hem_only.
    overlap_hem = train_counts.index.intersection(hem_counts.index)
    leakage_hem_df = pd.DataFrame(
        {
            "reduced_formula": overlap_hem,
            "n_train": train_counts.loc[overlap_hem].to_numpy(),
            "n_hem_only": hem_counts.loc[overlap_hem].to_numpy(),
        }
    ).sort_values(["n_train", "n_hem_only"], ascending=False)
    leakage_hem_path = audits_dir / "leakage_reduced_formula_hem_only.csv"
    if leakage_hem_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {leakage_hem_path} (use --overwrite)")
    leakage_hem_df.to_csv(leakage_hem_path, index=False)
    summary["outputs"]["leakage_hem_only_csv"] = str(leakage_hem_path)
    summary["outputs"]["env_json"] = str(audits_dir / "env.json")

    # Dataset audit markdown.
    def split_counts(df_: pd.DataFrame) -> str:
        n = int(df_.shape[0])
        n_pos = int((df_["label"] == 1).sum())
        n_neg = int((df_["label"] == 0).sum())
        ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")
        return f"n={n:,} (P={n_pos:,}, N={n_neg:,}, N/P={ratio:.3g})"

    md = []
    md.append("# Dataset Audit (v1)")
    md.append("")
    md.append("## Splits (raw)")
    md.append(f"- train: {split_counts(dfs['train'])}")
    md.append(f"- valid(final test): {split_counts(dfs['valid'])}")
    md.append(f"- valid_hem_only: {split_counts(dfs['valid_hem_only'])}  (note: current N=0)")
    md.append("")
    md.append("## Parsing / Feature Build")
    for split, info in summary["splits"].items():
        md.append(f"- {split}: in={info['n_in']:,}, ok={info['n_ok']:,}, bad={info['n_bad']:,}")
    md.append("")
    md.append("## Leakage Check (reduced_formula overlap)")
    md.append(f"- overlap_unique_reduced_formula: {int(leakage_df.shape[0]):,}")
    md.append(f"- overlap_unique_reduced_formula_vs_hem_only: {int(leakage_hem_df.shape[0]):,}")
    md.append("")
    md.append("### Notes")
    md.append("- `valid_llm_pn.jsonl` is fixed as final test and is not used for tuning or threshold selection.")
    md.append("- `valid_hem_only` currently has no negatives, so ROC-AUC/TNR are undefined and recall is the key metric.")
    md.append("")

    audit_md_path = audits_dir / "dataset_audit_v1.md"
    if audit_md_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {audit_md_path} (use --overwrite)")
    write_text(audit_md_path, "\n".join(md) + "\n")
    summary["outputs"]["audit_md"] = str(audit_md_path)

    write_json(audits_dir / "parse_audit_manifest.json", summary)


if __name__ == "__main__":
    main()
