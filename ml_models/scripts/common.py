from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_TRAIN_JSONL = "/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl"
DEFAULT_VALID_JSONL = "/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl"
DEFAULT_VALID_HEM_ONLY_JSONL = "/DATA/npj_compt.mat_project_github/data/valid_hem_only_llm_pn.jsonl"


COMPOSITION_RE = re.compile(
    r"Is the material\s+(?P<formula>.+?)\s+likely synthesizable\??",
    flags=re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_parquet_engine() -> None:
    """Exit with a helpful message if parquet IO dependencies are missing."""
    try:
        import pyarrow  # noqa: F401

        return
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return
        except Exception as e:
            raise SystemExit(
                "Parquet IO backend is missing. Install one of:\n"
                "  - conda install -c conda-forge pyarrow\n"
                "  - pip install pyarrow\n"
                "Then re-run the script."
            ) from e


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from e


def extract_label_from_messages(messages: List[Dict[str, Any]]) -> int:
    label_text: Optional[str] = None
    for msg in messages:
        if msg.get("role") == "assistant":
            label_text = msg.get("content")
    if label_text is None:
        raise ValueError("Missing assistant label message")
    label_text = str(label_text).strip()
    if label_text == "P":
        return 1
    if label_text == "N":
        return 0
    raise ValueError(f"Unexpected label: {label_text!r} (expected 'P' or 'N')")


def extract_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    raise ValueError("Missing user message")


def extract_composition_from_user_text(user_text: str) -> str:
    m = COMPOSITION_RE.search(user_text)
    if m:
        return m.group("formula").strip()

    lower = user_text.lower()
    key1 = "is the material"
    key2 = "likely synthesizable"
    if key1 in lower and key2 in lower:
        start = lower.find(key1) + len(key1)
        end = lower.find(key2, start)
        if end > start:
            return user_text[start:end].strip(" \t\n\r?.,")

    raise ValueError(f"Could not extract composition from user text: {user_text!r}")


def split_name_to_sample_id(split: str, line_no: int) -> str:
    return f"{split}:{line_no}"


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def safe_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def get_env_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "git_commit": get_git_commit(),
        "numpy": safe_version("numpy"),
        "pandas": safe_version("pandas"),
        "scikit_learn": safe_version("sklearn"),
        "pymatgen": safe_version("pymatgen"),
        "xgboost": safe_version("xgboost"),
        "torch": safe_version("torch"),
    }


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def element_feature_names() -> List[str]:
    # 1..118 inclusive
    names: List[str] = []
    for z in range(1, 119):
        el = Element.from_Z(z)
        names.append(f"frac_{el.symbol}")
    return names


def feature_names_v1(r_entropy: float) -> List[str]:
    return (
        element_feature_names()
        + [
            "n_elements",
            "max_frac",
            "min_frac",
            f"s_mix_r={r_entropy:g}",
        ]
    )


@dataclass(frozen=True)
class ParsedComposition:
    composition_raw: str
    reduced_formula: str
    n_elements: int
    max_frac: float
    min_frac: float
    s_mix: float
    frac_vector: np.ndarray  # shape (118,), float32


def parse_composition_to_features_v1(
    composition_raw: str,
    *,
    r_entropy: float = 1.0,
) -> ParsedComposition:
    comp = Composition(composition_raw)
    reduced_formula = comp.reduced_formula

    frac_vec = np.zeros((118,), dtype=np.float32)
    fractions: List[float] = []
    for el in comp.elements:
        frac = float(comp.get_atomic_fraction(el))
        fractions.append(frac)
        z = int(el.Z)
        if 1 <= z <= 118:
            frac_vec[z - 1] = frac
        else:
            raise ValueError(f"Unexpected element Z={z} for {el} in {composition_raw!r}")

    if not fractions:
        raise ValueError(f"Empty composition: {composition_raw!r}")

    n_elements = len(fractions)
    max_frac = float(max(fractions))
    min_frac = float(min(fractions))

    # Configurational entropy proxy (scaled by r_entropy). Use natural log.
    s_mix = 0.0
    for c in fractions:
        if c > 0:
            s_mix += -c * math.log(c)
    s_mix *= float(r_entropy)

    return ParsedComposition(
        composition_raw=composition_raw,
        reduced_formula=reduced_formula,
        n_elements=n_elements,
        max_frac=max_frac,
        min_frac=min_frac,
        s_mix=s_mix,
        frac_vector=frac_vec,
    )


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_total = int(y_true.shape[0])

    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    bacc = safe_div(tpr + tnr, 2.0) if not (math.isnan(tpr) or math.isnan(tnr)) else float("nan")

    mcc = float("nan")
    # MCC is only meaningful when both classes exist in y_true.
    # If the denominator is 0 (e.g., constant predictions), follow the common convention mcc=0.
    if n_pos > 0 and n_neg > 0:
        den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        if den == 0.0:
            mcc = 0.0
        else:
            mcc = float((tp * tn - fp * fn) / den)

    auc = float("nan")
    if n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = float("nan")

    return {
        "threshold": float(threshold),
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr_recall": tpr,
        "tnr_specificity": tnr,
        "precision": precision,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "roc_auc": auc,
    }


def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return {"fpr": None, "tpr": None, "thresholds": None}
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def choose_threshold_max_bacc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    roc = compute_roc(y_true, y_score)
    if roc["thresholds"] is None:
        # Can't optimize bAcc if only one class exists.
        return {"threshold": 0.5, "balanced_accuracy": float("nan")}

    fpr = roc["fpr"]
    tpr = roc["tpr"]
    thresholds = roc["thresholds"]

    tnr = 1.0 - fpr
    bacc = 0.5 * (tpr + tnr)
    best_idx = int(np.argmax(bacc))

    # Deterministic tie-break: if multiple maxima, pick the one with higher TPR, then higher TNR.
    max_bacc = float(bacc[best_idx])
    candidates = np.where(bacc == max_bacc)[0]
    if candidates.size > 1:
        cand = candidates
        tpr_c = tpr[cand]
        tnr_c = tnr[cand]
        # sort by (tpr, tnr) descending
        best_idx = int(cand[np.lexsort((-tnr_c, -tpr_c))][0])

    return {"threshold": float(thresholds[best_idx]), "balanced_accuracy": float(bacc[best_idx])}
