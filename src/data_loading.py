"""Loading, auditing and provenance for the raw datasets.

Rules enforced here:
* Raw files are read, never written. ``install_raw_file`` copies a file into
  ``data/raw`` once, verifies its checksum and marks it read-only.
* Every load is checked against the dataset contract in ``config.py``. A
  mismatch raises ``ContractError``; nothing is silently corrected.
* Loading is column-preserving: no cleaning, imputation or encoding happens
  here. Those steps live in the notebooks and in fitted pipelines.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .config import DatasetContract


class ContractError(RuntimeError):
    """Raised when a raw file does not match its declared contract."""


# --------------------------------------------------------------------------
# Checksums and raw-data installation
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_raw_file(contract: DatasetContract, source_dir: Path) -> str:
    """Copy one dataset from ``source_dir`` into ``data/raw`` and verify it.

    Never overwrites: if the destination already exists it is only verified.
    Returns a short status string.
    """
    source = Path(source_dir) / contract.filename
    dest = contract.path
    if dest.exists():
        actual = sha256_file(dest)
        if actual != contract.sha256:
            raise ContractError(
                f"{dest} exists but its checksum {actual} does not match the contract "
                f"{contract.sha256}. Refusing to overwrite raw data; investigate manually."
            )
        return f"already present, checksum OK: {dest.name}"
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    actual = sha256_file(source)
    if actual != contract.sha256:
        raise ContractError(
            f"Source {source} has checksum {actual}, expected {contract.sha256}. "
            "This is not the file the project was built on."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    os.chmod(dest, stat.S_IREAD)   # read-only: raw data is immutable
    return f"installed and verified: {dest.name}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(contract: DatasetContract, verify_checksum: bool = True) -> pd.DataFrame:
    """Read a raw CSV and verify it against its contract.

    Checks, in order: file exists, SHA-256 matches, row count matches,
    column names and order match. Any failure raises ``ContractError``.
    """
    path = contract.path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/setup_data.py --source <folder>` "
            "(see docs/dataset_sources.md for where to obtain the file)."
        )
    if verify_checksum:
        actual = sha256_file(path)
        if actual != contract.sha256:
            raise ContractError(
                f"Checksum mismatch for {path.name}: got {actual}, expected {contract.sha256}"
            )
    df = pd.read_csv(path)
    problems = []
    if len(df) != contract.n_rows:
        problems.append(f"rows: got {len(df)}, expected {contract.n_rows}")
    if tuple(df.columns) != contract.columns:
        problems.append(f"columns: got {list(df.columns)}, expected {list(contract.columns)}")
    if problems:
        raise ContractError(f"{path.name} violates its contract -> " + "; ".join(problems))
    return df


# --------------------------------------------------------------------------
# Auditing
# --------------------------------------------------------------------------
def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column audit: dtype, missing count/percent, unique count, example value."""
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        example = non_null.iloc[0] if len(non_null) else None
        if isinstance(example, str) and len(example) > 60:
            example = example[:57] + "..."
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(100 * series.isna().mean(), 2),
                "n_unique": int(series.nunique(dropna=True)),
                "example": example,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def duplicate_report(df: pd.DataFrame, subsets: dict[str, list[str] | None]) -> pd.DataFrame:
    """Count duplicate rows under several candidate keys.

    ``subsets`` maps a readable name to a list of columns (None = all columns).
    ``n_duplicates`` counts rows that repeat an earlier row under that key.
    """
    rows = []
    for name, cols in subsets.items():
        rows.append(
            {
                "key": name,
                "columns": "all columns" if cols is None else ", ".join(cols),
                "n_duplicates": int(df.duplicated(subset=cols).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("key")


def dataset_summary(df: pd.DataFrame, contract: DatasetContract) -> dict:
    """High-level facts about a raw frame: shape, missing cells, duplicates."""
    return {
        "dataset": contract.key,
        "file": contract.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "target_column": contract.target,
        "synthetic": contract.synthetic,
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def provenance_record(contract: DatasetContract) -> dict:
    """Checksum, size and shape of a raw file, measured now."""
    path = contract.path
    df = pd.read_csv(path)
    actual = sha256_file(path)
    return {
        "dataset": contract.key,
        "file": contract.filename,
        "source_url": contract.source_url,
        "bytes": path.stat().st_size,
        "sha256": actual,
        "sha256_matches_contract": actual == contract.sha256,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_provenance(contracts=None, out_path: Path | None = None) -> Path:
    """Write provenance records for the given contracts to a JSON file."""
    contracts = contracts or list(config.CONTRACTS.values())
    out_path = out_path or config.PROCESSED_DIR / "provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [provenance_record(c) for c in contracts]
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out_path
