"""Paths, contracts, raw-data immutability and auditing - against the REAL files."""
import re
import stat
from pathlib import Path

import pytest

from src import config
from src.data_loading import ContractError, audit_columns, load_raw, sha256_file


def test_paths_are_project_relative():
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    for p in (config.RAW_DIR, config.TABLES_DIR, config.FIGURES_DIR, config.MODELS_DIR):
        assert config.PROJECT_ROOT in p.parents


def test_no_hardcoded_absolute_paths_in_code():
    pattern = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:/Users|/home/|OneDrive")
    for folder in ("src", "scripts"):
        for py in (config.PROJECT_ROOT / folder).glob("*.py"):
            assert not pattern.search(py.read_text(encoding="utf-8")), py.name


def test_seed_and_protocol():
    assert config.RANDOM_STATE == 42
    assert config.TEST_SIZE == 0.20
    assert config.CV_FOLDS == 5


@pytest.mark.parametrize("contract", list(config.CONTRACTS.values()), ids=lambda c: c.key)
def test_raw_file_matches_contract(contract):
    df = load_raw(contract)                      # raises ContractError on any mismatch
    assert df.shape == (contract.n_rows, contract.n_cols)
    assert tuple(df.columns) == contract.columns


def test_verified_shapes_from_phase1():
    assert load_raw(config.STUDENT).shape == (10_000, 6)
    assert load_raw(config.REVIEWS).shape == (10_000, 8)
    assert load_raw(config.CAFE).shape == (10_000, 8)


def test_loading_does_not_modify_raw_files():
    before = {c.key: sha256_file(c.path) for c in config.CONTRACTS.values()}
    for c in config.CONTRACTS.values():
        load_raw(c)
    after = {c.key: sha256_file(c.path) for c in config.CONTRACTS.values()}
    assert before == after


def test_raw_files_are_read_only():
    for c in config.CONTRACTS.values():
        assert not (c.path.stat().st_mode & stat.S_IWRITE), f"{c.filename} is writable"


def test_contract_mismatch_is_loud(tmp_path, monkeypatch):
    bad = tmp_path / config.STUDENT.filename
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    with pytest.raises(ContractError):
        load_raw(config.STUDENT)                  # checksum differs
    with pytest.raises(ContractError):
        load_raw(config.STUDENT, verify_checksum=False)   # shape differs


def test_excluded_columns_never_features():
    for c in (config.STUDENT, config.REVIEWS):
        assert not set(c.feature_columns) & set(c.excluded)
        assert c.target not in c.feature_columns


def test_audit_reports_real_missing_counts():
    audit = audit_columns(load_raw(config.REVIEWS))
    assert audit.loc["Review", "n_missing"] == 45
    assert audit.loc["Rating", "n_missing"] == 38
    assert audit.loc["7514", "n_missing"] == 9_999
    assert audit_columns(load_raw(config.STUDENT))["n_missing"].sum() == 0
