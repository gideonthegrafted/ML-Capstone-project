"""Install the raw datasets into data/raw and verify them by SHA-256.

The raw CSVs are NOT committed to the repository (see docs/dataset_sources.md
for licensing and provenance). Each team member obtains the files once - from
the Kaggle pages listed there, or from the shared course folder - and runs:

    python scripts/setup_data.py --source "<folder containing the CSV files>"

Files are copied (never moved), verified against the checksums in
src/config.py and marked read-only. An existing file in data/raw is never
overwritten; it is only re-verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loading import ContractError, install_raw_file, write_provenance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="folder that contains the downloaded CSV files")
    args = parser.parse_args()

    ok = True
    for contract in config.CONTRACTS.values():
        try:
            print(f"[OK]   {contract.key:<20} {install_raw_file(contract, args.source)}")
        except (ContractError, FileNotFoundError) as exc:
            ok = False
            print(f"[FAIL] {contract.key:<20} {exc}")
            print(f"       expected file name: {contract.filename}   source: {contract.source_url}")
    if ok:
        print(f"Provenance written to {write_provenance().relative_to(config.PROJECT_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
