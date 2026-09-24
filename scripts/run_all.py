"""Execution engine: run the track notebooks top to bottom, then validate.

The notebooks ARE the implementation, so "running a track" means executing
its notebook in a fresh kernel, exactly as an examiner would with
"Restart & Run All". Executing only replaces cell outputs; Markdown cells
(including the team's written analysis) are left untouched.

    python scripts/run_all.py --track all            # reported run (in place)
    python scripts/run_all.py --track regression --mode dev

--mode dev sets ML_CAPSTONE_MODE=dev: every table and figure the notebook
saves gets a "_dev" suffix, and the executed notebook is written to
notebooks/_dev/ instead of overwriting the real one.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACK_NOTEBOOKS = {
    "regression": "regression.ipynb",
    "classification": "classification.ipynb",
}


def execute_notebook(path: Path, out_path: Path, timeout: int) -> None:
    # Imported lazily so that --help works without the notebook tooling.
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", choices=["regression", "classification", "all"], default="all")
    parser.add_argument("--mode", choices=["full", "dev"], default="full")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)   # keep our lines in order with subprocess output
    if sys.platform == "win32":   # pyzmq needs a selector event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ["ML_CAPSTONE_MODE"] = args.mode       # inherited by the kernel

    tracks = list(TRACK_NOTEBOOKS) if args.track == "all" else [args.track]
    if args.mode == "dev":
        print("=" * 70 + "\nDEV MODE - outputs carry the '_dev' suffix and are NOT reported results\n" + "=" * 70)

    failed = False
    for track in tracks:
        nb_path = ROOT / "notebooks" / TRACK_NOTEBOOKS[track]
        out_path = nb_path if args.mode == "full" else ROOT / "notebooks" / "_dev" / f"{nb_path.stem}_dev.ipynb"
        print(f"[RUN ] {track:<15} {nb_path.relative_to(ROOT)}")
        t0 = time.perf_counter()
        try:
            execute_notebook(nb_path, out_path, args.timeout)
            print(f"[DONE] {track:<15} {time.perf_counter() - t0:6.1f}s -> {out_path.relative_to(ROOT)}")
        except Exception as exc:   # report and continue so every track is attempted
            failed = True
            print(f"[FAIL] {track:<15} {type(exc).__name__}: {str(exc)[:2000]}")

    if not args.no_validate:
        print("\n[RUN ] validation")
        failed |= subprocess.call([sys.executable, str(ROOT / "scripts" / "validate_project.py")]) != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
