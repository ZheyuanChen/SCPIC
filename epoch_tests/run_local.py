"""Generate and run the SCPIC integration cases with a local EPOCH2D."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from generate_profiles import GENERATORS, generate_all

HERE = Path(__file__).resolve().parent
DEFAULT_EPOCH = HERE.parents[1] / "epoch_dev" / "epoch2d" / "bin" / "epoch2d"


def main():
    """Generate selected cases, run EPOCH2D and retain its combined log."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch-bin",
        type=Path,
        default=Path(os.environ.get("EPOCH2D_BIN", DEFAULT_EPOCH)),
    )
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("cases", nargs="*", choices=sorted(GENERATORS))
    args = parser.parse_args()
    if not args.epoch_bin.is_file():
        parser.error(f"EPOCH2D executable not found: {args.epoch_bin}")
    if args.ranks < 1:
        parser.error("--ranks must be positive")

    selected = args.cases or list(GENERATORS)
    run_root = HERE / "runs"
    generate_all(run_root)
    command = [str(args.epoch_bin)]
    if args.ranks > 1:
        command = ["mpirun", "-np", str(args.ranks), *command]

    for name in selected:
        run_dir = run_root / name
        for pattern in ("*.sdf", "*.visit", "epoch*.dat", "deck.status", "epoch.log"):
            for old_output in run_dir.glob(pattern):
                old_output.unlink()
        shutil.copy2(HERE / "cases" / name / "input.deck", run_dir / "input.deck")
        print(f"Running {name} with {args.ranks} rank(s)", flush=True)
        completed = subprocess.run(
            command,
            input=f"{run_dir.resolve()}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "epoch.log").write_text(completed.stdout)
        if completed.returncode:
            raise SystemExit(
                f"{name} failed with exit code {completed.returncode}; "
                f"see {run_dir / 'epoch.log'}"
            )
        dumps = sorted(run_dir.glob("*.sdf"))
        if not dumps:
            raise SystemExit(f"{name} completed but produced no SDF dumps")
        print(f"  passed loader/run smoke test ({len(dumps)} dumps)")


if __name__ == "__main__":
    main()
