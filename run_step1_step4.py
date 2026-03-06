#!/usr/bin/env python3
"""Run Step 1 and Step 4 sequentially with a single command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
STEP_DIR = ROOT_DIR / "stage1_roadnet processing"

STEP1_SCRIPT = STEP_DIR / "step1_analyze_road_network_alt.py"
STEP4_SCRIPT = STEP_DIR / "step4_build_road_graph_fixed2.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input/output configuration."""
    parser = argparse.ArgumentParser(
        description="Run Step1 and Step4 sequentially with configurable input/output paths."
    )
    parser.add_argument(
        "--input-dir",
        default="Geneva",
        help="Input dataset directory containing shapefiles (default: Geneva)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output base directory for generated files (default: same as input-dir)",
    )
    return parser.parse_args()


def run_step(script_path: Path, step_name: str, env: dict[str, str]) -> None:
    """Run one step script and fail fast if it exits with non-zero code."""
    if not script_path.exists():
        raise FileNotFoundError(f"{step_name} script not found: {script_path}")

    print(f"\n=== Running {step_name} ===")
    print(f"Script: {script_path}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT_DIR),
        env=env,
        check=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with exit code {result.returncode} "
            f"after {elapsed:.2f}s"
        )

    print(f"{step_name} completed in {elapsed:.2f}s")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    env = os.environ.copy()
    env["ROADNET_INPUT_DIR"] = str(input_dir)
    env["ROADNET_OUTPUT_DIR"] = str(output_dir)

    print("Start pipeline: Step1 -> Step4")
    print(f"Python: {sys.executable}")
    print(f"Project root: {ROOT_DIR}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    pipeline_start = time.time()
    try:
        run_step(STEP1_SCRIPT, "Step1", env)
        run_step(STEP4_SCRIPT, "Step4", env)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 1

    total_elapsed = time.time() - pipeline_start
    print(f"\nPipeline finished successfully in {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
