#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: check the metric library and views, and the files made from them.

Runs the metric library, analysis view and membw tree spec validators, then
regenerates the metric lock (metric_lock.py) and the metric description files
(metric_description_manager.py) and reports the generated files that change. With
--fix it also writes them, so pre-commit fails until the developer reviews and stages
their diff: that staging confirms the library or view edit was intended.
"""

import argparse
import contextlib
import filecmp
import io
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

TOOLS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_DIR.parent
sys.path[:0] = [str(TOOLS_DIR), str(PROJECT_ROOT / "src")]

import validate_membw_tree_spec
import validate_metric_library
import validate_views
from config_management import metric_lock
from config_management.metric_description_manager import (
    DOCS_DIR,
    PER_ARCH_DIR,
    generate_docs_from_per_arch,
    sync_arch,
)

from rocprof_compute_soc.metric_library import loader

# The directories this hook generates, in the order regenerate() takes them.
GENERATED_DIRS = (metric_lock.LOCK_DIR, PER_ARCH_DIR, DOCS_DIR)


def regenerate(lock_dir: Path, per_arch_dir: Path, docs_dir: Path) -> None:
    """Write the metric lock and the metric description files from the shipped
    metric library and views."""
    metric_lock.write_lock(lock_dir, loader.default_metric_library_dir())
    with contextlib.redirect_stdout(io.StringIO()):
        for arch in sorted(loader.METRIC_LIBRARY_PARENT_ARCH):
            if not sync_arch(arch, str(per_arch_dir)):
                raise RuntimeError(f"Cannot sync the metric descriptions of {arch}")
        if not generate_docs_from_per_arch(str(per_arch_dir), str(docs_dir), None):
            raise RuntimeError("Cannot generate the metric docs")


def stale_files(lock_dir: Path, per_arch_dir: Path, docs_dir: Path) -> List[Path]:
    """Return the generated files that regenerate() would add, change or remove."""
    with tempfile.TemporaryDirectory() as tmp:
        copies = []
        for index, directory in enumerate((lock_dir, per_arch_dir, docs_dir)):
            copy = Path(tmp) / str(index)
            shutil.copytree(directory, copy)
            copies.append(copy)
        regenerate(*copies)
        return [
            directory / name
            for directory, copy in zip((lock_dir, per_arch_dir, docs_dir), copies)
            for name in _differing_names(filecmp.dircmp(directory, copy))
        ]


def _differing_names(comparison: filecmp.dircmp) -> List[str]:
    """Return the file names that differ between two flat directories."""
    names = comparison.left_only + comparison.right_only + comparison.diff_files
    _match, mismatch, errors = filecmp.cmpfiles(
        comparison.left, comparison.right, comparison.same_files, shallow=False
    )
    return sorted(names + mismatch + errors)


def main() -> int:
    """Entry point. Returns 1 if a validator fails or, without --fix, if a generated
    file is stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix", action="store_true", help="Write the generated files that change"
    )
    args = parser.parse_args()

    validators = (validate_metric_library, validate_views, validate_membw_tree_spec)
    if any([validator.main() != 0 for validator in validators]):
        return 1
    stale = stale_files(*GENERATED_DIRS)
    if not stale:
        return 0
    if args.fix:
        regenerate(*GENERATED_DIRS)
        print("Regenerated; review and stage their diff:")
    else:
        print("Stale; run tools/config_management/check_metric_config.py --fix:")
    for path in stale:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0 if args.fix else 1


if __name__ == "__main__":
    sys.exit(main())
