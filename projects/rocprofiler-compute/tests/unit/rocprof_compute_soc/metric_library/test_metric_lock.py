# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric lock (tools/config_management/metric_lock.py) and the
hook that keeps it current (check_metric_config.py).

The hook checks that the shipped lock is current; these tests cover what that check
takes on trust: a metric's lock entry follows inheritance, and the hook reports every
lock file that is out of date.
"""

import shutil

from config_management.check_metric_config import GENERATED_DIRS, stale_files
from config_management.metric_lock import LOCK_DIR, arch_lock

from rocprof_compute_soc.metric_library import loader

# A gfx90a metric that gfx942 and gfx950 inherit unchanged and gfx908 lacks.
INHERITED_METRIC = "compute.bf16_ops"
FORMULA = "512 * SUM(SQ_INSTS_VALU_MFMA_MOPS_BF16) / SUM($denom)"
EDITED_FORMULA = "512 * SUM(SQ_INSTS_VALU_MFMA_MOPS_BF16 + TA_BUSY) / SUM($denom)"


def test_a_parent_edit_reaches_the_lock_of_every_arch_that_inherits_it(tmp_path):
    """Editing a gfx90a metric changes its formula and counters in the gfx90a, gfx942
    and gfx950 locks."""
    metric_dir = tmp_path / "metric_library"
    shutil.copytree(loader.default_metric_library_dir(), metric_dir)
    compute_file = metric_dir / "gfx90a" / "compute.yaml"
    text = compute_file.read_text(encoding="utf-8")
    compute_file.write_text(text.replace(FORMULA, EDITED_FORMULA), encoding="utf-8")

    for arch in ("gfx90a", "gfx942", "gfx950"):
        entry = arch_lock(metric_dir, arch)[INHERITED_METRIC]
        assert entry["formula"] == EDITED_FORMULA
        assert "TA_BUSY" in entry["counters"]
    assert INHERITED_METRIC not in arch_lock(metric_dir, "gfx908")


def test_stale_lock_files_are_reported(tmp_path):
    """A changed, a removed and an extra lock file are each reported."""
    lock_dir = tmp_path / "metric_lock"
    shutil.copytree(LOCK_DIR, lock_dir)
    (lock_dir / "gfx942.yaml").write_text("compute.bf16_ops: {}\n", encoding="utf-8")
    (lock_dir / "gfx950.yaml").unlink()
    (lock_dir / "gfx000.yaml").write_text("", encoding="utf-8")

    assert stale_files(lock_dir, *GENERATED_DIRS[1:]) == [
        lock_dir / "gfx000.yaml",
        lock_dir / "gfx942.yaml",
        lock_dir / "gfx950.yaml",
    ]
