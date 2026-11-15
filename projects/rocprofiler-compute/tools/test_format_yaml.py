#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

import pytest
from format_yaml import check_file

# A metric library file with one formula that needs formatting and one formula with a
# dangling operator.
LIBRARY_WITH_BROKEN_FORMULA = textwrap.dedent("""\
    l2.hit:
      formula: SUM(TCC_HIT_sum * 2)
    lds.ai:
      formula: SUM(SQ_INSTS_VALU_MFMA_MOPS_F64 * 512 + ) / SUM(SQ_LDS_IDX_ACTIVE)
""")

# Metric library descriptions: one too long for its line, one folded although it fits
# the key line, one folded too narrow, and an empty one
LDS_UTIL = (
    "Percentage of the kernel's duration the LDS was busy executing LDS instructions, "
    "averaged over compute units."
)
UNFILLED_DESCRIPTIONS = textwrap.dedent(f"""\
    lds.util:
      description: {LDS_UTIL}
    lds.insts:
      description: >-
        Number of LDS instructions issued.
    lds.bank_conflict_rate:
      description: >-
        Percentage of LDS cycles spent
        on bank conflicts. High conflict rates reduce effective LDS bandwidth.
      unit: pct
    lds.unused:
      description: ''
""")

FILLED_DESCRIPTIONS = textwrap.dedent("""\
    lds.util:
      description: >-
        Percentage of the kernel's duration the LDS was busy executing LDS instructions,
        averaged over compute units.
    lds.insts:
      description: Number of LDS instructions issued.
    lds.bank_conflict_rate:
      description: >-
        Percentage of LDS cycles spent on bank conflicts. High conflict rates reduce
        effective LDS bandwidth.
      unit: pct
    lds.unused:
      description: ''
""")


def write_config(tmp_path, relative_path, text):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_unparseable_equation_reported(tmp_path):
    """A broken equation is reported by key; the other equations are still fixed."""
    path = write_config(
        tmp_path, "metric_library/gfx942/l2.yaml", LIBRARY_WITH_BROKEN_FORMULA
    )

    is_valid, issues = check_file(path, fix=True)

    assert not is_valid
    assert issues[0].startswith("  lds.ai.formula: cannot parse equation: ")
    content = path.read_text(encoding="utf-8")
    assert "formula: 2 * SUM(TCC_HIT_sum)" in content
    assert "MOPS_F64 * 512 + ) /" in content


def test_unformatted_library_formula_reported(tmp_path):
    path = write_config(
        tmp_path,
        "metric_library/gfx908/l2.yaml",
        "l2.ai:\n  formula: ( SUM( (a * 2) ) / SUM(b) )\n",
    )
    is_valid, issues = check_file(path)
    assert not is_valid
    assert "l2.ai.formula: equation needs normalization" in issues[0]


def test_library_descriptions_filled(tmp_path):
    """Metric library descriptions are filled to 88 columns, on the key line when they
    fit there."""
    path = write_config(
        tmp_path, "metric_library/gfx950/lds.yaml", UNFILLED_DESCRIPTIONS
    )
    is_valid, issues = check_file(path)
    assert not is_valid
    assert issues == ["  descriptions need filling to 88 columns"]
    check_file(path, fix=True)
    assert path.read_text(encoding="utf-8") == FILLED_DESCRIPTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
