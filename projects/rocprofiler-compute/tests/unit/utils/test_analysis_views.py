# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/analysis_views.py: the values a view row gives, the errors for
broken views, and the profile mode imports.
"""

import copy
import importlib
import sys

from conftest import ProfileModeImportGuard

import utils
from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    iter_view_rows,
    load_views,
    view_metric_errors,
    view_schema_errors,
    views_to_panel_configs,
)
from utils.utils_common import metric_ids_by_name

ARCH = "gfx942"
# gfx942's compute.ipc_issued per kernel: the per-dispatch partners of its summed
# collectables.
IPC_ISSUED_PER_KERNEL = (
    "$ipc_valu_vmem_salu_smem_per_dispatch"
    " + $ipc_branch_sendmsg_vskipped_lds_per_dispatch"
)
# A view with one mistake of each kind the shape check reports.
BROKEN_VIEW = {
    "view": "Compute Units - Compute Pipeline",
    "archs": ["gfx942"],
    "tables": [
        {
            "title": "Compute Speed-of-Light",
            "render": "chart",
        },
        {
            "title": "Pipeline Statistics",
            "header": {"metric": "Metric", "avg": "Avg", "unit": "Unit"},
            "metrics": [
                {"metric": "compute.ipc", "metrics": {"avg": "compute.ipc"}},
                {"metrics": {"avg": "compute.ipc"}},
                {"metric": "compute.salu_util", "values": {"xfer": "HBM"}},
                {"metric": "compute.vmem_util", "peak": "theoretical"},
                {"metric": "compute.mfma_util", "unit": "Percent"},
            ],
        },
    ],
}


# Moves to MetricLibrary.collectable_formulas() in the analyze cutover.
def _collectable_formulas(library):
    """Return the formula of each collectable of library, by name."""
    return {name: entry["formula"] for name, entry in library.collectables.items()}


def metric_rows(arch):
    """Return the arch's views, metrics, and panel config rows by numeric id."""
    views = load_views(arch)
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    metrics = library.metrics
    rows = {}
    panels = views_to_panel_configs(views, metrics, _collectable_formulas(library))
    for panel in panels.values():
        for source in panel["data source"]:
            table = source.get("metric_table")
            if not table:
                continue
            for name, metric_id in metric_ids_by_name(table).items():
                rows[metric_id] = (name, table["metric"][name])
    return views, metrics, rows


def test_rows_show_their_metric_as_the_view_asks():
    """Each display field of a view row changes what its panel config row shows."""
    _, metrics, rows = metric_rows(ARCH)
    ipc = metrics["compute.ipc_issued"]
    assert rows["11.2.1"] == (
        ipc["name"],
        {
            "avg": ipc["formula"],
            "min": f"MIN({IPC_ISSUED_PER_KERNEL})",
            "max": f"MAX({IPC_ISSUED_PER_KERNEL})",
            "unit": ipc["unit"],
        },
    )
    occupancy = metrics["wave.occupancy_pct"]
    assert rows["3.1.0"][1]["value"] == f"ROUND({occupancy['formula']}, 1)"
    valu_flops = metrics["compute.valu_flops_f16"]
    assert rows["4.1.0"][1]["peak"] == valu_flops["peak_empirical"]
    assert rows["2.1.7"][1]["pct_of_peak"] is False
    assert rows["16.4.0"][0] == "NC - Read"
    assert rows["16.4.0"][1]["xfer"] == "Read"
    assert rows["18.4.0"][1]["read req"] == metrics["l2.chan_rd"]["formula"]


def test_panel_descriptions_come_from_the_library():
    views, metrics, _ = metric_rows(ARCH)
    panel = views_to_panel_configs(views, metrics, {})[1100]
    ipc = metrics["compute.ipc"]
    assert panel["metrics_description"][ipc["name"]] == ipc["description"]


def test_view_schema_errors():
    """A broken view, a second view of its panel, a file name without a panel id, and
    a file for a built-in view."""
    source = "gfx942/1100_compute_pipeline.yaml"
    errors = view_schema_errors({
        source: BROKEN_VIEW,
        "gfx942/1100_copy.yaml": {"view": "Copy", "tables": []},
        "gfx942/compute.yaml": {"view": "Compute", "tables": []},
        "gfx942/0000_top_stats.yaml": {"view": "Top Stats", "tables": []},
    })
    assert errors == [
        "[gfx942/compute.yaml] the file name must be the panel id and the view name, "
        "e.g. 0300_memory_chart.yaml",
        "[gfx942/0000_top_stats.yaml] panel 0 is a built-in view",
        f"[{source}, gfx942/1100_copy.yaml] panel 1100 has more than one view file",
        f"[{source}] 'archs' is not a known field",
        f"[{source}] table 1: render 'chart' is not one of "
        "('metric_table', 'raw_csv_table', 'pc_sampling_table')",
        f"[{source}] table 2: row 0: needs one of 'metric' and 'metrics'",
        f"[{source}] table 2: row 1: needs a 'label' when it has 'metrics'",
        f"[{source}] table 2: row 2: 'values' names 'xfer', which is not a column",
        f"[{source}] table 2: row 3: 'peak' may only be 'empirical'",
        f"[{source}] table 2: row 4: 'unit' is not a known field",
    ]


def test_view_metric_errors():
    """The shipped gfx942 views, with a metric and a measured peak the library lacks,
    and a Min and Max over summed collectables without per-dispatch partners."""
    views = load_views(ARCH)
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), ARCH)
    collectables = _collectable_formulas(library)
    assert view_metric_errors(views, library.metrics, collectables) == []
    broken = copy.deepcopy(views)
    rows = dict(iter_view_rows(broken))
    rows["11.2.0"]["metric"] = "compute.made_up"
    rows["11.2.1"]["metric"] = "fabric.hbm_bw"
    rows["2.1.11"]["peak"] = "empirical"
    assert view_metric_errors(broken, library.metrics, collectables) == [
        "2.1.11 compute.branch_util has no 'peak_empirical'",
        "11.2.0 compute.made_up: not a metric of the arch's library",
        "11.2.1 fabric.hbm_bw: Min and Max read $hbm_read_bandwidth, a sum over all "
        "dispatches; add $hbm_read_bandwidth_per_dispatch",
        "11.2.1 fabric.hbm_bw: Min and Max read $hbm_write_bandwidth, a sum over all "
        "dispatches; add $hbm_write_bandwidth_per_dispatch",
    ]


def test_imports_only_profile_mode_dependencies(monkeypatch):
    """The views and the equation normalizer use only stdlib, project and vendored
    code. The re-import rebinds the utils package attributes, so they are restored too:
    later tests patch these modules by name."""
    for module in ("utils.analysis_views", "utils.equation_format"):
        name = module.split(".")[1]
        monkeypatch.setattr(utils, name, getattr(utils, name))
        monkeypatch.delitem(sys.modules, module, raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.analysis_views")
