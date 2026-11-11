# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Analysis views: the panels analyze shows, built from the metric library.

A view is one panel of one arch: its tables, and for each table row the metric it
shows, by metric library id. Formulas, units, peaks and descriptions come from the
library; a view row stores only how the panel shows the metric: a label other than
the metric's name, rounding, the measured peak instead of the theoretical one, and the
literal values of typed columns. views_to_panel_configs() turns an arch's views into
the panel configs that load_panel_configs() reads from analysis_configs/.

views/<arch>/ holds the views of one metric library arch, one file per panel, and
BUILT_IN_VIEWS the panels every arch shows from the workload's CSV files. A panel's
id comes from its file name (0300_memory_chart.yaml is panel 300), a table's id from its
position in the panel (301 is the first table), and a row's numeric id (3.1.0) from its
position in the table.

Analyze, its listings and the memory chart read their panel configs here
(arch_panel_configs()), and the -b panel aliases come from the views. Profile counters
still come from analysis_configs/, whose structure test_analysis_views_sync.py keeps
equal to the views', except for a replaced panel (REPLACED_PANEL_IDS), whose rows'
counters come from the views and the metric library.
Standard library and the vendored YAML only: profile mode imports it.
"""

import copy
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import config
from rocprof_compute_soc.metric_library import loader
from utils.equation_format import (
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
    format_equation,
    parse_equation,
)
from utils.logger import console_error, console_warning
from utils.utils_common import (
    INVALID_BLOCK_HINT,
    METRIC_ID_RE,
    canonical_config_arch,
    convert_metric_id_to_panel_info,
    format_table_id,
    load_yaml,
    mapping_field_errors,
    resolve_block_token,
)
from utils.utils_counter_defs import (
    PER_DISPATCH_SUFFIX,
    parse_counters_text,
    per_dispatch_partner,
    summed_collectables,
)

# Table kinds, as the panel configs' data source keys; a table without "render" is a
# metric table.
METRIC_TABLE = "metric_table"
RAW_CSV_TABLE = "raw_csv_table"
RENDERS = (METRIC_TABLE, RAW_CSV_TABLE, "pc_sampling_table")
# A row shows its formula in the first of these columns its table has.
MAIN_COLUMNS = ("avg", "value", "expr")
# Columns that aggregate the main formula over kernels, with the function they apply.
AGGREGATE_COLUMNS = {"min": "MIN", "max": "MAX", "std dev": "STD"}
# The functions of a formula that aggregate over kernels; a per-kernel formula drops
# them.
KERNEL_AGGREGATES = ("SUM", "AVG")
VIEW_FIELDS = ("view", "alias", "tables")
TABLE_FIELDS = (
    "render",
    "title",
    "header",
    "placeholder_range",
    "metrics",
    "source",
    "columnwise",
)
ROW_FIELDS = (
    "metric",
    "metrics",
    "label",
    "round",
    "peak",
    "pct_of_peak",
    "values",
)
# The value of a column a row has no value for, as the panel configs write it.
NO_VALUE = "None"
# A row's "peak" that shows its metric's measured peak instead of the theoretical one.
EMPIRICAL_PEAK = "empirical"
# A view file name: the panel id, then the view's name.
VIEW_FILE_RE = re.compile(r"^(\d\d00)_[a-z0-9_]+\.yaml$")
# The panel of the top kernels and dispatches, which --list-stats shows.
TOP_STATS_PANEL_ID = 0
# The views of every arch, by panel id: the top stats and the system info. They show
# the workload's CSV files, not metrics.
BUILT_IN_VIEWS: Dict[int, Dict[str, Any]] = {
    TOP_STATS_PANEL_ID: {
        "view": "Top Stats",
        "alias": "topstats",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "Top Kernels",
                "source": "pmc_kernel_top.csv",
            },
            {
                "render": RAW_CSV_TABLE,
                "title": "Dispatch List",
                "source": "pmc_dispatch_info.csv",
            },
        ],
    },
    100: {
        "view": "System Info",
        "alias": "sysinfo",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "System Info",
                "source": "sysinfo.csv",
                "columnwise": True,
            }
        ],
    },
}
# Panels whose view replaces the panel file's rows (the Speed of Light, AIPROFCOMP-821):
# until profile reads the views, profile takes their counters from the views.
REPLACED_PANEL_IDS = (200,)


def default_views_dir() -> Path:
    """Return the views tree shipped with rocprof-compute."""
    return Path(config.rocprof_compute_home) / "rocprof_compute_soc" / "views"


def read_view_files(arch: str, views_dir: Optional[Path] = None) -> Dict[Path, Any]:
    """Read arch's view files, keyed by path, in file name order.

    Args:
        arch: A GPU arch, e.g. "gfx942" or "gfx1151".
        views_dir: The views tree; the shipped one by default.

    Returns:
        Each file's document; none if the arch has no views.
    """
    arch_dir = Path(views_dir or default_views_dir()) / canonical_config_arch(arch)
    return {path: load_yaml(path) for path in sorted(arch_dir.glob("*.yaml"))}


def load_views(
    arch: str, views_dir: Optional[Path] = None
) -> "OrderedDict[int, Dict[str, Any]]":
    """Return arch's views, the built-in ones and its files', keyed by panel id, in
    panel id order."""
    views = copy.deepcopy(BUILT_IN_VIEWS)
    views.update(
        (view_panel_id(path.name), view)
        for path, view in read_view_files(arch, views_dir).items()
    )
    return OrderedDict(sorted(views.items()))


def view_panel_id(file_name: str) -> int:
    """Return the panel id of a view file name, e.g. 300 for 0300_memory_chart.yaml."""
    match = VIEW_FILE_RE.match(file_name)
    if match is None:
        raise ValueError(f"{file_name}: not a view file name")
    return int(match.group(1))


def views_to_panel_configs(
    views: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    collectables: Dict[str, str],
) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs of an arch's views, as load_panel_configs() does.

    Args:
        views: One arch's views by panel id (load_views()).
        metrics: The arch's metric library metrics (MetricLibrary.metrics).
        collectables: The arch's collectable formulas by name
            (MetricLibrary.collectable_formulas()).

    Returns:
        Each view's panel config, keyed by panel id, in panel id order.
    """
    partners = per_dispatch_partners(collectables)
    return OrderedDict(
        (panel_id, _panel_config(panel_id, views[panel_id], metrics, partners))
        for panel_id in sorted(views)
    )


def per_dispatch_partners(collectables: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Return each summed collectable's per-dispatch partner, or None if it has none.

    A summed collectable is one value for all dispatches, so a Min, Max or Std Dev
    over kernels reads its partner instead.
    """
    return {
        name: per_dispatch_partner(name, collectables)
        for name in summed_collectables(collectables)
    }


def has_views(arch: str) -> bool:
    """Return whether rocprof-compute ships views for arch."""
    return canonical_config_arch(arch) in loader.METRIC_LIBRARY_PARENT_ARCH


def arch_panel_configs(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the panel configs analyze shows for arch, from the shipped views.

    Args:
        arch: A GPU arch with views (has_views()).

    Returns:
        Each panel config, keyed by panel id, in panel id order.
    """
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    return views_to_panel_configs(
        load_views(arch), library.metrics, library.collectable_formulas()
    )


def panel_aliases(views: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
    """Return panel number -> -b alias, "" for none, of an arch's views."""
    return {
        str(panel_id // 100): view.get("alias") or ""
        for panel_id, view in views.items()
    }


def get_arch_panel_id_to_alias(
    arch: str, views_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Return panel number -> -b alias of arch's views, "" for none; {} for an arch
    without views."""
    if not has_views(arch):
        return {}
    return panel_aliases(load_views(arch, views_dir))


def get_arch_alias_to_panel_id(arch: str) -> Dict[str, str]:
    """Return -b alias -> panel number, for the panels that have an alias."""
    return {
        alias: pid for pid, alias in get_arch_panel_id_to_alias(arch).items() if alias
    }


def convert_filter_blocks_to_panel_ids(
    filter_blocks: List[str], arch: Optional[str] = None
) -> Set[int]:
    """Inverse of convert_metric_id_to_panel_info: map metric ids like
    "2" or "11.1" to the set of file_id integers (e.g. {200, 1100}).
    Metric library ids (e.g. "compute.salu_util") select the panels that show the
    metric, and other tokens are looked up as panel aliases (e.g. "lds",
    "roofline") for the given arch. A numeric id selects its own panel, also when the
    views leave its row out, so workloads saved by older versions keep their panels.
    """
    if arch:
        filter_blocks = [
            placement
            for bid in map(str, filter_blocks)
            for placement in (
                metric_placements(bid, arch)
                if loader.METRIC_LIBRARY_ID_RE.match(bid)
                else [bid]
            )
        ]
    alias_map: Dict[str, str] = (
        get_arch_alias_to_panel_id(arch)
        if arch and any(not METRIC_ID_RE.match(str(bid)) for bid in filter_blocks)
        else {}
    )
    resolved: Set[int] = set()
    for bid in filter_blocks:
        metric_id = resolve_block_token(str(bid), alias_map)
        if metric_id is None:
            console_error(f"Invalid --block value {str(bid)!r}.{INVALID_BLOCK_HINT}")
        resolved.add(int(convert_metric_id_to_panel_info(metric_id)[0]))
    return resolved


def metric_placements(metric_id: str, arch: str) -> List[str]:
    """Return the numeric ids of the rows that show a metric library id on arch.

    Args:
        metric_id: A metric library id, e.g. "compute.salu_util".
        arch: A GPU arch with views.

    Returns:
        Every row's "block.table.row" id, in view order; an unknown id is an error.
    """
    if not has_views(arch):
        console_error(f"Metric ids such as {metric_id!r} are not supported on {arch}.")
    placements = [
        position
        for position, row in iter_view_rows(load_views(arch))
        if metric_id in row_metric_ids(row)
    ]
    if not placements:
        console_error(
            f"Unknown metric id {metric_id!r} for {canonical_config_arch(arch)}."
            f"{INVALID_BLOCK_HINT}"
        )
    return placements


def expand_metric_ids(tokens: List[str], arch: str) -> List[str]:
    """Return -b tokens as the numeric ids of arch's views.

    A metric library id becomes the ids of the rows that show it. Numeric ids are the
    ids analyze shows and are returned unchanged, as are block aliases.
    """
    expanded: List[str] = []
    for token in tokens:
        if loader.METRIC_LIBRARY_ID_RE.match(token):
            expanded.extend(metric_placements(token, arch))
        else:
            expanded.append(token)
    return expanded


def row_ids_as_metric_ids(tokens: List[str], arch: str) -> List[str]:
    """Return -b tokens with each numeric row id replaced by the metric library ids its
    row shows, so a saved workload keeps its rows when later views renumber them. Other
    tokens, and row ids no view row has, are returned unchanged; each token once."""
    rows = _rows_by_position(arch)
    replaced: List[str] = []
    for token in tokens:
        for new in (
            row_metric_ids(rows[token])
            if _is_row_id(token) and token in rows
            else [token]
        ):
            if new not in replaced:
                replaced.append(new)
    return replaced


def warn_numeric_metric_ids(tokens: List[str], arch: str) -> None:
    """Warn that each numeric row id ("block.table.row") is deprecated, naming the
    metric library id its row shows."""
    rows = _rows_by_position(arch)
    for token in tokens:
        if not _is_row_id(token) or token not in rows:
            continue
        hint = " or ".join(repr(metric_id) for metric_id in row_metric_ids(rows[token]))
        console_warning(
            f"Numeric metric id {token!r} is deprecated. Use {hint} instead."
        )


def iter_view_rows(
    views: Dict[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield each metric table row of an arch's views with its numeric id."""
    for position, row, _ in _iter_rows_with_columns(views):
        yield position, row


def row_metric_ids(row: Dict[str, Any]) -> List[str]:
    """Return the metric ids a row shows: its metric, or each column's."""
    if "metric" in row:
        return [row["metric"]]
    return [metric_id for metric_id in row.get("metrics", {}).values() if metric_id]


def replaced_panel_metric_ids(token: str, arch: Optional[str]) -> Optional[List[str]]:
    """Return the metric ids a -b block, table or row id or a block alias selects in a
    replaced panel (REPLACED_PANEL_IDS), or None if it selects another panel.

    A row id is the id analyze shows.
    """
    if not arch or not has_views(arch):
        return None
    block_id = resolve_block_token(token, get_arch_alias_to_panel_id(arch))
    if block_id is None:
        return None
    file_id = int(convert_metric_id_to_panel_info(block_id)[0])
    if file_id not in REPLACED_PANEL_IDS:
        return None
    if _is_row_id(block_id):
        row = _rows_by_position(arch).get(block_id)
        return row_metric_ids(row) if row else []
    return [
        metric_id
        for position, row in iter_view_rows(load_views(arch))
        if f"{position}.".startswith(f"{block_id}.")
        for metric_id in row_metric_ids(row)
    ]


def row_label(row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]) -> str:
    """Return the name a row shows: its label, or its metric's name."""
    if "label" in row:
        return row["label"]
    return metrics[row["metric"]]["name"]


def view_schema_errors(documents: Dict[str, Any]) -> List[str]:
    """Return what is wrong with the shape of one arch's view documents, keyed by file
    path.

    Checks the file names and that each panel id has one file, the fields each level
    may have, render types, that a row names its metric or its columns' metrics, and
    that a row with columns' metrics has a label.
    """
    errors = _file_name_errors(list(documents))
    for source, document in documents.items():
        errors.extend(f"[{source}] {error}" for error in _view_shape_errors(document))
    return errors


def view_metric_errors(
    views: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    collectables: Dict[str, str],
) -> List[str]:
    """Return what in an arch's views the arch's metric library cannot show.

    Checks that every metric id exists, a measured peak exists, and a row whose table
    shows Min, Max or Std Dev reads no summed collectable without a per-dispatch
    partner.
    """
    unpartnered = {
        name
        for name, partner in per_dispatch_partners(collectables).items()
        if partner is None
    }
    errors = []
    for position, row, columns in _iter_rows_with_columns(views):
        row_errors = _row_metric_errors(row, metrics)
        if not row_errors and any(column in AGGREGATE_COLUMNS for column in columns):
            row_errors = _summed_collectable_errors(row, metrics, unpartnered)
        errors.extend(f"{position} {error}" for error in row_errors)
    return errors


# ---------------------------------------------------------------------------
# Metric ids
# ---------------------------------------------------------------------------


def _is_row_id(token: str) -> bool:
    return bool(METRIC_ID_RE.match(token)) and token.count(".") == 2


def _rows_by_position(arch: str) -> Dict[str, Dict[str, Any]]:
    """Return view row id (the id analyze shows) -> row of arch's views."""
    if not has_views(arch):
        return {}
    return dict(iter_view_rows(load_views(arch)))


# ---------------------------------------------------------------------------
# Panel configs
# ---------------------------------------------------------------------------


def _render(table: Dict[str, Any]) -> str:
    return table.get("render", METRIC_TABLE)


def _iter_rows_with_columns(
    views: Dict[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Dict[str, Any], List[str]]]:
    """Yield each metric table row of an arch's views with its numeric id and its
    table's header columns."""
    for panel_id, view in views.items():
        for table_id, table in _numbered_tables(panel_id, view):
            if _render(table) != METRIC_TABLE:
                continue
            for index, row in enumerate(table["metrics"]):
                yield f"{format_table_id(table_id)}.{index}", row, list(table["header"])


def _numbered_tables(
    panel_id: int, view: Dict[str, Any]
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield each table of a view with its table id: the panel id plus its position."""
    for index, table in enumerate(view["tables"]):
        yield panel_id + index + 1, table


def _panel_config(
    panel_id: int,
    view: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    return {
        "id": panel_id,
        "title": view["view"],
        "data source": [
            {_render(table): _table_config(table_id, table, metrics, partners)}
            for table_id, table in _numbered_tables(panel_id, view)
        ],
        "metrics_description": _panel_descriptions(view, metrics),
    }


def _table_config(
    table_id: int,
    table: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    config_keys = ("title", "header", "source", "columnwise")
    table_config = {"id": table_id}
    table_config.update({key: table[key] for key in config_keys if key in table})
    if "metrics" not in table:
        return table_config
    columns = [column for column in table["header"] if column != "metric"]
    rows = {
        row_label(row, metrics): _row_values(row, columns, metrics, partners)
        for row in table["metrics"]
    }
    if "placeholder_range" in table:
        rows["placeholder_range"] = table["placeholder_range"]
    table_config["metric"] = rows
    return table_config


def _row_values(
    row: Dict[str, Any],
    columns: List[str],
    metrics: Dict[str, Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Return a row's value in each column, in column order."""
    metric = metrics.get(row.get("metric"))
    values = {}
    for column in columns:
        if column in row.get("values", {}):
            values[column] = row["values"][column]
        elif column in row.get("metrics", {}):
            metric_id = row["metrics"][column]
            values[column] = metrics[metric_id]["formula"] if metric_id else NO_VALUE
        elif column in MAIN_COLUMNS:
            values[column] = _main_formula(row, metric)
        elif column in AGGREGATE_COLUMNS:
            values[column] = _aggregate(column, metric, partners)
        elif column == "unit":
            values[column] = metric["unit"] if metric else NO_VALUE
        elif column == "peak":
            values[column] = _peak(row, metric)
        elif column == "pct_of_peak":
            values[column] = row.get("pct_of_peak", True)
        else:
            raise ValueError(f"row {row} has no value for column {column!r}")
    return values


def _main_formula(row: Dict[str, Any], metric: Optional[Dict[str, Any]]) -> str:
    """Return the formula a row shows in its main column, rounded if it asks."""
    if metric is None:
        return NO_VALUE
    if "round" in row:
        return f"ROUND({metric['formula']}, {row['round']})"
    return metric["formula"]


def _aggregate(
    column: str,
    metric: Optional[Dict[str, Any]],
    partners: Dict[str, Optional[str]],
) -> str:
    """Return a column's aggregate over kernels of the metric's formula, per kernel."""
    if metric is None:
        return NO_VALUE
    per_kernel = _per_kernel(parse_equation(metric["formula"]), partners)
    return format_equation(FunctionCallNode(AGGREGATE_COLUMNS[column], [per_kernel]))


def _per_kernel(node: ASTNode, partners: Dict[str, Optional[str]]) -> ASTNode:
    """Return a formula tree per kernel: each SUM(x) and AVG(x) becomes x, and each
    summed collectable with a per-dispatch partner becomes the partner."""
    if isinstance(node, FunctionCallNode):
        args = [_per_kernel(arg, partners) for arg in node.args]
        if node.name in KERNEL_AGGREGATES and len(args) == 1:
            return args[0]
        return FunctionCallNode(node.name, args)
    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.op, _per_kernel(node.left, partners), _per_kernel(node.right, partners)
        )
    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, _per_kernel(node.operand, partners))
    if isinstance(node, VariableNode):
        partner = partners.get(node.name.lstrip("$"))
        return VariableNode(f"${partner}") if partner else node
    return node


def _peak(row: Dict[str, Any], metric: Optional[Dict[str, Any]]) -> Union[int, str]:
    """Return the peak a row shows: its metric's theoretical or measured peak."""
    field = "peak_empirical" if row.get("peak") == EMPIRICAL_PEAK else "peak"
    if metric is None or field not in metric:
        return NO_VALUE
    return metric[field]


def _panel_descriptions(
    view: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """Return label -> description; the first row with a label describes it."""
    descriptions: Dict[str, str] = {}
    for table in view["tables"]:
        for row in table.get("metrics", []):
            label = row_label(row, metrics)
            metric = metrics.get(row.get("metric"))
            if label not in descriptions and metric is not None:
                descriptions[label] = metric["description"]
    return descriptions


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _file_name_errors(paths: List[str]) -> List[str]:
    """Report a view file name without a panel id, and a panel id with two files."""
    errors = []
    paths_by_panel: Dict[int, List[str]] = {}
    for path in paths:
        name = Path(path).name
        if not VIEW_FILE_RE.match(name):
            errors.append(
                f"[{path}] the file name must be the panel id and the view name, "
                "e.g. 0300_memory_chart.yaml"
            )
            continue
        panel_id = view_panel_id(name)
        if panel_id in BUILT_IN_VIEWS:
            errors.append(f"[{path}] panel {panel_id} is a built-in view")
        paths_by_panel.setdefault(panel_id, []).append(path)
    errors.extend(
        f"[{', '.join(panel_paths)}] panel {panel_id} has more than one view file"
        for panel_id, panel_paths in sorted(paths_by_panel.items())
        if len(panel_paths) > 1
    )
    return errors


def _view_shape_errors(document: object) -> List[str]:
    if not isinstance(document, dict):
        return ["a view must be a mapping"]
    errors = mapping_field_errors(document, ("view", "tables"), VIEW_FIELDS)
    tables = document.get("tables")
    if not isinstance(tables, list):
        return [*errors, "'tables' must be a list"]
    for position, table in enumerate(tables, start=1):
        errors.extend(
            f"table {position}: {error}" for error in _table_shape_errors(table)
        )
    return errors


def _table_shape_errors(table: object) -> List[str]:
    if not isinstance(table, dict):
        return ["a table must be a mapping"]
    errors = mapping_field_errors(table, ("title",), TABLE_FIELDS)
    render = _render(table)
    if render not in RENDERS:
        errors.append(f"render {render!r} is not one of {RENDERS}")
    if render == METRIC_TABLE:
        errors.extend(_metric_table_shape_errors(table))
    elif "metrics" in table or "header" in table:
        errors.append(f"a {render} has no header or metrics")
    return errors


def _metric_table_shape_errors(table: Dict[str, Any]) -> List[str]:
    header = table.get("header")
    if not isinstance(header, dict) or next(iter(header), None) != "metric":
        return ["'header' must be a mapping whose first column is 'metric'"]
    if not isinstance(table.get("metrics"), list):
        return ["'metrics' must be a list"]
    columns = [column for column in header if column != "metric"]
    errors = []
    for index, row in enumerate(table["metrics"]):
        errors.extend(
            f"row {index}: {error}" for error in _row_shape_errors(row, columns)
        )
    return errors


def _row_shape_errors(row: object, columns: List[str]) -> List[str]:
    if not isinstance(row, dict):
        return ["a row must be a mapping"]
    errors = mapping_field_errors(row, (), ROW_FIELDS)
    if ("metric" in row) == ("metrics" in row):
        errors.append("needs one of 'metric' and 'metrics'")
    elif "metrics" in row and "label" not in row:
        errors.append("needs a 'label' when it has 'metrics'")
    if row.get("peak", EMPIRICAL_PEAK) != EMPIRICAL_PEAK:
        errors.append(f"'peak' may only be {EMPIRICAL_PEAK!r}")
    errors.extend(
        f"'{field}' names {column!r}, which is not a column"
        for field in ("metrics", "values")
        for column in row.get(field, {})
        if column not in columns
    )
    return errors


def _row_metric_errors(
    row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    errors = [
        f"{metric_id}: not a metric of the arch's library"
        for metric_id in row_metric_ids(row)
        if metric_id not in metrics
    ]
    metric = metrics.get(row.get("metric"))
    if not errors and row.get("peak") == EMPIRICAL_PEAK:
        if metric is None or "peak_empirical" not in metric:
            errors.append(f"{row.get('metric')} has no 'peak_empirical'")
    return errors


def _summed_collectable_errors(
    row: Dict[str, Any], metrics: Dict[str, Dict[str, Any]], unpartnered: Set[str]
) -> List[str]:
    """Report the summed collectables without a partner that a row's Min, Max and
    Std Dev would read."""
    if "metric" not in row:
        return []
    _, names = parse_counters_text(metrics[row["metric"]]["formula"])
    return [
        f"{row['metric']}: Min and Max read ${name}, a sum over all dispatches; "
        f"add ${name}{PER_DISPATCH_SUFFIX}"
        for name in sorted(names & unpartnered)
    ]
