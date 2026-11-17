# Metric configuration tools

The metric library (`src/rocprof_compute_soc/metric_library/`) defines the metrics, and the
analysis views (`src/rocprof_compute_soc/views/`) say which metrics each panel shows. The
tools here check them and generate the files made from them.

| File | What it does |
|---|---|
| `check_metric_config.py` | The `metric-config-check` pre-commit hook: runs the metric library, view and membw tree spec checks, then regenerates the metric lock and the metric description files. With `--fix` it writes them; it fails until their diff is staged. |
| `metric_lock.py` | Writes `metric_lock/<arch>.yaml`: each metric's formula, unit, peaks and the counters profile collects for it, after inheritance. |
| `metric_lock/` | The generated lock, one file per metric library arch. Do not edit it. |
| `metric_description_manager.py` | Writes the per-arch description files (`tools/per_arch_metric_definitions/`) and the docs files (`docs/data/metrics/`) from the views and the library. |
| `utils_ruamel.py` | Round-trip YAML helpers for `metric_description_manager.py`. |

After editing the library or the views, run:

    python tools/config_management/check_metric_config.py --fix

and stage the files it lists. The diff of `metric_lock/` shows what the edit changes on
each arch: a changed formula, unit or peak, and counters added to or removed from profiling.

To edit an `rst` text by hand, edit the per-arch description file; the manager keeps an
`rst` that differs from `plain`.
