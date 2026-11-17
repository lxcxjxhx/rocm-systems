"""GPU-free checks for the distributor build inputs read by setup.py and
CMakeLists.txt: ROCSHMEM4PY_VERSION_SUFFIX,
ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS and ROCSHMEM4PY_INSTALL_RPATH.

These live outside tests/ because tests/conftest.py initializes rocSHMEM for
the whole session. Run from python/rocshmem:

    python -m pytest build_tests -v

The metadata tests need only setuptools>=61. The wheel-build tests also need
nanobind, cmake, readelf and a rocSHMEM install found the way setup.py finds
one (CMAKE_PREFIX_PATH, ROCSHMEM_HOME, ROCM_PATH, /opt/rocm), and are skipped
otherwise. They build for a single GPU target to stay fast.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("setuptools", minversion="61")

PROJECT_DIR = Path(__file__).resolve().parents[1]
BUILD_INPUTS = (
    "ROCSHMEM4PY_VERSION_SUFFIX",
    "ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS",
    "ROCSHMEM4PY_INSTALL_RPATH",
)
# TheRock's dev-build form; pip normalizes the '-' to '.'.
VERSION_SUFFIX = "+devrocm10.1.0.dev0-0123abcdef"
NORMALIZED_LOCAL = "devrocm10.1.0.dev0.0123abcdef"
REQUIREMENT = "rocm-sdk-core==10.1.0.dev0+0123abcdef"
RPATH = ":".join(
    [
        "$ORIGIN/_rocm_sdk_core/lib",
        "$ORIGIN/_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "$ORIGIN/_rocm_sdk_core/lib/llvm/lib",
    ]
)


def _base_version():
    text = (PROJECT_DIR / "setup.py").read_text()
    return re.search(r'^BASE_VERSION = "([^"]+)"', text, re.MULTILINE).group(1)


def _copy_project(tmp_path):
    """Copy the project so setuptools' egg-info/build output stays out of the
    source tree."""
    dst = tmp_path / "src"
    shutil.copytree(
        PROJECT_DIR,
        dst,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", "*.so", "tests", "build_tests"
        ),
    )
    return dst


def _env(**inputs):
    env = {k: v for k, v in os.environ.items() if k not in BUILD_INPUTS}
    env.update(inputs)
    return env


def _headers(metadata, name):
    header = metadata.split("\n\n", 1)[0]
    prefix = name + ":"
    return [
        line[len(prefix) :].strip()
        for line in header.splitlines()
        if line.startswith(prefix)
    ]


def _prepare_metadata(tmp_path, **inputs):
    """Run the PEP 517 metadata hook, which evaluates setup.py without
    building the extension."""
    src = _copy_project(tmp_path)
    out = tmp_path / "metadata"
    out.mkdir()
    script = "import sys\nfrom setuptools import build_meta\nbuild_meta.prepare_metadata_for_build_wheel(sys.argv[1])\n"
    proc = subprocess.run(
        [sys.executable, "-c", script, str(out)],
        cwd=src,
        env=_env(**inputs),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, out


def _metadata(tmp_path, **inputs):
    proc, out = _prepare_metadata(tmp_path, **inputs)
    assert proc.returncode == 0, proc.stderr
    (metadata_path,) = out.glob("*.dist-info/METADATA")
    return metadata_path.read_text()


def test_default_metadata_is_unchanged(tmp_path):
    metadata = _metadata(tmp_path)
    (version,) = _headers(metadata, "Version")
    public, _, local = version.partition("+")
    assert public == _base_version()
    assert local == "" or local.startswith("rocshmem")
    assert all("extra ==" in req for req in _headers(metadata, "Requires-Dist"))


@pytest.mark.parametrize(
    "suffix, local",
    [
        ("+rocm7.10.0", "rocm7.10.0"),
        ("+rocm7.10.0a20251124", "rocm7.10.0a20251124"),
        (VERSION_SUFFIX, NORMALIZED_LOCAL),
        ("+ROCm_7", "rocm.7"),
    ],
)
def test_version_suffix_replaces_local_segment(tmp_path, suffix, local):
    metadata = _metadata(tmp_path, ROCSHMEM4PY_VERSION_SUFFIX=suffix)
    assert _headers(metadata, "Version") == [f"{_base_version()}+{local}"]


@pytest.mark.parametrize(
    "suffix",
    [
        "rocm7.10.0",
        "+",
        "++rocm7",
        "+rocm 7",
        "+rocm!7",
        "+rocm..7",
        "+-rocm7",
        "+rocm7-",
        "+rocm7+abc",
    ],
)
def test_version_suffix_rejects_malformed_value(tmp_path, suffix):
    proc, _ = _prepare_metadata(tmp_path, ROCSHMEM4PY_VERSION_SUFFIX=suffix)
    assert proc.returncode != 0
    assert "ROCSHMEM4PY_VERSION_SUFFIX" in proc.stderr


def test_extra_install_requirements_added(tmp_path):
    metadata = _metadata(
        tmp_path,
        ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS=f"{REQUIREMENT}| packaging>=20 |",
    )
    requires = _headers(metadata, "Requires-Dist")
    assert REQUIREMENT in requires
    assert "packaging>=20" in requires
    assert any("extra ==" in req for req in requires), "optional extras must be kept"


@pytest.mark.parametrize(
    "requirements",
    ["rocm-sdk-core ==", "not a requirement!!", f"{REQUIREMENT}\npackaging"],
)
def test_extra_install_requirements_rejects_malformed_entry(tmp_path, requirements):
    proc, _ = _prepare_metadata(
        tmp_path, ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS=requirements
    )
    assert proc.returncode != 0
    assert "ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS" in proc.stderr


def _rocshmem_prefix():
    """First rocSHMEM install in setup.py's search order, with its targets."""
    prefixes = os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)
    prefixes += [
        os.environ.get("ROCSHMEM_HOME", ""),
        os.environ.get("ROCM_PATH", ""),
        "/opt/rocm",
    ]
    for prefix in filter(None, prefixes):
        header = Path(prefix) / "include" / "rocshmem" / "rocshmem_config.h"
        if header.is_file():
            match = re.search(
                r'#\s*define\s+ROCSHMEM_OFFLOAD_TARGETS\s+"([^"]+)"', header.read_text()
            )
            return (Path(prefix), match.group(1).split()) if match else (None, [])
    return None, []


def _wheel_build_skip_reason():
    try:
        import nanobind  # noqa: F401
    except ImportError:
        return "nanobind is not installed"
    for tool in ("cmake", "readelf"):
        if shutil.which(tool) is None:
            return f"{tool} is not on PATH"
    if _rocshmem_prefix()[0] is None:
        return "no rocSHMEM install with ROCSHMEM_OFFLOAD_TARGETS found"
    return None


_WHEEL_BUILD_SKIP_REASON = _wheel_build_skip_reason()
requires_wheel_build = pytest.mark.skipif(
    _WHEEL_BUILD_SKIP_REASON is not None, reason=str(_WHEEL_BUILD_SKIP_REASON)
)


def _build_wheel(tmp_path, src=None, name="wheel", **inputs):
    """Build a wheel from a fresh project copy, or from src to reuse its
    in-tree build directory as repeated `pip wheel .` runs do."""
    _, arches = _rocshmem_prefix()
    src = src or _copy_project(tmp_path)
    out = tmp_path / name
    env = _env(ROCSHMEM_GPU_TARGETS=arches[0], **inputs)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(out),
            str(src),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    (wheel,) = out.glob("rocshmem4py-*.whl")
    extracted = tmp_path / f"{name}-extracted"
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(extracted)
    (metadata_path,) = extracted.glob("*.dist-info/METADATA")
    (extension,) = extracted.glob("_rocshmem4py*.so")
    dynamic = subprocess.run(
        ["readelf", "-dW", str(extension)], capture_output=True, text=True, check=True
    ).stdout
    return wheel, metadata_path.read_text(), dynamic


def _dynamic_entry(dynamic, tag):
    match = re.search(rf"\({tag}\)\s+Library (?:rpath|runpath): \[([^\]]*)\]", dynamic)
    return match.group(1) if match else None


@requires_wheel_build
def test_wheel_applies_distributor_inputs(tmp_path):
    wheel, metadata, dynamic = _build_wheel(
        tmp_path,
        ROCSHMEM4PY_VERSION_SUFFIX=VERSION_SUFFIX,
        ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS=REQUIREMENT,
        ROCSHMEM4PY_INSTALL_RPATH=RPATH,
    )
    version = f"{_base_version()}+{NORMALIZED_LOCAL}"
    assert wheel.name.startswith(f"rocshmem4py-{version}-")
    assert _headers(metadata, "Version") == [version]
    assert REQUIREMENT in _headers(metadata, "Requires-Dist")
    # Exactly the requested entries, as DT_RPATH: no build-tree or build-host
    # ROCm directory may leak into a relocatable wheel.
    assert _dynamic_entry(dynamic, "RPATH") == RPATH
    assert _dynamic_entry(dynamic, "RUNPATH") is None


def _assert_source_build_defaults(metadata, dynamic):
    (version,) = _headers(metadata, "Version")
    assert version.startswith(f"{_base_version()}+rocshmem")
    assert all("extra ==" in req for req in _headers(metadata, "Requires-Dist"))
    # Source builds keep CMake's absolute build-tree RUNPATH.
    assert _dynamic_entry(dynamic, "RPATH") is None, dynamic
    runpath = _dynamic_entry(dynamic, "RUNPATH")
    assert runpath, dynamic
    assert all(os.path.isabs(entry) for entry in runpath.split(":") if entry), runpath


@requires_wheel_build
def test_wheel_without_inputs_has_source_build_defaults(tmp_path):
    _, metadata, dynamic = _build_wheel(tmp_path)
    _assert_source_build_defaults(metadata, dynamic)


@requires_wheel_build
def test_rebuild_without_inputs_drops_cached_rpath(tmp_path):
    src = _copy_project(tmp_path)
    _, _, dynamic = _build_wheel(
        tmp_path, src=src, name="first", ROCSHMEM4PY_INSTALL_RPATH=RPATH
    )
    assert _dynamic_entry(dynamic, "RPATH") == RPATH
    _, metadata, dynamic = _build_wheel(tmp_path, src=src, name="second")
    _assert_source_build_defaults(metadata, dynamic)


@requires_wheel_build
def test_install_rpath_fails_configure_without_driver_flag_support(tmp_path):
    """A compiler that cannot drop its own <rocm>/lib RPATH entry must fail the
    configure instead of producing a non-relocatable wheel."""
    import nanobind

    prefix, arches = _rocshmem_prefix()
    proc = subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_DIR),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_PREFIX_PATH={prefix};{nanobind.cmake_dir()}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DROCSHMEM_GPU_TARGETS={arches[0]}",
            f"-DROCSHMEM4PY_INSTALL_RPATH={RPATH}",
            # Pre-seeding the cached check result stands in for such a compiler.
            "-DROCSHMEM4PY_HAS_NO_RTLIB_ADD_RPATH=OFF",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0, proc.stdout[-4000:]
    assert "-fno-rtlib-add-rpath" in proc.stderr, proc.stderr[-4000:]
