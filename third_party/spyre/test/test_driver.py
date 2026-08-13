#!/usr/bin/env python3
"""Tests for the Spyre driver skeleton — ``SpyreUtils`` / ``SpyreLauncher``.

``utils`` and ``launcher_cls`` are an *undeclared* convention: they do not
appear on ``DriverBase``, but ``compiler.py`` reaches for them by name at four
sites, and ``JITFunction.run`` calls ``get_current_device()`` /
``get_current_stream()`` unconditionally. Before this skeleton existed,
``kernel[grid](...)`` died with ``AttributeError: launcher_cls``.

No device is needed. The end-to-end test drives ``kernel[grid](...)`` with a
duck-typed tensor — ``specialize.cc`` falls back to its tensor handler for any
object carrying ``data_ptr()`` (``python/src/specialize.cc:562``) — so the whole
compile-and-load path runs and the launch stops *inside*
``SpyreLauncher.__call__``. That is the whole point of the issue: the failure
must be ours and about the launch, not Triton's plumbing about a missing member.
"""

import hashlib
import io
import zipfile

import pytest
import triton
import triton.language as tl
from triton import knobs
from triton.backends.driver import DriverBase

from backend.driver import SpyreDriver, SpyreLauncher, SpyreUtils

from test_spyrecode_stage import _REQUIRED_FIXES, resolve_dbo_opt


# ---------------------------------------------------------------------------
# A duck-typed tensor. specialize.cc dispatches on type() first and falls back
# to "has a data_ptr attribute", so this is enough to reach the compile path
# without torch installed. .dtype must str() to something
# triton._utils.canonicalize_dtype knows.
# ---------------------------------------------------------------------------

class _FakeDType:

    def __str__(self):
        return "torch.float16"

    def __hash__(self):
        return hash("float16")


class FakeTensor:
    dtype = _FakeDType()

    def __init__(self, address):
        self._address = address

    def data_ptr(self):
        return self._address


@triton.jit
def _add_kernel_1core(x_ptr, y_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1],
                                        block_shape=[BLOCK_M, BLOCK_N])
    out_desc.store([0, 0], x_desc.load([0, 0]) + y_desc.load([0, 0]))


def _zip_bytes(entries):
    """Deterministic ZIP, matching what ``_make_spyrecode`` emits."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return buffer.getvalue()


_ARTIFACT = {
    "spyrecode.json": b'{"init_bin_file": "init_binary.bin"}',
    "init_binary.bin": b"\x00\x01\x02\x03",
    "debug/dfir.mlir": b"module {}\n",
}


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    """Point knobs.cache.dir at a temp dir so load_binary is observable."""
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# The members Triton reaches for by name
# ---------------------------------------------------------------------------

class TestDriverSurface:

    def test_utils_and_launcher_cls_assigned_in_init(self):
        driver = SpyreDriver()
        assert isinstance(driver.utils, SpyreUtils)
        assert driver.launcher_cls is SpyreLauncher

    def test_device_and_stream(self):
        driver = SpyreDriver()
        assert driver.get_current_device() == 0
        assert driver.get_current_stream(0) == 0

    def test_device_properties_has_exactly_one_key(self):
        # The other six CUDA keys are read only by python/triton/testing.py;
        # inventing them would make it print nonsense instead of failing.
        assert SpyreUtils().get_device_properties(0) == {"max_shared_mem": 0}

    def test_utils_is_not_a_singleton(self):
        # CudaUtils is one only to avoid recompiling driver.c; nothing here is
        # compiled, so a plain instance is correct.
        assert SpyreUtils() is not SpyreUtils()

    @pytest.mark.parametrize("member", [
        "get_device_interface",
        "get_empty_cache_for_benchmark",
        "clear_cache",
        "set_current_device",
    ])
    def test_members_stay_absent(self, member):
        # Absent, not stubbed: a fabricated value is worse than an
        # AttributeError, because it fails somewhere else and later.
        assert not hasattr(SpyreDriver(), member)

    def test_benchmarker_and_profile_scratch_still_raise(self):
        driver = SpyreDriver()
        with pytest.raises(NotImplementedError):
            driver.get_benchmarker()
        with pytest.raises(NotImplementedError):
            driver.allocate_default_profile_scratch(0, 0, 0)

    def test_base_class_is_driver_base_not_gpu_driver(self):
        # GPUDriver.__init__ hard-imports torch.cuda, which is not available and
        # would not mean anything here.
        from triton.backends.driver import GPUDriver
        assert issubclass(SpyreDriver, DriverBase)
        assert not issubclass(SpyreDriver, GPUDriver)


# ---------------------------------------------------------------------------
# load_binary / unload_module
# ---------------------------------------------------------------------------

class TestLoadBinary:

    def test_returns_the_five_tuple(self, cache_dir):
        artifact = _zip_bytes(_ARTIFACT)
        module, function, n_regs, n_spills, n_max_threads = \
            SpyreUtils().load_binary("", artifact, 0, 0)
        # Both handles are the same directory: the program *is* the directory,
        # so there is no loaded-module / entry-point split to mirror.
        assert module == function
        assert (n_regs, n_spills) == (0, 0)
        # Must be >= 1 so num_warps * warp_size (1) does not exceed it.
        assert n_max_threads >= 1

    def test_unpacks_flat_with_debug_beside(self, cache_dir):
        module, _, _, _, _ = SpyreUtils().load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        root = cache_dir / SpyreUtils.MODULE_CACHE / hashlib.sha256(
            _zip_bytes(_ARTIFACT)).hexdigest()
        assert str(root) == module
        # prepare_kernel opens <dir>/spyrecode.json and the init_bin_file it
        # names, both by name with no directory scan, so debug/ beside them is
        # invisible to it.
        assert (root / "spyrecode.json").is_file()
        assert (root / "init_binary.bin").is_file()
        assert (root / "debug" / "dfir.mlir").is_file()

    def test_keyed_on_the_artifact_digest(self, cache_dir):
        artifact = _zip_bytes(_ARTIFACT)
        module, _, _, _, _ = SpyreUtils().load_binary("", artifact, 0, 0)
        assert hashlib.sha256(artifact).hexdigest() == module.rsplit("/", 1)[-1]

    def test_name_is_not_part_of_the_key(self, cache_dir):
        # metadata["name"] is "" (issue #104), so keying on it would collide
        # every kernel into one directory.
        artifact = _zip_bytes(_ARTIFACT)
        first, _, _, _, _ = SpyreUtils().load_binary("", artifact, 0, 0)
        second, _, _, _, _ = SpyreUtils().load_binary("something_else", artifact, 0, 0)
        assert first == second

    def test_same_artifact_reuses_its_directory(self, cache_dir):
        utils = SpyreUtils()
        artifact = _zip_bytes(_ARTIFACT)
        first, _, _, _, _ = utils.load_binary("", artifact, 0, 0)
        # A marker survives the second load, proving nothing was re-extracted.
        marker = cache_dir / SpyreUtils.MODULE_CACHE / first.rsplit("/", 1)[-1] / "marker"
        marker.write_text("kept")
        second, _, _, _, _ = utils.load_binary("", artifact, 0, 0)
        assert second == first
        assert marker.read_text() == "kept"

    def test_different_artifacts_do_not_collide(self, cache_dir):
        utils = SpyreUtils()
        other = dict(_ARTIFACT, init_binary=b"different")
        other["init_binary.bin"] = b"\x04\x05\x06\x07"
        first, _, _, _, _ = utils.load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        second, _, _, _, _ = utils.load_binary("", _zip_bytes(other), 0, 0)
        assert first != second
        assert (cache_dir / SpyreUtils.MODULE_CACHE / first.rsplit("/", 1)[-1]).is_dir()
        assert (cache_dir / SpyreUtils.MODULE_CACHE / second.rsplit("/", 1)[-1]).is_dir()

    def test_no_partial_directory_is_left_behind(self, cache_dir):
        SpyreUtils().load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        staging = [p.name for p in (cache_dir / SpyreUtils.MODULE_CACHE).iterdir()
                   if p.name.startswith("tmp.")]
        assert staging == []

    def test_unload_module_keeps_the_directory(self, cache_dir):
        utils = SpyreUtils()
        module, _, _, _, _ = utils.load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        utils.unload_module(module)
        # Deliberate: content-addressed, so a cache rather than a leak, and
        # debug/dfir.mlir is the only on-disk record of what ran.
        from pathlib import Path
        assert (Path(module) / "spyrecode.json").is_file()


# ---------------------------------------------------------------------------
# SpyreLauncher
# ---------------------------------------------------------------------------

class TestSpyreLauncher:

    def test_init_touches_no_option_fields(self):
        # CudaLauncher reads global_scratch_size / profile_scratch_* /
        # launch_cooperative_grid / launch_pdl as hard attributes. SpyreOptions
        # has none of them, so reading any would be an immediate AttributeError.
        class Bare:

            def __getattr__(self, name):
                raise AssertionError(f"launcher read metadata.{name}")

        SpyreLauncher(object(), Bare())

    def test_call_raises_not_implemented(self):
        launcher = SpyreLauncher(object(), object())
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            launcher(1, 1, 1, 0, "/some/dir", (), None, None, None, "a", "b")

    def test_call_message_names_the_program(self):
        launcher = SpyreLauncher(object(), object())
        with pytest.raises(NotImplementedError) as excinfo:
            launcher(4, 1, 1, 0, "/some/dir", (), None, None, None, "a")
        message = str(excinfo.value)
        assert "/some/dir" in message
        # The grid is a tile count, not an axis partition (issue #100).
        assert "tile count" in message


# ---------------------------------------------------------------------------
# End to end: kernel[grid](...) reaches the launcher
# ---------------------------------------------------------------------------

class TestSubscriptLaunch:

    def test_reaches_the_launcher_not_an_attribute_error(self):
        if resolve_dbo_opt(required=False) is None:
            pytest.skip("dbo-opt not resolvable")
        tensors = [FakeTensor(slot * (1 << 34)) for slot in range(3)]
        with pytest.raises(NotImplementedError) as excinfo:
            _add_kernel_1core[(1, )](*tensors, M=1, N=64, BLOCK_M=1, BLOCK_N=64,
                                     required_fixes=dict(_REQUIRED_FIXES))
        # The program path in the message is the unpacked spyreCodeDir, so
        # reaching here proves compile + load + launcher dispatch all ran.
        from pathlib import Path
        program = Path(str(excinfo.value).split("program at ")[1]
                       .split(". The launch")[0])
        assert (program / "spyrecode.json").is_file()
        assert (program / "init_binary.bin").is_file()
        assert (program / "debug").is_dir()
