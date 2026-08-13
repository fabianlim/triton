#!/usr/bin/env python3
"""Tests for the ``spyrecode`` compile stage — KTIR to a loadable Spyre binary.

Before this stage the last thing a Triton compile produced was KTIR *text*, so
there was no artifact for a driver to load. ``SpyreBackend._make_spyrecode``
bakes the pointer base addresses into the KTIR, runs ``dbo-opt``, and returns
the resulting spyreCodeDir as ZIP bytes.

No device is needed for any of this: the backend is driven directly via
``ASTSource`` + ``triton.compiler.compile``, the same way the rest of this suite
drives it (see ``utils.compile_to_ttir`` / ``utils.make_ktir_mod``).

The address-policy and ceiling checks need neither ``dbo-opt`` nor a build, so
they run unconditionally; the compile tests skip when ``dbo-opt`` cannot be
resolved (see ``knobs.spyre.dbo_opt``).
"""

import hashlib
import io
import zipfile

import pytest
import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile

from backend.compiler import (
    SpyreBackend,
    SpyreOptions,
    base_addresses_for,
    resolve_dbo_opt,
)

_TARGET = GPUTarget(backend="spyre", arch=1, warp_size=1)

# Elementwise on purpose. A reduction kernel would additionally need the
# DropReductionInitFill fix pass, which is not in this checkout.
_SIGNATURE = {"x_ptr": "*fp16", "y_ptr": "*fp16", "output_ptr": "*fp16"}
_CONSTEXPRS = {"M": 1, "N": 64, "BLOCK_M": 1, "BLOCK_N": 64}

# The scheduler inside dbo-opt requires the compute to be a ``linalg`` op whose
# ``outs`` is a fresh ``tensor.empty``. Tensor-level ``arith`` leaves the memref
# as ``strided<..., offset: ?>`` and dbo-opt rejects the ``ktdp.load`` operand;
# an aliased ``outs`` fails later in the same way. Both fixes must anchor on a
# *core* pass — ``_make_ktir`` silently ignores any other anchor — and dict
# order decides which runs first, so unalias_linalg_outs is listed second.
_REQUIRED_FIXES = {
    "convert_elementwise_to_linalg": "lower_compute_ops",
    "unalias_linalg_outs": "lower_compute_ops",
}


@triton.jit
def _add_kernel_1core(x_ptr, y_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1],
                                         block_shape=[BLOCK_M, BLOCK_N])
    x = x_desc.load([0, 0])
    y = y_desc.load([0, 0])
    out_desc.store([0, 0], x + y)


def _compile_options():
    return {
        "grid": (1,),
        "base_addresses": base_addresses_for(_SIGNATURE.values()),
        "required_fixes": dict(_REQUIRED_FIXES),
    }


@pytest.fixture(scope="module")
def dbo_opt():
    """Resolved dbo-opt path, or skip the test."""
    path = resolve_dbo_opt(required=False)
    if path is None:
        pytest.skip(f"dbo-opt not resolvable from knobs.spyre.dbo_opt="
                    f"{knobs.spyre.dbo_opt!r}")
    return path


@pytest.fixture(scope="module")
def compiled(dbo_opt):
    src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                    constexprs=dict(_CONSTEXPRS))
    return triton_compile(src, target=_TARGET, options=_compile_options())


# ---------------------------------------------------------------------------
# base_addresses — the fixed slot policy
# ---------------------------------------------------------------------------

class TestBaseAddresses:

    def test_element_indexed_16gib_slots(self):
        # Slot i is based at i * 16 GiB, expressed in ELEMENTS: for fp16 that
        # divides by 2. These are the same three constants torch-spyre's
        # Inductor path bakes for a 3-buffer fp16 kernel.
        assert base_addresses_for(["*fp16", "*fp16", "*fp16"]) == (
            0, 8589934592, 17179869184)

    def test_scales_with_element_width(self):
        assert base_addresses_for(["*fp32", "*fp32"]) == (0, 4294967296)
        assert base_addresses_for(["*i8", "*i8"]) == (0, 17179869184)

    def test_skips_non_pointer_arguments(self):
        # Only ``*``-prefixed entries are pointers; a runtime scalar or a
        # constexpr consumes no address slot. Positions must stay dense so slot
        # i and pointer i agree by construction.
        assert base_addresses_for(["*fp16", "i32", "*fp16", "constexpr"]) == (
            0, 8589934592)

    def test_seven_pointers_is_the_ceiling(self):
        assert len(base_addresses_for(["*fp16"] * 7)) == 7
        with pytest.raises(ValueError, match="at most 7 pointer arguments"):
            base_addresses_for(["*fp16"] * 8)

    def test_const_pointer_marker_is_ignored(self):
        # "*k" marks a const pointer; it does not change the element width.
        assert base_addresses_for(["*kfp16", "*fp16"]) == (0, 8589934592)

    def test_unknown_element_width_raises(self):
        with pytest.raises(ValueError, match="no usable byte width"):
            base_addresses_for(["*float128"])

    def test_sub_byte_element_raises(self):
        # i1 is a bit, and a base address is an element index, so there is no
        # honest byte width to divide by.
        with pytest.raises(ValueError, match="no usable byte width"):
            base_addresses_for(["*i1", "*i1"])


# ---------------------------------------------------------------------------
# compile_time_launch_options — the grid and the addresses as compile inputs
# ---------------------------------------------------------------------------

class TestCompileTimeLaunchOptions:

    def _backend(self):
        return SpyreBackend(_TARGET)

    def test_grid_and_addresses(self):
        spec = [("*fp16", 16), ("*fp16", 16), ("i32", None)]
        extra = self._backend().compile_time_launch_options((4, ), spec)
        assert extra == {"grid": (4, ), "base_addresses": (0, 8589934592)}

    def test_keys_are_option_fields(self):
        # _pack_args rejects any launch kwarg that is not a field of the parsed
        # options, so every key this returns must be one.
        extra = self._backend().compile_time_launch_options((1, ), [("*fp16", 16)])
        assert set(extra) <= set(SpyreOptions.__dataclass_fields__)

    def test_no_grid_on_warmup(self):
        # warmup=True passes grid=None; leave the option default alone.
        extra = self._backend().compile_time_launch_options(None, [("*fp16", 16)])
        assert "grid" not in extra

    def test_callable_grid_rejected(self):
        # The grid is baked into the artifact, so it cannot be a launch-time
        # function of the bound arguments.
        with pytest.raises(ValueError, match="callable grid"):
            self._backend().compile_time_launch_options(lambda meta: (1, ),
                                                        [("*fp16", 16)])

    def test_other_backends_contribute_nothing(self):
        # The hook's default on BaseBackend must stay a no-op for the GPU paths.
        from triton.backends.compiler import BaseBackend
        assert BaseBackend.compile_time_launch_options(
            object(), (1, ), [("*fp16", 16)]) == {}


# ---------------------------------------------------------------------------
# hash() — dbo-opt and device identity in the cache key
# ---------------------------------------------------------------------------

class TestBackendHash:

    def test_folds_dbo_opt_and_device(self):
        h = SpyreBackend(_TARGET).hash()
        assert h.startswith("spyre-1-")
        assert "dbo_opt-" in h and "device-" in h

    def test_unset_device_is_not_an_empty_digest(self, monkeypatch):
        # No device file named means dbo-opt picks its own default; there is no
        # digest to fold, so the key says so rather than hashing "".
        monkeypatch.setattr(knobs.spyre, "device", None)
        assert "device-dbo_opt_default" in SpyreBackend(_TARGET).hash()

    def test_named_device_changes_the_key(self, monkeypatch, tmp_path):
        device = tmp_path / "device.mlir"
        device.write_text("module {}\n")
        monkeypatch.setattr(knobs.spyre, "device", str(device))
        with_file = SpyreBackend(_TARGET).hash()
        device.write_text("module { /* different */ }\n")
        assert SpyreBackend(_TARGET).hash() != with_file

    def test_repointing_dbo_opt_changes_the_key(self, monkeypatch, tmp_path):
        before = SpyreBackend(_TARGET).hash()
        fake = tmp_path / "dbo-opt"
        fake.write_bytes(b"not really dbo-opt")
        monkeypatch.setattr(knobs.spyre, "dbo_opt", str(fake))
        assert SpyreBackend(_TARGET).hash() != before

    def test_missing_dbo_opt_hashes_instead_of_raising(self, monkeypatch):
        # hash() runs on every compile, including KTIR-only work that never
        # invokes the tool, so a missing tool must not raise here.
        monkeypatch.setattr(knobs.spyre, "dbo_opt", "definitely-not-on-path")
        assert "dbo_opt-missing-" in SpyreBackend(_TARGET).hash()


# ---------------------------------------------------------------------------
# The stage itself
# ---------------------------------------------------------------------------

class TestSpyrecodeStage:

    def test_binary_ext_is_set_on_the_instance(self):
        # CompiledKernel builds a fresh backend via make_backend(), so this has
        # to come from __init__ rather than add_stages.
        assert SpyreBackend(_TARGET).binary_ext == "spyrecode"

    def test_stage_is_registered_last(self):
        backend = SpyreBackend(_TARGET)
        stages = {}
        backend.add_stages(stages, backend.parse_options({}))
        assert list(stages) == ["ttir", "ktir", "spyrecode"]

    def test_artifact_holds_the_spyre_code_dir(self, compiled):
        # metadata["name"] is "" (issue #104), so the artifact is keyed by the
        # source function name; what matters is the ZIP's contents.
        names = set(zipfile.ZipFile(io.BytesIO(compiled.kernel)).namelist())
        assert {"spyrecode.json", "init_binary.bin"} <= names
        assert any(n.startswith("debug/") for n in names), sorted(names)

    def test_artifact_is_bytes(self, compiled):
        # binary_ext decides bytes-vs-text when CompiledKernel reads the cache
        # back; the spyrecode artifact must come back as bytes.
        assert isinstance(compiled.kernel, bytes)

    def test_cache_files_include_the_artifact(self, compiled):
        exts = {p.rsplit(".", 1)[-1] for p in compiled.metadata_group}
        assert {"ttir", "ktir", "spyrecode", "json"} <= exts

    def test_recompile_hits_the_cache(self, compiled):
        src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                        constexprs=dict(_CONSTEXPRS))
        again = triton_compile(src, target=_TARGET, options=_compile_options())
        assert again.hash == compiled.hash
        assert again.kernel == compiled.kernel

    def test_artifact_bytes_are_deterministic(self, compiled, monkeypatch):
        # The artifact digest is what SpyreUtils.load_binary unpacks under, so
        # identical inputs must give identical bytes. Real ZIP mtimes would make
        # every recompile look like a new binary.
        monkeypatch.setattr(knobs.compilation, "always_compile", True)
        src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                        constexprs=dict(_CONSTEXPRS))
        rebuilt = triton_compile(src, target=_TARGET, options=_compile_options())
        assert hashlib.sha256(rebuilt.kernel).hexdigest() == \
            hashlib.sha256(compiled.kernel).hexdigest()

    def test_missing_dbo_opt_raises_actionably(self, monkeypatch):
        monkeypatch.setattr(knobs.spyre, "dbo_opt", "definitely-not-on-path")
        with pytest.raises(RuntimeError, match="TRITON_SPYRE_DBO_OPT"):
            resolve_dbo_opt()

    def test_missing_device_file_raises(self, monkeypatch, tmp_path, dbo_opt):
        monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
        src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                        constexprs=dict(_CONSTEXPRS))
        with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
            triton_compile(src, target=_TARGET, options=_compile_options())


class TestSpyreOptionsFictions:
    """The three values that exist only to make Triton's ceiling checks no-ops."""

    def test_num_warps_and_shared_defaults(self):
        options = SpyreBackend(_TARGET).parse_options({})
        # num_warps * warp_size (1*1) must not exceed n_max_threads (1), and
        # shared (0) must not exceed max_shared_mem (0).
        assert options.num_warps == 1
        assert options.shared == 0

    def test_instrumentation_mode_is_tolerated(self):
        # JITFunction.run injects this into kwargs unconditionally, and
        # _pack_args rejects launch kwargs absent from the parsed options.
        options = SpyreBackend(_TARGET).parse_options({"instrumentation_mode": ""})
        assert options.instrumentation_mode == ""
        assert "instrumentation_mode" in options.__dict__
