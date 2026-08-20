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
    _segment_addresses,
    infer_base_addresses_from_ptr_types,
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
    # No base_addresses: the backend derives them from the TTIR pointer types.
    return {
        "grid": (1,),
        "required_fixes": dict(_REQUIRED_FIXES),
    }


def _lower_to_ktir(signature=None, **options):
    """Run the ``ttir`` and ``ktir`` stages only; return ``(module, metadata)``.

    Stops short of ``spyrecode`` so the tests that care about the derivation need
    neither ``dbo-opt`` nor a device.
    """
    from triton._C.libtriton import ir

    backend = SpyreBackend(_TARGET)
    opts = backend.parse_options(options)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(fn=_add_kernel_1core,
                    signature=dict(signature or _SIGNATURE),
                    constexprs=dict(_CONSTEXPRS))
    mod = src.make_ir(_TARGET, opts, backend.get_codegen_implementation(opts),
                      backend.get_module_map(), context)
    # Keep the context alive for as long as the module: it owns the IR, and a
    # collected context leaves the returned module dangling (same reason
    # utils.make_ktir_mod does this).
    mod.context = context

    metadata = {}
    mod = backend._make_ttir(mod, metadata, opts)
    return backend._make_ktir(mod, metadata, opts), metadata


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
# base_addresses — the fixed segment policy
# ---------------------------------------------------------------------------

class TestBaseAddresses:
    """The policy function. Inputs are MLIR spellings, as
    ``ir.module.get_function_signature`` produces them ("*f16", not "*fp16")."""

    def test_element_indexed_16gib_slots(self):
        # Segment i is based at i * 16 GiB, expressed in ELEMENTS: for f16 that
        # divides by 2. These are the same three constants torch-spyre's
        # Inductor path bakes for a 3-buffer fp16 kernel.
        assert _segment_addresses(["*f16", "*f16", "*f16"]) == (
            0, 8589934592, 17179869184)

    def test_scales_with_element_width(self):
        assert _segment_addresses(["*f32", "*f32"]) == (0, 4294967296)
        assert _segment_addresses(["*i8", "*i8"]) == (0, 17179869184)
        assert _segment_addresses(["*bf16", "*bf16"]) == (0, 8589934592)
        # The f8 family names its width the same way, right after the "f".
        assert _segment_addresses(["*f8E4M3FN", "*f8E4M3FN"]) == (0, 17179869184)

    def test_each_pointer_uses_its_own_width(self):
        # A single global element stride would misplace the narrower type in a
        # mixed-precision kernel: segment 1 is 16 GiB either way, but how many
        # elements that is depends on the pointer sitting in it.
        assert _segment_addresses(["*f32", "*f16", "*i8"]) == (
            0, 8589934592, 34359738368)

    def test_skips_non_pointer_arguments(self):
        # Only ``*``-prefixed entries are pointers; a runtime scalar consumes no
        # address segment. Positions must stay dense so segment i and pointer i agree
        # by construction.
        assert _segment_addresses(["*f16", "i32", "*f16", "index"]) == (
            0, 8589934592)

    def test_seven_pointers_is_the_ceiling(self):
        assert len(_segment_addresses(["*f16"] * 7)) == 7
        with pytest.raises(ValueError, match="at most 7 pointer arguments"):
            _segment_addresses(["*f16"] * 8)

    def test_unknown_element_width_raises(self):
        with pytest.raises(ValueError, match="no usable byte width"):
            _segment_addresses(["*float128"])
        # A pointer to a pointer has no element width of its own.
        with pytest.raises(ValueError, match="no usable byte width"):
            _segment_addresses(["*!tt.ptr<f16>"])

    def test_sub_byte_element_raises(self):
        # i1 is a bit, and a base address is an element index, so there is no
        # honest byte width to divide by.
        with pytest.raises(ValueError, match="no usable byte width"):
            _segment_addresses(["*i1", "*i1"])


class TestInferBaseAddresses:
    """Derivation inside the backend, off the TTIR entry function.

    This is what replaced the hook contributing ``base_addresses``: the widths
    live in the IR as ``!tt.ptr<f16>``, and the specialized signature they come
    from is already part of the cache key, so deriving late cannot produce a
    wrong key.
    """

    def test_derived_addresses_land_in_metadata(self):
        # metadata is how the derivation reaches the spyrecode stage; nothing on
        # the launch path is involved.
        _, metadata = _lower_to_ktir()
        assert metadata["base_addresses"] == (0, 8589934592, 17179869184)

    def test_scales_with_the_kernels_own_element_type(self):
        _, metadata = _lower_to_ktir(
            signature={name: "*fp32" for name in _SIGNATURE})
        assert metadata["base_addresses"] == (0, 4294967296, 8589934592)

    def test_pointee_types_are_gone_after_the_pipeline(self):
        # Why the derivation cannot be deferred to the spyrecode stage:
        # ConvertFunctions has by then rewritten every !tt.ptr<f16> argument to a
        # plain `index`, and the element widths are no longer in the IR.
        mod, _ = _lower_to_ktir()
        text = str(mod)
        assert "!tt.ptr" not in text
        entry = next(line for line in text.splitlines() if "func.func" in line)
        assert entry.count("index") == len(_SIGNATURE), entry

    def test_a_module_without_a_kernel_raises(self, tmp_path):
        # The KTIR module is one such: ConvertFunctions turns the tt.func into a
        # func.func, which get_entry_func_name (looking for a Triton kernel) does
        # not recognize, so it must not be walked into blindly.
        from triton._C.libtriton import ir
        context = ir.context()
        ir.load_dialects(context)
        SpyreBackend(_TARGET).load_dialects(context)
        path = tmp_path / "empty.mlir"
        path.write_text("module {}\n")
        mod = ir.parse_mlir_module(str(path), context)
        mod.context = context
        with pytest.raises(RuntimeError, match="no kernel entry function"):
            infer_base_addresses_from_ptr_types(mod)

    def test_the_spyrecode_stage_requires_them(self):
        # Empty metadata means nobody inferred the addresses — a compile entered
        # at the .ktir stage, where the widths are already gone. Say so, rather
        # than emit a binary based at whatever the scheduler defaults to.
        backend = SpyreBackend(_TARGET)
        mod, _ = _lower_to_ktir()
        with pytest.raises(RuntimeError, match="no base addresses were derived"):
            backend._make_spyrecode(mod, {}, backend.parse_options({}))

    def test_the_count_alone_does_not_determine_the_addresses(self):
        # Guards the function's name as much as its behaviour: same number of
        # pointer arguments, different addresses, because each segment is 16 GiB
        # expressed in that pointer's own elements.
        assert (_segment_addresses(["*f32", "*f32"])
                != _segment_addresses(["*f16", "*f16"]))


# ---------------------------------------------------------------------------
# base_addresses as an explicit override
# ---------------------------------------------------------------------------

# Addresses that the i * 16 GiB policy cannot produce: this is the shape of what
# `dft triton-lower` supplies, dev_ptr // word_length out of a fixture's
# data_dti.json, i.e. where that fixture's buffers actually live on the device.
_DTI_ADDRESSES = (0, 262144, 524288)


class TestBaseAddressesOverride:
    """``SpyreOptions.base_addresses``, set by a caller that has real addresses.

    dataflow-test-framework's ``dft triton-lower`` (``dftest/triton.py``) is the
    live example, and it needs both halves: the field, and a ``required_fixes``
    entry naming ``materialize_base_addresses`` so the pass runs at KTIR time.
    """

    def _materialized_entry(self, mod):
        return next(line for line in str(mod).splitlines() if "func.func" in line)

    def test_required_fixes_materializes_at_ktir_time(self):
        # The dft path exactly: name the pass, anchored on the last core pass,
        # and hand it the DTI addresses.
        mod, _ = _lower_to_ktir(
            base_addresses=list(_DTI_ADDRESSES),
            required_fixes={"materialize_base_addresses": "convert_functions"})
        text = str(mod)
        # A zero-argument entry function is what the dataflow scheduler wants.
        assert "func.func" in text
        assert self._materialized_entry(mod).count("index") == 0, text
        for address in _DTI_ADDRESSES[1:]:
            assert str(address) in text, f"{address} missing from:\n{text}"

    def test_derived_addresses_do_not_overwrite_the_override(self, monkeypatch):
        # The regression this pins: the override must survive the derivation that
        # _make_ktir now also does. Stop before dbo-opt — the pass has run by
        # then, and this needs neither the tool nor a device.
        import backend.compiler as compiler_module

        class _Stop(Exception):
            pass

        def _stop(*args, **kwargs):
            raise _Stop
        monkeypatch.setattr(compiler_module, "resolve_dbo_opt", _stop)

        backend = SpyreBackend(_TARGET)
        options = backend.parse_options({"base_addresses": list(_DTI_ADDRESSES)})
        mod, metadata = _lower_to_ktir()
        assert metadata["base_addresses"] == (0, 8589934592, 17179869184)

        with pytest.raises(_Stop):
            backend._make_spyrecode(mod, metadata, options)

        text = str(mod)
        assert self._materialized_entry(mod).count("index") == 0, text
        for address in _DTI_ADDRESSES[1:]:
            assert str(address) in text, f"{address} missing from:\n{text}"
        # The derived segment-1 address must not have been baked in instead.
        assert "8589934592" not in text, text

    def test_a_list_is_normalized_to_a_tuple(self):
        # SPYRE_OPTIONS arrives as JSON, so the field is handed a list; it has to
        # end up hashable for the option hash and dataclass equality.
        options = SpyreBackend(_TARGET).parse_options(
            {"base_addresses": list(_DTI_ADDRESSES)})
        assert options.base_addresses == _DTI_ADDRESSES
        assert isinstance(options.base_addresses, tuple)

    def test_the_override_is_in_the_cache_key(self):
        backend = SpyreBackend(_TARGET)
        derived = backend.parse_options({})
        overridden = backend.parse_options({"base_addresses": list(_DTI_ADDRESSES)})
        assert derived.hash() != overridden.hash()


# ---------------------------------------------------------------------------
# compile_time_launch_options — the grid, and nothing else
# ---------------------------------------------------------------------------

class TestCompileTimeLaunchOptions:

    def _backend(self):
        return SpyreBackend(_TARGET)

    def test_contributes_the_grid_alone(self):
        # The grid is the one thing that cannot be recovered later. Anything
        # derivable from the specialization is derived inside the backend
        # instead, because the specialization is already in the cache key.
        spec = [("*fp16", 16), ("*fp16", 16), ("i32", None)]
        extra = self._backend().compile_time_launch_options((4, ), spec)
        assert extra == {"grid": (4, )}

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


class TestSymbolicArgs:
    """The argument-passing mode, as an explicit compile option.

    ``symbolic_args=False`` bakes the fixed HBM base addresses in and lets
    torch-spyre bind the buffers positionally from the tensor list;
    ``symbolic_args=True`` would leave them symbolic for a runtime that patches
    them through the correction table, which nothing here emits yet.
    """

    def test_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("BUNDLE_SYMBOLIC_ARGS", raising=False)
        assert SpyreBackend(_TARGET).parse_options({}).symbolic_args is False

    @pytest.mark.parametrize("env, expected", [
        # Polarity follows torch-spyre's prepare_kernel.cpp, where
        # bind_io_addresses_ = (env == nullptr || env != "1"): only the literal
        # "1" means symbolic, and anything else binds from the tensor list.
        ("1", True),
        ("0", False),
        ("", False),
        ("true", False),
    ])
    def test_env_var_sets_the_default(self, monkeypatch, env, expected):
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", env)
        options = SpyreBackend(_TARGET).parse_options({})
        assert options.symbolic_args is expected

    def test_explicit_option_wins_over_the_env(self, monkeypatch):
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", "1")
        options = SpyreBackend(_TARGET).parse_options({"symbolic_args": False})
        assert options.symbolic_args is False

    def test_participates_in_the_cache_key(self):
        backend = SpyreBackend(_TARGET)
        bound = backend.parse_options({"symbolic_args": False})
        symbolic = backend.parse_options({"symbolic_args": True})
        # It changes the emitted artifact, so the two must not share a key.
        assert bound.hash() != symbolic.hash()

    def test_the_env_var_reaches_the_key_too(self, monkeypatch):
        # The point of reading the env in parse_options rather than at
        # pass-install time: a compile under one value must not be served from
        # the cache under the other.
        backend = SpyreBackend(_TARGET)
        monkeypatch.delenv("BUNDLE_SYMBOLIC_ARGS", raising=False)
        unset = backend.parse_options({}).hash()
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", "1")
        assert backend.parse_options({}).hash() != unset

    def test_symbolic_mode_raises_rather_than_skipping(self, monkeypatch):
        # Skipping MaterializeBaseAddresses alone would leave the entry function
        # taking `index` pointer arguments, which the scheduler rejects deep
        # inside dbo-opt with an operand-type error. Fail here instead, and
        # without needing dbo-opt at all.
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", "1")
        src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                        constexprs=dict(_CONSTEXPRS))
        with pytest.raises(NotImplementedError,
                           match="BUNDLE_SYMBOLIC_ARGS=1"):
            triton_compile(src, target=_TARGET, options=_compile_options())

    def test_mutually_exclusive_with_base_addresses(self):
        # Honouring one and dropping the other would silently pick a mode the
        # caller did not ask for.
        with pytest.raises(ValueError, match="mutually exclusive"):
            SpyreBackend(_TARGET).parse_options(
                {"symbolic_args": True, "base_addresses": [0, 262144]})

    def test_no_addresses_are_inferred_in_symbolic_mode(self, monkeypatch):
        # Inferring them anyway would surface a pointer-width or pointer-count
        # complaint instead of the NotImplementedError the caller is about to get.
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", "1")
        _, metadata = _lower_to_ktir()
        assert "base_addresses" not in metadata

    def test_symbolic_mode_reports_itself_not_a_width_problem(self, monkeypatch):
        # The ordering that guard buys: a kernel whose pointee type has no usable
        # width must still report the mode, not the width.
        monkeypatch.setenv("BUNDLE_SYMBOLIC_ARGS", "1")
        backend = SpyreBackend(_TARGET)
        options = backend.parse_options({})
        mod, metadata = _lower_to_ktir(symbolic_args=True)
        with pytest.raises(NotImplementedError):
            backend._make_spyrecode(mod, metadata, options)


class TestSpyreOptionsFictions:
    """``num_warps`` and ``shared`` are inert; ``instrumentation_mode`` is not."""

    def test_num_warps_and_shared_defaults(self):
        options = SpyreBackend(_TARGET).parse_options({})
        # num_warps * warp_size (1*1) must not exceed n_max_threads (1), and
        # shared (0) must not exceed max_shared_mem (0).
        assert options.num_warps == 1
        assert options.shared == 0

    def test_a_non_default_num_warps_is_tolerated(self):
        # Deliberately NOT an error. num_warps is the standard portable Triton
        # knob -- every triton.Config in an autotune list carries one -- and
        # nothing in the Spyre pipeline reads it, so rejecting it would break
        # portable kernels for no correctness gain. warp_size = 1 makes it vacuous.
        options = SpyreBackend(_TARGET).parse_options({"num_warps": 4})
        assert options.num_warps == 4

    def test_the_default_instrumentation_mode_is_tolerated(self):
        # JITFunction.run injects this into kwargs unconditionally, and
        # _pack_args rejects launch kwargs absent from the parsed options, so the
        # field has to exist and the empty default has to pass.
        options = SpyreBackend(_TARGET).parse_options({"instrumentation_mode": ""})
        assert options.instrumentation_mode == ""
        assert "instrumentation_mode" in options.__dict__

    def test_a_requested_instrumentation_mode_raises(self):
        # Unlike num_warps, a non-empty value here means the caller believes
        # instrumentation is running. Nothing on Spyre honours it, so accepting it
        # is a silent wrong answer rather than a harmless no-op.
        with pytest.raises(ValueError, match="instrumentation_mode"):
            SpyreBackend(_TARGET).parse_options({"instrumentation_mode": "profile"})

    def test_the_env_knob_is_reported_not_ignored(self, monkeypatch):
        # The realistic route in: TRITON_INSTRUMENTATION_MODE feeds
        # knobs.compilation.instrumentation_mode, which run() injects every launch.
        monkeypatch.setattr(knobs.compilation, "instrumentation_mode", "profile")
        with pytest.raises(ValueError, match="TRITON_INSTRUMENTATION_MODE"):
            SpyreBackend(_TARGET).parse_options(
                {"instrumentation_mode": knobs.compilation.instrumentation_mode})
