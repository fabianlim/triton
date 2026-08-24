# RUN: %python -m pytest %s -q

"""Backend options and the base-address policy — python-driven lit test.

A lit test that runs pytest on itself, following dataflow-scheduler's
``test/python``. These assert structured values, exception messages and cache
keys, none of which is IR, so there is no pass to run and nothing for FileCheck to
match; pytest's own asserts report a mismatch far better than a failed pattern
does. lit still owns discovery, so ``uv run lit`` remains the one entry point.

Nothing here needs ``dbo-opt`` or a device. The tests that do stay in
``test_spyrecode_stage.py``.
"""

from spyre_backend_fixtures import *  # noqa: F401,F403  (shared kernels + helpers)
from spyre_backend_fixtures import (
    _CONSTEXPRS,
    _SIGNATURE,
    _TARGET,
    _add_kernel_1core,
    _compile_options,
    _lower_to_ktir,
)

import pytest
import triton
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from backend.compiler import (
    SpyreBackend,
    SpyreOptions,
    _segment_addresses,
    infer_base_addresses_from_ptr_types,
    resolve_dbo_opt,
)

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
    """``num_warps`` and ``shared`` are not options at all; ``instrumentation_mode``
    has to be one."""

    def test_num_warps_and_shared_are_not_options(self):
        # They are read only from metadata, by CompiledKernel._init_handles, so
        # _make_ttir reports them there instead. Keeping them out of the options
        # keeps them out of options.hash(), so they cannot key an artifact they
        # cannot change.
        fields = set(SpyreOptions.__dataclass_fields__)
        assert "num_warps" not in fields
        assert "shared" not in fields

    def test_num_warps_at_a_call_site_is_rejected(self):
        # Spyre has no warps, and nothing in the pipeline reads this. Refusing it
        # says so, where accepting and ignoring it would not. triton.autotune,
        # which is where a portable kernel would sweep num_warps, cannot run on
        # this backend anyway -- get_benchmarker raises.
        with pytest.raises(TypeError, match="num_warps"):
            SpyreOptions(num_warps=4)

    def test_only_the_terminal_stage_reports_them(self):
        # Reported by _make_spyrecode, not by an earlier stage: a compile can
        # start partway down (an IRSource of a .ttir skips _make_ttir, one of a
        # .ktir skips _make_ktir too), and metadata without them means an
        # AttributeError inside _init_handles at the first launch. So the ttir+ktir
        # stages alone must NOT be what supplies them.
        _, metadata = _lower_to_ktir(**_compile_options())
        assert "num_warps" not in metadata
        assert "shared" not in metadata

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
