#!/usr/bin/env python3
"""The ``spyrecode`` compile stage — KTIR to a loadable Spyre binary.

Two classes, split by whether the test needs a real ``dbo-opt`` compile:

* ``TestStageWithoutTheTool`` — the stage's registration and the two path
  resolvers. Nothing here shells out, so nothing here skips.
* ``TestStageThroughARealCompile`` — everything that needs an actual binary, and
  so skips when no ``dbo-opt`` resolves.

Both stay in pytest rather than moving to ``test/python/`` as lit tests, because
those files are named ``*-test.py`` with a hyphen and pytest does not collect
them: moving these would leave ``pytest third_party/spyre/test``, the command in
both README.md and CLAUDE.md, not exercising this stage at all. The option surface
and the base-address arithmetic, which need neither a tool nor a compile, are lit
tests in ``test/python/backend-options-test.py``.

The kernel comes from whichever fixture variants declare ``compiles_to_binary`` in
their ``meta.py`` -- today one, the loop-free single-tile add, because dbo-opt
refuses the loop the others outline from their program-id distribution.

Every setting these need is a knob -- ``knobs.spyre.dbo_opt`` and
``knobs.spyre.device`` -- and the backend reads them only through ``knobs``, never
from the environment. Override them the way any Triton knob is overridden.
"""

import hashlib
import io
import zipfile

import pytest
from triton import knobs
from triton.compiler.compiler import ASTSource, compile as triton_compile

from conftest import EXAMPLES
from utils import spyre_target

from backend.compiler import SpyreBackend, resolve_dbo_opt, resolve_device


def _source(key):
    """An ASTSource for a fixture variant, for the tests that recompile it."""
    entry = EXAMPLES[key]
    constexprs = {k: v[0] for k, v in entry["params"].items()
                  if k in entry["constexpr"]}
    return ASTSource(fn=entry["kernel_fn"], signature=dict(entry["signature"]),
                     constexprs=constexprs)


class TestStageWithoutTheTool:
    """Registration and resolvers. No compile, so no skip."""

    def test_binary_ext_is_set_on_the_instance(self):
        # CompiledKernel builds a fresh backend via make_backend(), so this has
        # to come from __init__ rather than add_stages.
        assert SpyreBackend(spyre_target()).binary_ext == "spyrecode"

    def test_stage_is_registered_last(self):
        backend = SpyreBackend(spyre_target())
        stages = {}
        backend.add_stages(stages, backend.parse_options({}))
        assert list(stages) == ["ttir", "ktir", "spyrecode"]

    def test_missing_dbo_opt_raises_actionably(self, monkeypatch):
        monkeypatch.setattr(knobs.spyre, "dbo_opt", "definitely-not-on-path")
        with pytest.raises(RuntimeError, match="TRITON_SPYRE_DBO_OPT"):
            resolve_dbo_opt()

    def test_resolve_device_reports_unset_as_no_file(self, monkeypatch):
        # Unset is a real configuration, not a failure: dbo-opt falls back to its
        # own default. The resolver has to say "no file" rather than raise, or a
        # compile with the knob unset could never run.
        monkeypatch.setattr(knobs.spyre, "device", None)
        assert resolve_device() is None
        assert resolve_device(required=False) is None

    def test_resolve_device_returns_an_absolute_path(self, monkeypatch, tmp_path):
        f = tmp_path / "dev.mlir"
        f.write_text("// device")
        monkeypatch.setattr(knobs.spyre, "device", str(f))
        assert resolve_device() == str(f.resolve())

    def test_resolve_device_tolerates_a_missing_file_when_not_required(
            self, monkeypatch, tmp_path):
        # What SpyreBackend.hash() needs: describing the setting must not raise,
        # because a cache key has to exist even for a misconfigured device.
        monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
        assert resolve_device(required=False) is None
        with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
            resolve_device()


class TestStageThroughARealCompile:
    """Tests that need a binary, and so a resolvable ``dbo-opt``.

    Three conftest fixtures serve this class, and they are not three things you
    must remember to ask for -- they form a chain, so asking for the last one
    brings the others:

    ==================== ================================================
    ``binary_example``   the variant key; ``params=COMPILES_TO_BINARY``, so
                         requesting it (directly or not) is what parametrizes
    ``spyrecode_options`` compile options for that variant. Depends on
                         ``binary_example``. Pure data -- it never skips
    ``dbo_opt``          the resolved tool path, or ``pytest.skip``. This, and
                         only this, is what makes a test skip
    ==================== ================================================

    ``compiled`` depends on all three, so a test taking ``compiled`` alone is
    already both parametrized and skipped -- one input, not three. Take a fixture
    only where the body uses it: ``binary_example`` appears below just in the two
    tests that build their own ``ASTSource`` from it.

    Subsets are safe, with one exception. Because the fixtures are module-scoped
    and pytest caches each per parameter, a partial request can never hand back a
    different variant than its siblings -- there is one ``binary_example`` value
    per test run either way. The exception is asking for ``spyrecode_options``
    (or ``binary_example``) and then compiling *without* ``dbo_opt``: that is
    parametrized but not gated, so on a machine with no tool it fails instead of
    skipping. If a test compiles, it takes ``dbo_opt`` or it takes ``compiled``.
    """

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

    def test_recompile_hits_the_cache(self, compiled, spyrecode_options):
        again = triton_compile(compiled.src, target=spyre_target(),
                               options=spyrecode_options)
        assert again.hash == compiled.hash
        assert again.kernel == compiled.kernel

    def test_artifact_bytes_are_deterministic(self, compiled, spyrecode_options,
                                              monkeypatch):
        # The artifact digest is what SpyreUtils.load_binary unpacks under, so
        # identical inputs must give identical bytes. Real ZIP mtimes would make
        # every recompile look like a new binary.
        monkeypatch.setattr(knobs.compilation, "always_compile", True)
        rebuilt = triton_compile(compiled.src, target=spyre_target(),
                                 options=spyrecode_options)
        assert hashlib.sha256(rebuilt.kernel).hexdigest() == \
            hashlib.sha256(compiled.kernel).hexdigest()

    def test_derived_addresses_reach_the_artifact(self, dbo_opt, spyrecode_options,
                                                 binary_example):
        # The one surviving assertion on the *value* _make_ktir derives from the
        # TTIR pointer types: three fp16 pointers land on the i * 16 GiB segments,
        # counted in elements. It runs through a whole compile because metadata is
        # the only place the derivation is observable, and only a compile produces
        # metadata -- the tool-free lowering the option tests use
        # (utils.make_ktir_mod) returns the module alone. The rest of what used to
        # be asserted here -- widths per pointer, the seven-pointer ceiling,
        # non-pointer arguments, sub-byte rejection -- is asserted directly on
        # _segment_addresses in backend-options-test.py::TestBaseAddresses, which
        # needs neither a module nor this tool, so only one test needs to pay for
        # a compile.
        #
        # It builds its own source rather than taking ``compiled``, because it
        # compiles in the non-default mode: symbolic_args=False is explicit, since
        # symbolic is the default and nothing is derived in that mode.
        src = _source(binary_example)
        baked = triton_compile(src, target=spyre_target(),
                               options={**spyrecode_options,
                                        "symbolic_args": False})
        assert tuple(baked.metadata.base_addresses) == (
            0, 8589934592, 17179869184)

    def test_missing_device_file_raises(self, dbo_opt, spyrecode_options,
                                        binary_example, monkeypatch, tmp_path):
        # Through the stage, so the resolver is reached the way a compile reaches
        # it, not just called directly. Its own source again, because ``compiled``
        # is module-scoped and would have been built before this monkeypatch.
        monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
        src = _source(binary_example)
        with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
            triton_compile(src, target=spyre_target(), options=spyrecode_options)
