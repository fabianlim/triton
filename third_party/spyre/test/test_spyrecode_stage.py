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
from triton.compiler.compiler import compile as triton_compile

from utils import spyre_target

from backend.compiler import SpyreBackend, resolve_dbo_opt, resolve_device


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

    Every test here gets its binary from a fixture; none builds an ASTSource of its
    own. Two do call triton_compile a second time, because recompiling is precisely
    what they assert, and take ``spyrecode_options`` to do it with the same options
    the first compile used. The conftest fixtures are a chain, not a checklist --
    each depends on the one above, so asking for a later one brings the earlier:

    ======================= =============================================
    ``compilable_example``  the variant key; ``params=COMPILES_TO_BINARY``,
                            so requesting it, directly or not, parametrizes
    ``spyrecode_options``   compile options for that variant. Pure data --
                            it never skips
    ``dbo_opt``             the resolved tool path, or ``pytest.skip``. This,
                            and only this, is what makes a test skip
    ``binary_source``       an ASTSource for the variant, not yet compiled
    ``compiled``            that source through every stage, symbolic
    ``compiled_baked``      the same, with addresses baked in -- the
                            non-default mode, where the backend derives them
    ======================= =============================================

    So ``compiled`` or ``compiled_baked`` alone is already both parametrized and
    gated: one input, not three. Ask for nothing the body does not use.

    ``test_missing_device_file_raises`` is the one exception, and it is not an
    inconsistency worth removing: it asserts that a compile *fails*, so a fixture
    whose value is a compile that succeeded cannot serve it. It takes
    ``binary_source`` plus ``dbo_opt``, which is the subset rule below applied
    rather than broken.

    Subsets are safe, with one trap. Because these are module-scoped and pytest
    caches each per parameter, a partial request can never hand back a different
    variant than its siblings -- there is one ``compilable_example`` value per run
    either way. The trap is taking ``spyrecode_options`` (or ``binary_source``) and
    compiling *without* ``dbo_opt``: parametrized but not gated, so where no tool
    exists it fails instead of skipping. Anything that compiles takes ``dbo_opt``,
    or takes a fixture that already did.
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

    def test_derived_addresses_reach_the_artifact(self, compiled_baked):
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
        # ``compiled_baked`` rather than ``compiled`` because nothing is derived in
        # the default symbolic mode; baking is what makes the value exist.
        assert tuple(compiled_baked.metadata.base_addresses) == (
            0, 8589934592, 17179869184)

    def test_missing_device_file_raises(self, dbo_opt, binary_source,
                                        spyrecode_options, monkeypatch, tmp_path):
        # The one test here that does not take a compile, and cannot: it asserts a
        # compile *fails*, so a fixture returning one that succeeded is the wrong
        # input -- taking it would compile twice and assert on neither. It takes the
        # source instead, and ``dbo_opt`` directly for the skip that ``compiled``
        # would otherwise have carried.
        monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
        with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
            triton_compile(binary_source, target=spyre_target(),
                           options=spyrecode_options)
