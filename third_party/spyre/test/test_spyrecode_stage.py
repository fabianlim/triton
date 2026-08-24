#!/usr/bin/env python3
"""The ``spyrecode`` compile stage — KTIR to a loadable Spyre binary.

Stays in pytest because every test here needs ``dbo-opt``: the stage shells out to
it, so there is nothing to assert without one. The option surface and the
base-address policy, which need neither a tool nor a device, are python-driven lit
tests in ``test/python/backend-options.py``.

Every setting these need is a knob -- ``knobs.spyre.dbo_opt`` and
``knobs.spyre.device`` -- and the backend reads them only through ``knobs``, never
from the environment. Override them the way any Triton knob is overridden, in
process or from the environment via their ``TRITON_SPYRE_*`` names; the tests do
not read the environment either.
"""

from backend_utils import *  # noqa: F401,F403  (shared kernels + helpers)
from backend_utils import (
    _CONSTEXPRS,
    _SIGNATURE,
    _TARGET,
    _add_kernel_1core,
    _compile_options,
)

import io
import zipfile

import pytest
from triton import knobs
from triton.compiler.compiler import ASTSource, compile as triton_compile

from backend.compiler import SpyreBackend, resolve_dbo_opt, resolve_device

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
        # Through the stage, so the resolver is reached the way a compile reaches
        # it, not just called directly.
        monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
        src = ASTSource(fn=_add_kernel_1core, signature=dict(_SIGNATURE),
                        constexprs=dict(_CONSTEXPRS))
        with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
            triton_compile(src, target=_TARGET, options=_compile_options())

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


