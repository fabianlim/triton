# RUN: %python -m pytest %s -q

"""The ``spyrecode`` stage's surface — registration, resolvers, tool failures.

Everything here runs without a working ``dbo-opt``, so this file carries no
``REQUIRES`` line. The tests that need one are in ``spyrecode-compile-test.py``,
which does.

Two tests do run a compile, and still belong here: they point
``knobs.spyre.dbo_opt`` at a tool that exists and exits 1 (``/bin/false``, and a
one-line script), because what they assert is the *failure* message. A working
compiler would defeat them.
"""

import pytest
from triton import knobs
from triton.compiler.compiler import compile as triton_compile

from utils import spyre_target

from backend.compiler import SpyreBackend, resolve_dbo_opt, resolve_device


class TestStageRegistration:

    def test_binary_ext_is_set_on_the_instance(self):
        # CompiledKernel builds a fresh backend via make_backend(), so this has
        # to come from __init__ rather than add_stages.
        assert SpyreBackend(spyre_target()).binary_ext == "spyrecode"

    def test_stage_is_registered_last(self):
        backend = SpyreBackend(spyre_target())
        stages = {}
        backend.add_stages(stages, backend.parse_options({}))
        assert list(stages) == ["ttir", "ktir", "spyrecode"]


class TestToolResolution:

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


class TestAFailingTool:
    """A tool that resolves and then fails -- the other half of "not found".

    These take ``binary_source`` without ``dbo_opt``, which the compile file calls
    a trap. It is not one here: the tool under test is one that exits 1, so no real
    dbo-opt is wanted and nothing is left ungated.
    """

    def test_reported_with_its_origin(self, binary_source, spyrecode_options,
                                      monkeypatch):
        monkeypatch.setattr(knobs.spyre, "dbo_opt", "/bin/false")
        with pytest.raises(RuntimeError) as excinfo:
            triton_compile(binary_source, target=spyre_target(),
                           options=spyrecode_options)
        message = str(excinfo.value)
        assert "/bin/false" in message
        assert "knobs.spyre.dbo_opt='/bin/false'" in message
        assert "TRITON_SPYRE_DBO_OPT" in message

    def test_a_bare_name_is_reported_as_coming_from_path(self, binary_source,
                                                        spyrecode_options,
                                                        monkeypatch, tmp_path):
        # The case nobody chose deliberately: a bare knob value means PATH picked
        # the binary, so an old install wins silently and the message must say so.
        failing = tmp_path / "dbo-opt"
        failing.write_text("#!/bin/sh\nexit 1\n")
        failing.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(knobs.spyre, "dbo_opt", "dbo-opt")
        with pytest.raises(RuntimeError, match="PATH, since knobs.spyre.dbo_opt"):
            triton_compile(binary_source, target=spyre_target(),
                           options=spyrecode_options)
