# RUN: %python -m pytest %s -q
# REQUIRES: dbo-opt, spyre-device

"""Launch a compiled fixture on the Spyre device and check it numerically.

The last link: a fixture compiles to a spyreCodeDir, the device runs it, and the
result is compared against the same oracle and the same tolerances the CPU
numerical tests use. Nothing here is a second convention -- inputs, reference,
``output_key``, ``rtol`` and ``atol`` all come from the variant's ``meta.py``.

Two features gate this file, and both are needed: ``dbo-opt`` to produce the
binary, ``spyre-device`` for an interpreter that can launch it. Absent either, lit
reports Unsupported, which is visible -- unlike a pytest skip inside a passing RUN
line, which is not. See ``test/lit.cfg.py``.

The launch itself happens in a **child process**, ``scripts/device_runner.py``, run
by a different interpreter. That is not indirection for its own sake:

* ``torch_spyre/_C.so`` is dlopen'd at import and ``ld.so`` reads
  ``LD_LIBRARY_PATH`` at exec, so it must be set before the interpreter starts;
* a Spyre device opens exclusively per process, so this process must not hold one
  while the child opens it -- hence nothing here imports ``torch_spyre``;
* the triton venv has no ``torch``, and the interpreter that does also carries a
  pip ``triton`` that would fight the editable one.

Serialization comes for free: no ``pytest-xdist`` is installed and nothing sets
``-n``, so these run one at a time and spawn one child at a time. If that ever
changes, this file needs a lock, because the device does not.
"""

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

from conftest import EXAMPLES

# scripts/device_runner.py, relative to this file: test/python -> spyre/scripts.
_RUNNER = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "device_runner.py")


@pytest.fixture(scope="module")
def launch_python():
    """The interpreter that can import torch_spyre, or skip.

    A path rather than a discovery rule, and an environment variable rather than
    anything hardcoded: which venv carries torch-spyre is a property of the
    machine. ``TRITON_SPYRE_LAUNCH_PYTHON`` names it. The same variable decides
    lit's ``spyre-device`` feature, so a lit run and a direct pytest run agree.
    """
    python = os.environ.get("TRITON_SPYRE_LAUNCH_PYTHON")
    if not python or not os.access(python, os.X_OK):
        pytest.skip("TRITON_SPYRE_LAUNCH_PYTHON is unset or not executable")
    return python


def _stage(directory, entry, artifact):
    """Write the spyreCodeDir and the fixture's inputs where the runner expects."""
    import shutil
    from backend.driver import SpyreUtils

    # Through the driver's own unpack, so the child sees the layout a real launch
    # would see rather than a second interpretation of the ZIP.
    module, _, _, _, _ = SpyreUtils().load_binary("", artifact, 0, 0)
    shutil.copytree(module, directory / "spyrecode")

    inputs = entry["inputs"](**entry["param_values"])
    for name, arr in inputs.items():
        np.save(directory / f"{name}.npy", arr)
    return inputs


def test_launches_and_matches_the_oracle(launch_python, compiled, compilable_example,
                                         tmp_path):
    entry = EXAMPLES[compilable_example]
    inputs = _stage(tmp_path, entry, compiled.kernel)

    # Kernel order, which is the address binding: the correction flit is built by
    # walking the launch arguments positionally, so segment i belongs to pointer i.
    pointers = [n for n in entry["signature"] if str(entry["signature"][n]).startswith("*")]
    output_key = entry["output_key"]
    assert output_key in pointers, (output_key, pointers)

    env = dict(os.environ)
    # Explicit rather than inherited. torch_spyre's import sets this by
    # setdefault, but an inherited "0" would win and silently switch the child to
    # binding addresses -- wrong for the symbolic artifact the backend emits.
    env["BUNDLE_SYMBOLIC_ARGS"] = "1"

    proc = subprocess.run(
        [launch_python, str(_RUNNER), "--dir", str(tmp_path),
         "--pointers", ",".join(pointers), "--output", output_key],
        capture_output=True, text=True, env=env)
    # The child's stdout is one JSON object per stage; on failure both streams are
    # reported, because the useful line is as often a dlopen error on stderr as it
    # is a Python traceback.
    if proc.returncode != 0:
        pytest.fail(f"device_runner failed (exit {proc.returncode})\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    stages = [json.loads(line) for line in proc.stdout.splitlines()
              if line.startswith("{")]
    by_stage = {s["stage"]: s for s in stages}
    assert "done" in by_stage, proc.stdout

    # The plan a symbolic artifact must have: the host step builds the correction
    # flit, the H2D ships it, and only then does the device compute. A binary with
    # a single Compute step was baked, not symbolic.
    assert by_stage["prepare"]["steps"] == ["HostCompute", "H2D", "Compute"], \
        by_stage["prepare"]

    got = np.load(tmp_path / by_stage["done"]["output"])
    ref = entry["reference"](inputs)

    # An all-zero output would pass assert_allclose against an all-zero reference
    # while proving only that nothing ran, so the write is asserted separately
    # from the values.
    assert by_stage["done"]["nonzero"] > 0, (
        "device wrote nothing: the output buffer is still the zeros it was "
        f"launched with ({by_stage['done']})")

    np.testing.assert_allclose(got, ref, rtol=entry.get("rtol", 1e-6),
                               atol=entry.get("atol", 0.0))
