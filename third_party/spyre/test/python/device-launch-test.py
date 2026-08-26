# RUN: %python -m pytest %s -q
# REQUIRES: dbo-opt, spyre-device

"""Launch a compiled fixture on the Spyre device and check it numerically.

The last link: a fixture compiles to a spyreCodeDir, the device runs it, and the
result is compared against the same oracle and the same tolerances the CPU
numerical tests use. Nothing here is a second convention -- the binary comes from
the ``compiled`` fixture, and inputs, ``reference``, ``output_key``, ``rtol`` and
``atol`` all come from the variant's ``meta.py``. The launch itself is
:class:`SpyreDeviceTester`, the device peer of ``KTIRCpuTester``; the two numerical
tests are the same test against two backends.

The variants covered are whatever declares ``compiles_to_binary`` -- the
``compilable_example`` fixture parametrizes over them, so a second one extends this
with no change here. Today exactly one qualifies.

Two features gate this file, and both are needed: ``dbo-opt`` to produce the
binary, ``spyre-device`` for an interpreter that can launch it. Absent either, lit
reports Unsupported, which is visible -- unlike a pytest skip inside a passing RUN
line, which is not. See ``test/lit.cfg.py``.

Nothing in this process imports ``torch_spyre``: a Spyre device opens exclusively
per process, so the parent must not hold one while the child does. See
``SpyreDeviceTester`` for why the launch is a separate process at all, and for the
serialization this relies on.
"""

import numpy as np

from conftest import EXAMPLES, SpyreDeviceTester


class TestDeviceLaunch(SpyreDeviceTester):
    """The compilable variants, launched on hardware and compared to the oracle."""

    def test_launches_and_matches_the_oracle(self, compiled, compilable_example,
                                             tmp_path):
        entry = EXAMPLES[compilable_example]
        inputs = entry["inputs"](**entry["param_values"])

        result = self.run_device(
            compiled.kernel, directory=tmp_path, signature=entry["signature"],
            output_key=entry["output_key"], inputs=inputs)

        # The plan a symbolic artifact must have: the host step builds the
        # correction flit, the H2D ships it, and only then does the device compute.
        # A binary with a single Compute step was baked, not symbolic.
        assert result.steps == ["HostCompute", "H2D", "Compute"], result.stages

        # An all-zero output would pass assert_allclose against an all-zero
        # reference while proving only that nothing ran, so the write is asserted
        # separately from the values.
        assert result.nonzero > 0, (
            "device wrote nothing: the output buffer is still the zeros it was "
            f"launched with ({result.stages['done']})")

        np.testing.assert_allclose(result.output, entry["reference"](inputs),
                                   rtol=entry.get("rtol", 1e-6),
                                   atol=entry.get("atol", 0.0))
