"""SIGNATURE + VARIANTS + reference oracle + input generator for chain.

Named for the *structure* under test rather than a mathematical function, the
same departure from the one-folder-per-function convention that ``spyreop``
makes: what these variants have in common is that the kernel contains more than
one compute, not what the computes do.

That structure is the whole point. A value handed from one ``linalg`` op straight
to the next cannot be scheduled -- the dataflow scheduler puts both computes in
one local schedule and each compute template claims the whole local register file
of its functional unit, so the second one fails address assignment. ``HbmRoundtrip``
(a spyrecode-stage pass, on by default via ``SpyreOptions.hbm_roundtrip``) breaks
that edge by storing the value to HBM after the producer and loading it back
before each consumer, allocating a spill buffer as an extra base address where
the kernel did not already write the value out.

So every variant here sets ``compiles_to_binary``: the pass runs only at the
spyrecode stage, and a variant that stopped at KTIR would exercise none of it.
The KTIR a structural test sees is pre-spyrecode -- ``math.exp``/``math.sqrt``
inside ``linalg.generic`` bodies, no ``ktdp.store`` between them -- which is also
what ``ktir_cpu`` executes for the numerical check, exactly as ``spyreop``'s
variants do. The roundtrip itself is asserted where it happens: the
``hbm-roundtrip.mlir`` lit test over the pass, and the device launch below over
the binary.

``chain3`` earns its place next to ``chain`` by being one compute longer, which
is what makes buffer reuse observable rather than merely implemented: its middle
compute reads the first spilled value and writes the second, so both live in one
buffer and the kernel needs one extra address, not two. Reuse is not an
optimization here -- segment 7 of the address space holds the program, so a
kernel has seven base addresses in total.

fp32 only. ``tl.sqrt``/``tl.exp`` are each ``@_check_dtype(dtypes=["fp32",
"fp64"])`` upstream (``python/triton/language/math.py``) and reject fp16 at
compile time, the same constraint ``spyreop``'s module docstring records.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def _make_inputs(shape, *, dtype="fp32") -> dict:
    """``x`` / zeroed ``output`` buffers of *shape*. Never random, never zero.

    A ramp over [0.1, 1.1), the range ``spyreop`` uses: away from 0, where
    sqrt's relative error grows fastest, and small enough that ``exp`` applied
    twice stays well inside fp32.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    x = (np.arange(total, dtype=np.float32) / total + 0.1).astype(np_dtype)
    return {"x_ptr": x.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def make_inputs(n_elements, DTYPE="fp32", **_unused) -> dict:
    return _make_inputs(n_elements, dtype=DTYPE)


def reference_chain(inputs):
    return np.sqrt(np.exp(inputs["x_ptr"]))


def reference_chain3(inputs):
    return np.exp(np.sqrt(np.exp(inputs["x_ptr"])))


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}


def _stick_1d(dtype: str) -> tuple:
    """Labelled 1D stick layout at *dtype*, ``[n]`` -> ``[ceil(n/S), S]``."""
    stick = sticksize({"p": f"*{dtype}"}, "p")
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


# ---------------------------------------------------------------------------
# Structural checks
#
# What a *pre-spyrecode* structural test can assert: the chain is still a chain,
# with both math ops present and nothing having split it yet. The absence of a
# store between them is the condition HbmRoundtrip exists to remove, so asserting
# it here is asserting that this fixture still poses the problem it is for.
# ---------------------------------------------------------------------------

def _chain_checks(t):
    t.assert_present("math.exp")
    t.assert_present("math.sqrt")
    t.assert_absent("spyreop.exp")
    t.assert_absent("ktdp.hbm_roundtrip_buffers")


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "base": None,
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D `out = sqrt(exp(x))` over a single tile, no distribution loop. "
            "Two chained computes, so one value passes through HBM."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements` and writes "
            "`out = sqrt(exp(x))`. One tile, one core, no loop.\n\n"
            "Two `linalg.generic` bodies in a row is the point: the scheduler "
            "cannot give a second compute registers in the same local schedule, "
            "so `HbmRoundtrip` stores the `exp` result to a spill buffer and "
            "loads it back for the `sqrt`. That buffer is an extra base address "
            "the launcher allocates, which is why this variant's launch passes "
            "three buffers for a two-pointer kernel."
        ),
        "kernel_fn":    kernel.chain_1d_device,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params": {
            "n_elements": [128], "BLOCK_SIZE": [128], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
        },
        "grid":         [1],
        # No tl.program_id distribution loop, so DistributeWork has nothing to
        # place and the presence check would fail on a correct kernel.
        "parallel":     False,
        "compiles_to_binary": True,
        "reference":    reference_chain,
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
        "extra_checks": _chain_checks,
    },

    # The multi-core counterpart, the same way elementwise has 1d_device_grid2:
    # two cores, one tile each, still no loop. It is what exercises the spill
    # buffer's geometry -- one slab per compute tile, anchored at
    # `tile_id * tile_shape[0]` -- which a single-core grid leaves at offset zero
    # and so cannot distinguish from a buffer of exactly one tile.
    "grid2": {
        "base": "default",
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "program-id-1d", "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D `out = sqrt(exp(x))` across two cores, one fp32 stick each. "
            "Two chained computes, so the spill buffer is slabbed per core."
        ),
        "grid":         [2],
        "params": {
            # 128 elements over BLOCK_SIZE=64 is exactly two fp32 sticks, one
            # per core, so core i owns stick i.
            "n_elements": [128], "BLOCK_SIZE": [64], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
        },
        # Inherits parallel=False from the base and means it: two cores each run
        # one tile, so there is still no scf.for for DistributeWork to place.
    },

    # Three computes, so two spilled values and -- because the middle compute
    # reads the first and writes the second -- one buffer serving both.
    "deep": {
        "base": "default",
        "summary": (
            "1D `out = exp(sqrt(exp(x)))` over a single tile. Three chained "
            "computes and two spilled values, sharing one reused buffer."
        ),
        "kernel_fn":    kernel.chain3_1d_device,
        "reference":    reference_chain3,
    },
}
