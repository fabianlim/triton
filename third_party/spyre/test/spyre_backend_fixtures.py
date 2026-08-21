"""Shared fixtures for the Spyre backend tests.

Imported by both the pytest suite (``test_spyrecode_stage.py``) and the
python-driven lit tests under ``test/python/``, so the kernels, the signature and
the ttir+ktir lowering helper exist once rather than twice. Not itself a test
module: lit only collects ``.py`` under ``test/python/``, and pytest only collects
``test_*.py``, so neither picks this up.
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


