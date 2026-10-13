"""The device ops a kernel may request with ``tl.spyre_op``, and the fallback each
one is traced as.

``tl.spyre_op(target, x)`` is a request that ``x`` be computed by the device op
``target``, named in full -- ``"spyreop.sigmoid"``. The kernel author writes only
the target; what is traced is the FALLBACK registered here -- the same
computation in plain ``tl`` ops -- inside a ``tts.spyre_op`` region. So a request
means the same thing at every stage:

* the ``ktir`` artifact holds the fallback, inlined and hinted
  ``tts.spyreop_hint``, which any KTIR reader runs as written;
* the ``spyrecode`` stage replaces the fallback, whole, with ``target``.

This module is the one registry. Each entry is registered with
``@spyre_intrinsic(target, dtypes=...)`` on its fallback, and says everything the
frontend checks when the call is traced: the operand count (the fallback's), the
dtypes every operand may have, and the result rule. The IR carries only the
target's name, so the compiler needs no copy of this table: ``LowerSpyreOps``
builds the op by that name, and the op's own verifier checks the operands.

The decorator is the backend's: it is not exported through ``triton.language``,
so kernel authors cannot register targets.

Every fallback widens to fp32 and narrows back. ``tl.exp`` and ``tl.erf`` are
fp32-only in the frontend, and the casts cost nothing on the device path, since
``LowerSpyreOps`` replaces the body they are part of, casts included. At fp32
the casts are no-ops and are not traced.

The frontend reaches this registry through the ``spyre_intrinsic`` codegen hook
(see ``SpyreBackend.get_codegen_implementation``), the channel
``extra_math_dtypes`` already uses, so ``triton.language`` names no backend
module.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import triton
import triton.language as tl


def target_name(target):
    """``target`` as an op name: a string as given, or the ``OPERATION_NAME`` of
    an MLIR Python op class such as ``mlir_ktdp.dialects.spyreop.Sigmoid``."""
    if isinstance(target, str):
        return target
    name = getattr(target, "OPERATION_NAME", None)
    if isinstance(name, str):
        return name
    raise TypeError(f"tl.spyre_op: a target is an op name or an op class, got {target!r}")


def _result_is_operand0(arg_types):
    return [arg_types[0]]


@dataclass(frozen=True)
class Intrinsic:
    """One registered target: the op, its fallback, and what it takes.

    ``fallback`` is a ``@triton.jit`` function of ``arity`` tensors, traced at the
    operands' types. ``dtypes`` is what every operand may be. ``result_types``
    maps the operand types to the declared result types: the op is created before
    its body is traced, so the result types cannot be read off the body, and the
    frontend refuses a fallback that returns anything else.
    """

    target: str
    fallback: Callable
    arity: int
    dtypes: Tuple[tl.dtype, ...]
    result_types: Callable


#: Target name -> Intrinsic, filled by ``spyre_intrinsic``.
REGISTRY = {}


def spyre_intrinsic(target, *, dtypes, result_types=_result_is_operand0):
    """Register the decorated ``@triton.jit`` function as the fallback for the op
    ``target``, taking operands of ``dtypes``. ``result_types`` maps the operand
    types to the result types; by default the one result has operand 0's type.
    Raises on a target that already has a fallback."""
    name = target_name(target)
    dtypes = tuple(dtypes)
    for dtype in dtypes:
        if not isinstance(dtype, tl.dtype):
            raise TypeError(f"tl.spyre_op: {name!r} takes dtypes as tl dtypes, got {dtype!r}")

    def register(fallback):
        if name in REGISTRY:
            raise ValueError(f"tl.spyre_op: target {name!r} has two fallbacks")
        REGISTRY[name] = Intrinsic(
            target=name,
            fallback=fallback,
            arity=len(fallback.arg_names),
            dtypes=dtypes,
            result_types=result_types,
        )
        return fallback

    return register


def lookup(target):
    """The registration for ``target``. Raises ``ValueError`` naming the
    registered targets when there is none."""
    name = target_name(target)
    entry = REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"tl.spyre_op: no fallback is registered for {name!r}; the Spyre "
                         f"backend registers {', '.join(sorted(REGISTRY)) or 'none'}")
    return entry


@spyre_intrinsic("spyreop.gelu", dtypes=(tl.float16, ))
@triton.jit
def gelu(x):
    # The exact form, x * CDF(x), which is what spyreop.gelu is documented as.
    xf = x.to(tl.float32)
    return (0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))).to(x.dtype)


@spyre_intrinsic("spyreop.silu", dtypes=(tl.float16, tl.float32))
@triton.jit
def silu(x):
    xf = x.to(tl.float32)
    return (xf / (1.0 + tl.exp(-xf))).to(x.dtype)


@spyre_intrinsic("spyreop.sigmoid", dtypes=(tl.float16, tl.float32))
@triton.jit
def sigmoid(x):
    xf = x.to(tl.float32)
    return (1.0 / (1.0 + tl.exp(-xf))).to(x.dtype)
