#!/usr/bin/env python3
"""``tl.spyre_op`` from the kernel line to the module dbo-opt receives.

Every case registers its own fallback for one real target, ``spyreop.sigmoid``,
over an empty copy of the registry, so nothing here changes when a target is
added and no registration outlives its test. The claims, one per stage:

* tracing builds a ``tts.spyre_op`` holding the bound fallback, which its region
  verifier admits;
* the ``ktir`` artifact holds that fallback alone, inlined and hinted
  ``tts.spyreop_hint`` -- no ``tts`` op, which a KTIR reader could not parse;
* the ``spyrecode`` pipeline fuses the hinted ops into one body and replaces it
  with exactly one op, the target, and no hint is left for dbo-opt.

The last is reproduced with the registered pipeline rather than through the
``spyrecode`` stage itself, which shells out to dbo-opt -- CLAUDE.md's
"Reproducing that module". And the refusals the frontend owns: a target with no
fallback registered, a dtype the registration does not take, the wrong arity, a
fallback returning other than operand 0's type, and the registrations the
decorator refuses -- among them a target that is not a registered op.

Numerical coverage of the real fallbacks, through ``ktir_cpu``, is the
``spyreop`` fixture's.
"""

import re

import pytest
import triton
import triton.language as tl

# The module compile_to_ttir's SpyreBackend reads its fallbacks from.
from backend import intrinsics
from utils import compile_to_ttir, make_ktir_mod


@triton.jit
def _request(x_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc.store([0], tl.spyre_op(OP, x_desc.load([0])))


def _trace(kernel, op, dtype):
    signature = {"x_ptr": f"*{dtype}", "out_ptr": f"*{dtype}", "N": "constexpr", "OP": "constexpr"}
    return compile_to_ttir(kernel, signature, {"N": 128, "OP": op})


def _ktir(ttir, tmp_path):
    path = tmp_path / "kernel.ttir"
    path.write_text(ttir)
    return make_ktir_mod(path, grid=[1])


def _spyrecode(mod):
    """The module the ``spyrecode`` stage hands dbo-opt, in symbolic mode."""
    from triton._C.libtriton import ir, spyre
    pm = ir.pass_manager(mod.context)
    spyre.passes.ttir_to_ktdp.add_spyrecode_pipeline(pm)
    pm.run(mod, "spyrecode")
    return str(mod)


#: The target every case requests. A real op, so selection can build it; what
#: the fallback computes is the test's own, so the two stages differ on purpose.
TARGET = "spyreop.sigmoid"


@pytest.fixture
def register(monkeypatch):
    """``spyre_intrinsic`` for ``TARGET`` at fp16 and fp32, over an empty copy of
    the registry, so a test's registrations are gone when it ends."""
    monkeypatch.setattr(intrinsics, "REGISTRY", {})

    def register(target=TARGET, dtypes=(tl.float16, tl.float32)):
        return intrinsics.spyre_intrinsic(target, dtypes=dtypes)

    return register


@triton.jit
def func1(x):
    xf = x.to(tl.float32)
    return (xf * xf + xf).to(x.dtype)


@triton.jit
def func2(x):
    xf = x.to(tl.float32)
    return (tl.exp(xf) - 1.0).to(x.dtype)


@pytest.mark.parametrize("fallback", [func1, func2], ids=["func1", "func2"])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_call_site_reaches_one_intrinsic(register, fallback, dtype, tmp_path):
    register()(fallback)
    ttir = _trace(_request, TARGET, dtype)
    # The call site, verified, holding the fallback as a call the `ttir` stage
    # inlines.
    assert f'tts.spyre_op "{TARGET}"' in ttir
    assert "tts.spyreop_yield" in ttir

    mod = _ktir(ttir, tmp_path)
    ktir = str(mod)
    assert "tts.spyre_op" not in ktir and "tts.spyreop_yield" not in ktir
    assert f'tts.spyreop_hint = {{id = 0 : i64, target = "{TARGET}"}}' in ktir
    assert "tt.call" not in ktir

    spyrecode = _spyrecode(mod)
    elem = {"fp16": "f16", "fp32": "f32"}[dtype]
    # Fused into one body and selected: one op, the target, at the operand's type.
    selected = re.findall(r"(spyreop\.\w+) %\S+ : (\w+)", spyrecode)
    assert selected == [(TARGET, elem)], selected
    # The fallback went whole, casts included, and so did every hint.
    assert "tts.spyreop_hint" not in spyrecode
    assert "math." not in spyrecode
    assert "arith.extf" not in spyrecode and "arith.truncf" not in spyrecode


def _raises(kernel, op, dtype, match):
    from triton.compiler.errors import CompilationError
    with pytest.raises(CompilationError) as exc:
        _trace(kernel, op, dtype)
    assert match in str(exc.value)


@triton.jit
def _constant_request(out_ptr, N: tl.constexpr, OP: tl.constexpr):
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc.store([0], tl.spyre_op(OP, tl.full([N], 1.0, tl.float32)))


def test_constant_operand_is_refused(register, tmp_path, capfd):
    register()(func1)
    ttir = compile_to_ttir(_constant_request, {"out_ptr": "*fp32", "N": "constexpr", "OP": "constexpr"},
                           {"N": 128, "OP": TARGET})
    with pytest.raises(Exception):
        _ktir(ttir, tmp_path)
    assert f'tl.spyre_op("{TARGET}"): operand 0 is a constant' in capfd.readouterr().err


def test_target_without_a_fallback_is_refused():
    # A real op the backend registers no fallback for.
    assert "spyreop.softplus" not in intrinsics.REGISTRY
    _raises(_request, "spyreop.softplus", "fp16",
            "tl.spyre_op: no fallback is registered for 'spyreop.softplus'; the Spyre backend registers")


def test_dtype_the_intrinsic_does_not_take_is_refused(register):
    register()(func1)
    _raises(_request, TARGET, "bf16", f"tl.spyre_op('{TARGET}'): the intrinsic takes ['fp16', 'fp32'], not bf16")


def test_wrong_arity_is_refused(register):
    register()(func1)

    @triton.jit
    def two_operands(x_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[N])
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
        x = x_desc.load([0])
        out_desc.store([0], tl.spyre_op(OP, x, x))

    _raises(two_operands, TARGET, "fp32", f"tl.spyre_op('{TARGET}'): takes 1 tensor operand(s), got 2")


def test_fallback_returning_other_types_is_refused(register):

    @register()
    @triton.jit
    def widens(x):
        return x.to(tl.float32)

    _raises(_request, TARGET, "fp16", f"tl.spyre_op('{TARGET}'): the registered fallback returns")


def test_the_backend_registers_its_targets():
    assert set(intrinsics.REGISTRY) == {"spyreop.gelu", "spyreop.silu", "spyreop.sigmoid"}
    for name, entry in intrinsics.REGISTRY.items():
        assert entry.target == name and entry.arity == 1


def test_an_op_class_registers_as_its_name(register):

    class Sigmoid:
        OPERATION_NAME = TARGET

    register(Sigmoid)(func1)
    assert intrinsics.lookup(Sigmoid) is intrinsics.lookup(TARGET)


def test_a_dtype_that_is_not_a_tl_dtype_is_refused(register):
    with pytest.raises(TypeError, match="takes dtypes as tl dtypes, got 'fp16'"):
        register(dtypes=("fp16", ))


@pytest.mark.parametrize("target", ["spyreop.sigmiod", "sigmoid"])
def test_a_target_that_is_not_a_registered_op_is_refused(register, target):
    with pytest.raises(ValueError, match=f"target '{target}' is not a registered op"):
        register(target)


def test_a_fallback_taking_no_operand_is_refused(register):

    @triton.jit
    def nullary():
        return tl.full([128], 1.0, tl.float32)

    with pytest.raises(ValueError, match="takes no operand"):
        register()(nullary)


def test_a_second_fallback_for_one_name_is_refused(register):
    register()(func1)
    with pytest.raises(ValueError, match=f"target '{TARGET}' has two fallbacks"):
        register()(func2)


@pytest.mark.parametrize("backend", ["cuda", "hip", None])
def test_raises_off_backend(monkeypatch, backend):
    # The guard fires before the builtin touches ``_semantic`` or its operands.
    from triton.backends.compiler import GPUTarget
    from triton.language import target_info
    target = None if backend is None else GPUTarget(backend=backend, arch=1, warp_size=1)
    monkeypatch.setattr(target_info, "current_target", lambda: target)
    with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
        tl.spyre_op(TARGET, None, _semantic=object())
