"""Chained-compute kernels: several computes in one kernel, one after another.

What these exercise is not the arithmetic -- it is deliberately the simplest
chain that type-checks -- but the fact that there is more than one compute. The
dataflow scheduler forms one local schedule per SSA-connected region of compute
and each compute template claims the whole local register file of its unit, so a
value handed straight from one ``linalg`` op to the next cannot be scheduled.
``HbmRoundtrip`` breaks that edge by storing the value to HBM and loading it
back; these are the kernels whose KTIR has such an edge to break.

Every kernel here is loop-free and one tile per core, for the same reason
``elementwise``'s and ``spyreop``'s device variants are: dbo-opt rejects the
``scf.for`` a program-id distribution loop outlines.

- :func:`chain_1d_device` -- two computes, ``sqrt(exp(x))``.
- :func:`chain3_1d_device` -- three computes, ``exp(sqrt(exp(x)))``. One more
  than the pair above, which is what makes buffer reuse observable: the middle
  compute reads the first spill and writes the second, so one buffer serves both.
"""

import triton
import triton.language as tl


@triton.jit
def chain_1d_device(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = sqrt(exp(x))`` over exactly one tile, no distribution loop."""
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    out_desc.store([offset], tl.sqrt(tl.exp(x)))


@triton.jit
def chain3_1d_device(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = exp(sqrt(exp(x)))`` -- three computes, so two spilled values."""
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    out_desc.store([offset], tl.exp(tl.sqrt(tl.exp(x))))
