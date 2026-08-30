# RewriteDescriptorLayout: Reconciling Kernels with Device Tensor Layouts

`RewriteDescriptorLayout` reconciles a logical Triton kernel against a device
tensor layout that is already stickified. The layout comes from a user
annotation, `tl.spyre_tensor_layout`, on each tensor descriptor; the pass does
not choose or infer layouts, it only rewrites the kernel body so every op sees
the physical shape the annotation demands. It runs after
`LowerDescriptorMemory`, `LowerScalarLoad`, and `LowerComputeOps` have already
lowered the kernel to KTDP. See `rewrite-descriptor-layout-refactor.md` for the
implementation plan.

## Definitions and assumptions

A **stick** is the hardware's contiguous innermost memory unit, fixed at
**128 bytes**. Elements per stick is derived from the element size, not a
constant: `128 / itemsize`, which gives **32 for fp32** and **64 for fp16**
(`STICK_BYTES` and `sticksize()` in `test/utils.py`). Layouts and this doc
always write the stick size as `S`; the lit fixtures use `S=64` on f32
tensors as a test convenience, not a hardware value — do not generalize from
it.

**Physicalization** (stick expansion) splits one logical dim `d` into two
physical dims, `d floordiv S` and `d mod S`. It is a coordinate-map rewrite of
how a dim is indexed, not data movement — the underlying buffer is
unchanged. Physical extents follow from the split: a `floordiv` dim gets
extent `ceil(N/S)` (covers a partial boundary stick when `S` does not divide
`N`), a `mod` dim gets extent `S`, and an identity dim keeps its logical
extent. Worked example: a logical `[M, N]` tensor stick-split on `N` becomes
physical `[ceil(N/S), M, S]`.

The **lane** is the physical dim carrying the `mod` role. It is found by
searching for that role; it is not necessarily the innermost dim.

The annotation itself, `tl.spyre_tensor_layout(desc, layout)`, takes one
entry per physical dim, each either `src` (identity), `(src, "floordiv", S)`,
or `(src, "mod", S)`, where `src` is the logical source dim. It lowers to
three i64 arrays on the descriptor: `phys_src`, `phys_op` (0 = identity, 1 =
floordiv, 2 = mod), and `phys_arg` (the `S` operand, unused for identity). The
docstring example (`spyre_tensor_layout` in `python/triton/language/core.py`):

```python
# desc describes a [M, N] logical tensor; physical layout is
# [ceil(N/64), M, 64] -- N stick-split, M and the stick dim untouched.
tl.spyre_tensor_layout(desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])
```

Assumptions the pass relies on: the author supplies the layout and stick
size, the pass never infers them; a logical dim spans at most two physical
dims — one `floordiv` and one `mod` — enforced by
`SpyreTensorLayoutOp::verify()`; and a `mod` dim cannot be sub-stick, i.e. its
extent must equal `S`.

## Phases

The pass runs in three phases, stated as a contract per phase.

**Phase 1 — physicalize annotated descriptors.**
Input: descriptors carrying layout markers, and the logical `ktdp.load` chain
below them.
Lowering: for each marker, rebuild the memory view and access tile at
physical shape, retype the load, and forward-retype the elementwise chain
that consumes it, stopping at the first op that cannot absorb the new rank.
Output: a physical-shaped load chain up to that boundary; markers stay live.

**Phase 2 — resolve remaining shape mismatches.**
Input: the Phase 1 output, markers still live.
Lowering: for every op with a physicalized operand, apply the case check
below — retype in place, or convert by emitting a loop or a transpose.
Output: every op's inputs and outputs agree in physical shape. Markers stay
live throughout, since the case check reads each operand's coordinate map from
its marker.

**Phase 3 — cleanup.** Erase the markers and any now-dead bridge casts.

## The case check

Phase 2 examines each op with a physicalized operand, compares the physical
shape reaching its inputs against the physical shape demanded of its
outputs, and acts on the relation:

| Case | Input -> output | Action |
|---|---|---|
| 1 | both physical, same shape | Retype in place. No loop, no slicing. |
| 2 | one side logical, the other physical | Convert: emit a loop. |
| 3 | both physical, different shapes | Convert per shape relation (transpose). |

Loop emission is a consequence of a shape mismatch, not a property of the op.
An op contributes exactly one thing to the decision: whether it can absorb an
arbitrary physical shape.

| Op | Absorbs arbitrary physical shape | Why |
|---|---|---|
| `linalg.reduce` | Yes | `dimensions` is a list, so a stick-split reduce axis is expressible in one op. |
| `ktdp.store` | Yes | `AnyTensor`; the verifier checks only that data-tile and access-tile shapes agree. |
| `linalg.matmul`, `linalg.batch_matmul` | No | Contracts exactly one `K` axis, so a split `K` requires cross-stick accumulation. |

`linalg.reduce` and `ktdp.store` are therefore one category, differing only
in which case their shapes land in — there is no separate sink path. Phase
1's forward retype applies this same absorption check to the elementwise
chain.

Case 1 illustrated: a reduce over a stick-split axis, physical `[2, 64, 64]`
reduced to `[64]`.

```mlir
%r = linalg.reduce ins(%phys : tensor<2x64x64xf32>) outs(%acc : tensor<64xf32>) dimensions = [0, 2]
```

This is legal because `dimensions` is `DenseArrayStrictlySorted` with no
adjacency or trailing-position requirement.

## Shape polymorphism of the memory ops

`ktdp.load` and `ktdp.store` take `AnyTensor` and their verifiers check only
that data-tile and access-tile shapes agree — no rank cap, no dim placement
rule (`KTDP_LoadOp`, `KTDP_StoreOp` in `KTDP.td`). By contrast `tt.dot` caps
operand rank at 2 or 3 at parse time, and `linalg.reduce` requires a strictly
sorted `dimensions` list. Memory ops constrain almost nothing, compute ops
constrain tightly — this is why case discrimination is a shape question
rather than an op-type question.

## Idempotence

Phase 2 runs under a greedy pattern driver that re-enqueues an op whenever a
neighbour it feeds or consumes is rewritten, so ops are visited repeatedly
until a fixpoint. Every pattern's match condition must therefore be
falsified by its own rewrite. For converting cases this follows from the
rank change; for case 1, which retypes in place without changing rank, the
pattern must test whether the work is already done — a reduce whose
`dimensions` already covers the physical reduce dims does not match. Running
the pass on its own output is a no-op.

## Rejected inputs

| Input | Rejected because |
|---|---|
| A `mod` dim whose block extent is smaller than `S` | A `mod` dim cannot be sub-stick. |
| A stick-split on the indirect (row) dim of a gather | The indirect dim indexes rows one at a time and cannot be split across sticks. |
| Operands sharing a stick-split contraction axis where not all carry a marker | Cross-stick accumulation requires every operand on that axis to agree on the split. |
| A physical-to-physical shape relation that is not a transpose | Case 3 only recognizes a transpose; any other relation has no defined conversion. |
