# The output axis space: how a synthesized op's result gets a physical shape

Design note for `OutputAxisSpace` and the `SourceOpSpec` fields around it, in
`RewriteDescriptorLayout`. It explains one decision — *which axes does the
emitted op's output have* — and why that had to become an explicit choice rather
than staying implicit.

Read `spyre-tensor-layouts.md` first for the phase structure and the vocabulary
(`role`, `dimRoles`, `targetOrder`, marker, stick split). This note goes one
level down, into the part that only the source-op patterns
(`linalg.reduce`, `linalg.matmul`, `linalg.batch_matmul`) touch.

## Phase map

Where the pieces live, for orientation. `+` marks what the batch-dim work added
and `~` what it modified; everything unmarked predates it.

```
  markers (tt.spyre_tensor_layout)
        │
   ┌────▼──────────────────────────────────────────────────────────┐
   │ PHASE 1  physicalize the loads   (RewriteDescriptorLayout.cpp)│
   │  memView + access tiles + ktdp.load retyped to physical shape │
   └────┬───────────────────────────────────────────────┬──────────┘
        │ physicalValues (roots)                        │ physMemViewToMarker
   ┌────▼───────────────────────────────────────────────▼──────────┐
   │ PHASE 2A  analysis, no IR mutation (PhysicalTypeAnalysis.cpp) │
   │  worklist forward from roots; one PhysicalPropagationPattern  │
   │  per op kind → PhysicalTypeInfo{type, marker, transposePerm}  │
   └────┬──────────────────────────────────────────────────────────┘
        │ PhysicalTypeMap ──▶ ctx.physicalTypeAnalysis  (read, const)
        │                 ◀── ctx.physicalTypes.carryForward  (write, 1 op)
   ┌────▼──────────────────────────────────────────────────────────┐
   │ PHASE 2B  rewrite (ContractionSynthesis.cpp)                  │
   │  SourceOpSpec{absorbReduceLoopDims, outputAxes, outputMarker} │
   │  → dispatchSource → classify → reconcileOperandSet            │
   │  → resolveOperand → emitNarrowStage                           │
   └────┬──────────────────────────────────────────────────────────┘
        │ verifyPhysicalTypeAgreement(physicalValues ⊆ PhysicalTypeMap)
        ▼  PHASE 3  erase markers, bridge casts, dead logical views
```

### Phase 1 — physicalize the loads

| item | kind | role | |
|---|---|---|---|
| `PhysicalViewPlan` | struct | per-descriptor decision record | |
| `planPhysicalization` / `materializePhysicalView` | fn | analysis then emission per marker | |
| `rewriteAccessTile`, `rewriteIndirectAccessTile` | fn | tiles at physical block shape | |
| `retypeLoad` | fn | retype the load, **seed** `physicalValues` | |
| `PhysicalValueInfo` | struct | `{marker, transposePerm}` | |
| `PassContext` | struct | shared maps + analysis read pointer + carry-forward handle | `~` |
| `CoordOp`, `applyCoordMap`, `rebuildInIndexDomain` | enum/fn | coordinate-map algebra | |

### Phase 2A — analysis

| item | kind | role | |
|---|---|---|---|
| `PhysicalTypeMap` / `PhysicalTypeInfo` | struct | value → decided type + marker | |
| `runPhysicalTypeAnalysis` | fn | cycle-detecting worklist from the roots | |
| `PhysicalPropagationPattern` | base | per-op type rule | |
| `populatePhysicalPropagationPatterns` | fn | registers the rules; takes `const MarkerByMemView &`, not the `PassContext` | `~` |
| `ReducePropagation` | pattern | Physical iff the induced layout equals the store's marker and tile shape; holds the same `const MarkerByMemView &` | `~` |
| `Store/Matmul/Transpose/Reshape/Broadcast/Elementwise` | patterns | the other rules | |
| `findStoreDestination` → `StoreDestination` | fn | walk to the store a value feeds (used by 2A *and* 2B), over `const MarkerByMemView &` | `+` |
| `canRebuildPhysicalInit` | fn | is the DPS init rebuildable at physical shape | `+` |
| `OutputAxisSpace` | enum | Logical vs Physical output axes | `+` |
| `verifyPhysicalTypeAgreement` | fn | the subset assertion, no exemptions | `~` |
| `PhysicalTypeCarryForward` | class | 2B's only write into the map, one method | `+` |

### Phase 2B — rewrite

| item | kind | role | |
|---|---|---|---|
| `SourceOpSpec` | struct | per-instance contract incl. `outputAxes` / `outputMarker` | `~` |
| `SourceOperandSpec`, `OperandCoords`, `ClassifiedDims`, `OperandPlan` | struct | coords and dim buckets | `~` |
| `buildDimRoles` | fn | number output axes in the given space | `~` |
| `classify` | fn | bucket physical dims into `scatterDims` / `reduceLoopDims` / `opTileDims` | `~` |
| `isFloorCoord` | fn | property: this dim is a stick index | `+` |
| `isScatterDim` | fn | decision: survives ∧ floor coord ∧ Logical | `+` |
| `dispatchSource` | fn | classify → reconcile → emit → replace → carry forward | `~` |
| `reconcileOperandSet`, `resolveOperand`, `emitCountedLoop`, `emitTranspose` | fn | trip counts, permutations, slicing | |
| `emitNarrowStage` | fn | build the op tile and the accumulator | `~` |
| `rebuildPhysicalInit` | fn | re-emit empty/fill at physical shape | `+` |
| `eraseDeadProducers` | fn | drop the dead logical init chain | `+` |
| `RewriteReducePattern` | pattern | reads 2A's verdict; target order from `buildDimRoles` | `~` |
| `RewriteMatmulPattern`, `RewriteBatchMatmulPattern` | pattern | always Logical, `absorbReduceLoopDims=false` | `~` |
| `RewriteStorePattern`, `emitWidenStage` | pattern/fn | widen sink; defers on a predicted-physical producer | `~` |
| `RewriteElementwisePattern`, `RewriteTransposePattern` | pattern | forward retyping, transpose erasure | |

The 2A rules take `const MarkerByMemView &` rather than the `PassContext`, and
`PhysicalTypeCarryForward` is a handle with one method rather than a non-const
map pointer, for the same reason: the phase split is only worth anything if 2A
cannot write 2B's state and 2B cannot rewrite 2A's answers. Both are cases of
handing over the narrowest thing that does the job.

`isFloorCoord` and `isScatterDim` were one predicate named `isFloorDim`, and
`scatterDims`/`reduceLoopDims` were `floorDims`/`loopDims`, which is worth
knowing when reading anything written before this change.

### The two stick-index buckets: scatter vs accumulate

`classify()` puts stick-index dims in one of two buckets, and it is worth being
explicit about what separates them, because it is *not* how they are sliced.
Both get `SliceKind::StickIndex` — one stick per iteration of their own loop —
so the slicing is mechanically identical. What differs is what the loop does
with the slice:

| bucket | which dims | the loop |
|---|---|---|
| `scatterDims` | surviving (`role >= 0`) stick indices | **scatters** — each iteration writes a different slice of the output |
| `reduceLoopDims` | reduced (`role == -1`) dims beyond the first, since `opInnerDim` takes that one | **accumulates** — every iteration folds into the same accumulator |

Naming one of them for the `floordiv` coordinate they *both* carry hid exactly
that, and it stopped even being descriptive once a surviving floor coordinate
could land inside the op tile as a batch dim (the `Physical` space below). A
floor *coordinate* is still a floor coordinate — `isFloorCoord` and
`CoordOp::FloorDiv` keep their names, because they name the coordinate, not a
fate.

The distinction is load-bearing in two places. `RewriteStorePattern` reads it
directly as its two preconditions: an empty `scatterDims` means nothing to
scatter, and a non-empty `reduceLoopDims` means a reduction the store cannot
express. And `absorbReduceLoopDims` folds only the accumulating bucket into the
op tile — absorbing a reduce axis says nothing about where the output axes are,
so `scatterDims` stay outside it either way.

**A naming discrepancy to be aware of:** the 2A/2B vocabulary is this doc's and
`Passes.td`'s, not the driver's. `RewriteDescriptorLayout.cpp` says "Phase 2A" and
never "Phase 2B" — its second block is labelled just "Phase 2", and the
file-header staged-model comment lists only Phase 1 and Phase 3. The code is the
odd one out; reconciling it is unfinished business rather than a subtlety.

## The problem: `role` was two things at once

A `role` answers "which output axis does this physical dim feed". Every source
op numbered those axes **per surviving logical dim**, which silently assumed:

> one surviving logical axis ⟹ one output axis

That holds for every op the pass was built for. It stops holding the moment a
surviving logical axis is *stick-split*, because then one logical axis occupies
**two** physical dims — a stick index and a lane — and both survive.

Concretely, a `[M=64, N=128]` fp16 input stick-tiled on N is physically
`[2, 64, 64]` = (stick index, M, lane). Fold M and N survives, split in two:

| physical dim | logical | survives? |
|---|---|---|
| 0 `N floordiv 64` | N | yes — stick index |
| 1 `M` | M | no — reduced |
| 2 `N mod 64` | N | yes — lane |

Under logical numbering, dims 0 and 2 both get role 0. The accumulator is built
as `accDims[role] = extent`, so they collide: whichever is written last wins and
`accTy` comes out rank-1. The stick-tiled result `tensor<2x64xf16>` that the
store wants is not merely unbuilt, it is **inexpressible**.

The old code was not wrong; it was complete for its inputs. What it lacked was a
way to say that this op's output axes are counted differently.

## The fix: name the space

`OutputAxisSpace` (in `PermutationUtils.h`, beside `CoordOp`) makes the numbering
an explicit choice with two values:

- **`Logical`** — one output axis per surviving *logical* dim. A surviving stick
  index is not an output axis at all: it is sliced to extent 1 or scattered by an
  enclosing loop, bucketed into `scatterDims`, and the accumulator carries the op's
  logical rank. Every matmul-like op is here, and so is a reduce whose result
  acquires no layout of its own.
- **`Physical`** — one output axis per surviving *physical* dim, in physical
  order. A surviving stick index rides along as a **batch dim** of the emitted op
  and gets an accumulator axis of its own. This is the space in which one logical
  axis can occupy two output axes.

The property that makes this cheap: **roles stay unique in both spaces.** Only
the numbering of survivors changes, never whether a dim survives. So
`accDims[role] = extent` and the transpose-permutation uniqueness assumption are
untouched — the accumulator code did not have to change at all.

`canonicalAxes` keeps answering *whether* a logical dim survives, which is
space-independent. `buildDimRoles` takes the space and numbers the survivors.

## Who decides, and why it is not the pattern

The space is a property of the **op instance**, not the op kind. The same
`linalg.reduce` belongs in either space depending on something outside itself:
whether the descriptor its result is stored to declares exactly the layout the
input's surviving stick structure induces.

So the pattern does not choose. **Phase 2A decides and the pattern reads the
verdict**, which is the direction `df4b8b1d0` set ("decide physical types before
rewriting, and delete the guards that guessed").

`ReducePropagation` computes it: reducing a logical axis away deletes the
physical dims sourced from it, so the operand's layout *induces* an output layout
— the surviving physical dims in order, coord ops and args intact, axes
renumbered. If the output descriptor's marker declares that induced layout (all
three coordinate arrays **and** the access-tile shape), the result is physical
under that marker and the op is in the `Physical` space. Otherwise `Logical`.

This is also what the pre-existing comment already said the rule had to be: *"a
result is physical only under a layout of its own, which a reduce result acquires
only when the output descriptor is annotated."* The change made that measurable
instead of asserted.

## What `SourceOpSpec` carries, and why each field is separate

```cpp
struct SourceOpSpec {
  llvm::SmallVector<SourceOperandSpec> operands;
  unsigned logicalRank;
  bool absorbReduceLoopDims = false;                        // per op KIND
  OutputAxisSpace outputAxes = OutputAxisSpace::Logical;    // per op INSTANCE
  triton::SpyreTensorLayoutOp outputMarker;                 // set iff Physical
  ... emitOp
};
```

Two absorption-ish flags sit here and they are **not** the same question:

- `absorbReduceLoopDims` — "can this op kind fold its whole *reduce* axis set into one
  emitted op?" True for `linalg.reduce`, whose `dimensions` takes a sorted list;
  false for matmul, which contracts one axis at a time. A property of the op
  kind, fixed at the pattern.
- `outputAxes` — "does *this* op's output get physical axes?" A property of the
  instance, supplied by Phase 2A.

They were briefly one flag. Merging them is wrong in both directions: a reduce
can absorb its reduce axes while still being `Logical` (no output marker), and
the two answers come from different places at different times.

`outputMarker` is the layout the result carries, and is null in the `Logical`
space — where the result has no layout of its own and any physical form is built
afterwards by the store's widen stage.

## Invariants this had to re-establish

1. **`targetOrder` ⟷ `opTileDims`, 1:1 in physical order.** A surviving stick
   index can now be an op-tile dim, so the exclusion rule had to stay identical
   on both sides. `RewriteReducePattern` calls `buildDimRoles` itself and filters
   with the same `isScatterDim` in the same space that `classify()`
   will use. The two cannot disagree because they ask one predicate with one
   argument — rather than one side re-deriving the rule.

2. **Idempotence.** The pre-existing guard (a `dimensions` entry `>= logicalRank`
   means the op was already rewritten) cannot see the new form, because
   `dimensions = [1]` is *within* logical range. The added guard is the direct
   statement instead: the result already being in `ctx.physicalValues` *is* "this
   is final". Verified by running the pass on its own output — no diff.

3. **Phase 2A's subset invariant.** `verifyPhysicalTypeAgreement` asserts that
   every value Phase 2 found physical, the analysis predicted. This matters
   because Phase 2B reads *absence* from that map as "genuinely logical" and
   commits to a rewrite on that basis, so an under-claiming analysis would
   silently mis-lower.

   A value Phase 2 *mints* (a source pattern's replacement) postdates the
   analysis and so cannot be in the map on its own. Rather than exempt it, the
   pattern that mints it hands the analysis the one fact it is missing: this new
   value carries the decision already made for the value it replaced.
   `PhysicalTypeCarryForward::carryForward` copies that entry across, and
   containment then holds **by construction** — with no exemption in the check,
   which keeps its full strength for Phase 1's roots and for everything the
   elementwise and transpose patterns record.

   The carry-forward is deliberately the *only* write path from 2B into the map:
   `ctx.physicalTypeAnalysis` stays a pointer to const, and the handle holds the
   map privately behind one named method whose stated precondition is that the
   replaced value has an entry. It does — that entry's presence is what put the
   pattern in the `Physical` space in the first place — so absence is a drift
   between those two, and asserts rather than quietly writing nothing.

   Note this assertion is live in the default build
   (`TritonRelBuildWithAsserts`), not debug-only in practice. It is the library's
   only `#ifndef NDEBUG`, and not an `LLVM_DEBUG` on purpose: `LLVM_DEBUG` is
   gated on a runtime flag, so it would run only when someone asked, whereas this
   has to run unasked in CI. A plain `assert()` would drop the condition but not
   the walk over `physicalValues` that feeds it.

4. **Store ordering.** `RewriteStorePattern` could fire first, build a widen loop
   for a logical result, and then have the reduce replace that value underneath
   it (`expected 2 offset values, got 1`). It now defers while its data tile is
   *predicted* physical but not yet registered — the analysis-shaped sibling of
   the existing `pendingElementwiseRetype` deferral.

5. **The init operand.** A physical output means the accumulator is a different
   *physicalization*, not a slab of the logical one, so it is rebuilt at physical
   shape: `tensor.empty`, or `linalg.fill` re-emitted over one. The fill is
   re-emitted rather than dropped because a reduction's payload reads its `outs`
   and the neutral element is what makes that well defined; dropping it stays
   `DropReductionInitFill`'s job on the binary path. Only those two producers
   qualify, and Phase 2A asks the same predicate, so the decision and the
   emission cannot drift.

## Alternatives rejected

- **A third array (`accAxisOfTarget`) beside `targetOrder`.** Would be identical
  to `targetOrder` for every existing op — a whole indexing concept whose only
  distinct use is one new case. The enum says the same thing with no new array.
- **Absorb surviving scatter dims unconditionally for reduce.** A simpler flag with
  wrong answers: `middle_axis` would emit a rank-3 result against a logical rank-2
  store, needing a bridge path that does not exist. Two existing lit tests
  regress. The decision genuinely depends on the output descriptor.
- **Special-case the reduce in `emitNarrowStage`, or mutate the reduce in place.**
  Provably safe in this one configuration (all-`WholeBlock`, identity
  permutation, so no slicing), but it carves a per-op path through shared code —
  the thing to avoid, since the next op with a stick-split surviving axis would
  need its own.
- **Loosening `verifyPhysicalTypeAgreement`**, whether broadly or by exempting
  minted values. Something has to give here — with nothing done, the assertion
  fires on this case — but every form of exemption spends the check to buy it,
  and a flag on the entry saying "do not check me" spends it in the place hardest
  to notice. Carrying the decision forward pays the same debt by making the
  invariant true instead of unenforced. Verified the check can still fire:
  suppressing a legitimate analysis entry trips it.

## Extending this

A new source op needs `absorbReduceLoopDims` for its kind and, if its result can carry
a layout of its own, a `PhysicalPropagationPattern` that states which output
layout its operand's layout induces. It does **not** need to touch the
accumulator or the permutation code — that is what keeping roles unique bought.

The case a per-logical-dim role provably cannot express is covered by the rank-4
case in `test/Conversion/rewrite-descriptor-layout-reduce-batch-dim.mlir`: two
batch dims, a stick index and an untouched dim. If that test ever needs a special
case to pass, the space abstraction has sprung a leak.
