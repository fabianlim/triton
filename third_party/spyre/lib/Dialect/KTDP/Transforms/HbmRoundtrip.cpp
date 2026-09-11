//===- HbmRoundtrip.cpp - Break compute-to-compute edges through HBM ------===//
//
// The dataflow scheduler forms one *local schedule* per maximal SSA-connected
// region of compute, and every compute template claims the whole local register
// file of the functional unit it lands on. So two `linalg` ops joined by a
// tensor SSA value are put in one schedule and the second one has no registers
// left:
//
//   error: Failed to allocate "SFP_LRFREG" memory: Allocation of 128 bytes
//          exceeds "SFP_LRFREG" capacity (2048 bytes available, 2048 bytes
//          already allocated)
//
// A compute cannot hand its result to another compute. What splits the region
// is memory: a `ktdp.store` ends one schedule and a `ktdp.load` begins the
// next. Hand-written reference KTIR states this directly -- a chained
// normalisation or softmax passes every intermediate through HBM and gets one
// `local_schedule_N` per compute.
//
// This pass performs that split. For each tensor value produced by a `linalg`
// op and consumed by another one, it stores the value to HBM straight after the
// producer and loads it back straight before each consumer.
//
// Scope: generic and fill only
// ----------------------------
// A function is left completely untouched unless every `linalg` op in its entry
// block is a `linalg.generic` or a `linalg.fill`. The gate is on the *op set*,
// not on what the compute does: a generic carrying reduction iterators is in
// scope and gets a roundtrip like any other -- the hand-written reference `exx2`
// is one. What is excluded is every *named* `linalg` op except `fill`:
// `linalg.reduce`, `linalg.matmul`, `linalg.broadcast`, `linalg.transpose`.
//
// This pass is a temporary stand-in for a real spill/schedule decision, and
// narrowing it to the ops it was written against is what keeps that honest rather
// than having it silently mis-handle a compute it never saw. The consequence,
// stated narrowly: today's LowerComputeOps lowers `tt.reduce` to a *named*
// `linalg.reduce`, so a kernel that comes through that path is skipped.
//
// A `linalg.fill` is a `linalg` op but it is not a compute: the scheduler's
// named-op allowlist is `linalg.{add,mul,sub,max,min,reduce,generic,yield}` and a
// fill is not among them, so it does not become a compute group of its own and
// the edge from a fill to the `outs` of a generic is not an edge this pass has to
// break. A fill is therefore
// excluded from the spill scan in both roles, and handled like the
// `tensor.empty` it writes into: cloned into each compute group that reads it.
//
// Where the buffer comes from
// ---------------------------
// Two cases, in order of preference:
//
//   1. The value is *already* stored to HBM by the kernel, and that store
//      precedes the consumer. Softmax's exponentials are like this: they are a
//      real output of the kernel and also the input of two later computes. Then
//      no buffer is needed -- the consumer reads back from the buffer the
//      kernel already wrote, through a clone of that store's access tile.
//
//   2. Otherwise a fresh spill buffer, which becomes a new `index` argument of
//      the entry function. The caller supplies its address the same way it
//      supplies the kernel's own pointer addresses (see MaterializeBaseAddresses
//      and `SpyreOptions.base_addresses`), and the launcher allocates it -- the
//      buffers this pass creates are reported on the module as
//      `ktdp.hbm_roundtrip_buffers` -- one space-separated `12x64x64xf32` per
//      buffer, in argument order -- for exactly that.
//
// Buffers are reused. A buffer whose value has been read by its last consumer
// is available to a later spill of the same tile type. What reuse economizes is
// not memory -- a spill buffer is one tile per compute tile, a few KiB -- but
// *address slots*: the baked base-address policy MaterializeBaseAddresses
// implements hands out one segment per address and segment 7 holds the program,
// so a kernel has seven in total and its own pointers already take some. That is
// a property of the policy, not of the hardware, and it is also why this pass
// runs only where that policy does (see `_make_spyrecode`).
//
// Buffer geometry
// ---------------
// A spilled value is one compute tile's worth of data and every compute tile
// runs the same program, so the buffer holds one slab per tile: the tile shape
// with its leading dimension multiplied by the grid, dense row-major, and the
// access tile anchored at `linear_tile_id * tile_shape[0]` along that dimension.
// Same rank as the tile, so no reshape is involved either way. With a grid of
// one the buffer is exactly the tile and the anchor is zero.
//
// One group, one copy of everything
// ---------------------------------
// Splitting the compute is only half the job. The scheduler moves each group's
// operations into a schedule module of its own, so anything two groups read
// cannot be moved into either, and a kernel written when there was one group
// shares a great deal. Three separate assertions come out of that, and the pass
// answers all three by duplicating rather than by reasoning about ownership:
//
//   - a `ktdp.construct_access_tile` read by both a store and a load:
//     "StoreOp found before any LoadOp" (ComputeGroupExtraction.cpp:570).
//     Every store and load gets its own, even where two are identical
//     operand-for-operand.
//   - the `tensor.empty` (or the `linalg.fill` over it) on `outs`, or a
//     `ktdp.load` of an input two computes read: "Operation should have no uses
//     left" (:472). Cloned in front of the compute that reads it, which also
//     settles *position* -- groups are spans of the block, so a `tensor.empty`
//     sitting inside the first group's span is in that group even when only the
//     second reads it.
//   - the index arithmetic behind an access tile, shared by two of them: the
//     same assertion, reached through an `arith.divsi` rather than through
//     anything tensor-shaped. So each memory operation's whole address cone is
//     rebuilt, down to the base address, which is what the hand-written
//     reference chains do by re-reading the tile id and re-constructing their
//     views per compute.
//
// A direct consequence: **CSE must not run after this pass.**
// `construct_access_tile` and `tensor.empty` are Pure, so CSE would merge the
// copies back together and reinstate every one of those assertions.
// `_make_spyrecode` in third_party/spyre/backend/compiler.py runs
// canonicalize/CSE ahead of this pass for that reason.
//
// Where it looks
// --------------
// The entry block of each public function. A value produced inside a region
// (an `scf.for` body, say) is out of scope -- the scheduler rejects such a loop
// on its own account, before the question of where the value lives arises.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_HBMROUNDTRIP
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// One compute-to-compute edge, keyed by the value that crosses it.
struct Spill {
  Value value;
  /// The `linalg` ops that read `value`, in block order, deduplicated: one
  /// load is emitted per consumer op however many operands it reads `value`
  /// through.
  SmallVector<Operation *> consumers;
  /// A `ktdp.store` of `value` that already precedes every consumer, if the
  /// kernel wrote this value to HBM itself. Non-null means case 1 above: read
  /// back from that buffer instead of allocating one.
  mlir::ktdp::StoreOp existingStore;
};

/// A spill buffer: one new `index` argument, one memory view over it, and the
/// tile type it holds (which is what makes it reusable by a later spill).
struct Buffer {
  RankedTensorType tileType;
  /// The view's own shape -- the tile shape with dim 0 scaled by the grid.
  SmallVector<int64_t> viewShape;
  BlockArgument baseArg;
  Value view;
};

/// Row-major strides for `shape`.
static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

struct HbmRoundtripPass
    : public mlir::triton::ktdp::impl::HbmRoundtripBase<HbmRoundtripPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<std::string> reported;
    func::FuncOp reportedFor;
    bool rewrote = false;

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (!funcOp.isPublic() || funcOp.getBody().empty())
        continue;
      SmallVector<std::string> buffers;
      FailureOr<bool> touched = roundtrip(funcOp, buffers);
      if (failed(touched))
        return signalPassFailure();
      rewrote |= *touched;
      if (buffers.empty())
        continue;
      // The buffer list is reported on the module and consumed positionally,
      // as one list of extra addresses for one entry function. Two entry
      // functions each needing buffers would need two lists, and silently
      // concatenating them would hand the second function's addresses to the
      // first.
      if (reportedFor) {
        funcOp.emitError()
            << "HbmRoundtrip: both @" << reportedFor.getName() << " and @"
            << funcOp.getName()
            << " are public functions needing spill buffers, and the buffer "
               "list is reported per module; compile one entry function at a "
               "time";
        return signalPassFailure();
      }
      reportedFor = funcOp;
      reported = std::move(buffers);
    }

    if (!rewrote)
      return;
    if (!reported.empty())
      module->setAttr("ktdp.hbm_roundtrip_buffers",
                      StringAttr::get(&getContext(),
                                      llvm::join(reported, " ")));
    // Privatizing left the originals of everything it cloned behind, unused.
    // Swept rather than left in place because an operation belonging to no
    // compute group is one more thing for the scheduler to place.
    mlir::triton::ktdp::cleanupDeadOps(module);
  }

private:
  //===--------------------------------------------------------------------===//
  // Step 1: find the compute-to-compute edges
  //===--------------------------------------------------------------------===//

  /// Whether `op` is a compute, i.e. a `linalg` op that becomes a compute group
  /// of its own and so cannot hand its result to another one. A `linalg.fill` is
  /// not: the scheduler's named-op allowlist does not include it, so it is
  /// materialized into whichever group reads it rather
  /// than scheduled on a functional unit. Spilling a fill would therefore put a
  /// store and a load around something that was never a schedule boundary.
  static bool isCompute(Operation *op) {
    return isa<linalg::LinalgOp>(op) && !isa<linalg::FillOp>(op);
  }

  /// Whether every `linalg` op in `entry` is one of the two this pass is written
  /// for. A named op other than `fill` -- `linalg.reduce` from a `tt.reduce`, a
  /// `linalg.matmul`, a `linalg.broadcast` -- means the function is left exactly
  /// as it was found: this is a temporary pass, and a deliberate no-op is a better
  /// answer than a roundtrip placed around a compute whose scheduling it has not
  /// been checked against. Note this says nothing about iterator types: a generic
  /// that reduces is admitted like any other generic.
  static bool onlyGenericsAndFills(Block &entry) {
    for (Operation &op : entry)
      if (isa<linalg::LinalgOp>(&op) &&
          !isa<linalg::GenericOp, linalg::FillOp>(&op))
        return false;
    return true;
  }

  /// The edges of `entry`, in producer order. `order` maps each op of the block
  /// to its position, which is how "precedes" is decided throughout.
  static SmallVector<Spill>
  collectSpills(Block &entry, const DenseMap<Operation *, int64_t> &order) {
    SmallVector<Spill> spills;
    DenseMap<Value, unsigned> indexOf;

    for (Operation &op : entry) {
      if (!isCompute(&op))
        continue;
      // Every operand, not just `ins`: an `outs` fed by a compute is the same
      // unschedulable edge. UnaliasLinalgOuts normally leaves a tensor.empty (or
      // a fill over one) there, so in practice this loop finds `ins`.
      for (Value operand : op.getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || producer->getBlock() != &entry)
          continue;
        if (!isCompute(producer))
          continue;
        auto it = indexOf.find(operand);
        if (it == indexOf.end()) {
          indexOf[operand] = spills.size();
          spills.push_back(Spill{operand, {&op}, nullptr});
          continue;
        }
        auto &consumers = spills[it->second].consumers;
        if (!llvm::is_contained(consumers, &op))
          consumers.push_back(&op);
      }
    }

    // Case 1: the kernel already stores this value, ahead of every consumer.
    for (Spill &spill : spills) {
      int64_t firstConsumer = order.at(spill.consumers.front());
      for (Operation *user : spill.value.getUsers()) {
        auto store = dyn_cast<mlir::ktdp::StoreOp>(user);
        if (!store || store->getBlock() != &entry)
          continue;
        if (order.at(store.getOperation()) < firstConsumer) {
          spill.existingStore = store;
          break;
        }
      }
    }
    return spills;
  }

  //===--------------------------------------------------------------------===//
  // Step 2: the linear compute-tile id, and the grid it is linearized against
  //===--------------------------------------------------------------------===//

  /// Product of the function's `grid` attribute, or 1 when it has none.
  static int64_t gridSize(func::FuncOp funcOp) {
    auto grid = funcOp->getAttrOfType<ArrayAttr>("grid");
    if (!grid)
      return 1;
    int64_t total = 1;
    for (auto extent : grid.getAsRange<IntegerAttr>())
      total *= extent.getInt();
    return total;
  }

  /// The single linear tile index for a grid of any rank, or null when the grid
  /// holds one tile (in which case every slab offset is zero and no value is
  /// needed). `ktdp.get_compute_tile_id` is variadic -- one result per grid
  /// dimension -- so a multi-axis grid is folded row-major:
  /// `((p0 * g1) + p1) * g2 + p2`.
  FailureOr<Value> linearTileId(func::FuncOp funcOp, Block &entry,
                               int64_t gridTotal) {
    if (gridTotal == 1)
      return Value();

    mlir::ktdp::GetComputeTileIdOp tileIdOp;
    for (Operation &op : entry)
      if (auto candidate = dyn_cast<mlir::ktdp::GetComputeTileIdOp>(&op)) {
        tileIdOp = candidate;
        break;
      }
    if (!tileIdOp) {
      funcOp.emitError()
          << "HbmRoundtrip: grid has " << gridTotal
          << " compute tiles but the function never locates itself in it "
             "(no ktdp.get_compute_tile_id), so a per-tile spill slab cannot "
             "be addressed";
      return failure();
    }

    auto grid = funcOp->getAttrOfType<ArrayAttr>("grid");
    if (tileIdOp->getNumResults() != grid.size()) {
      funcOp.emitError() << "HbmRoundtrip: ktdp.get_compute_tile_id returns "
                         << tileIdOp->getNumResults()
                         << " index/indices but the grid has rank "
                         << grid.size();
      return failure();
    }

    OpBuilder builder(tileIdOp);
    builder.setInsertionPointAfter(tileIdOp);
    Location loc = tileIdOp.getLoc();
    Value linear = tileIdOp->getResult(0);
    for (unsigned axis = 1; axis < tileIdOp->getNumResults(); ++axis) {
      int64_t extent =
          cast<IntegerAttr>(grid.getValue()[axis]).getInt();
      Value scale = arith::ConstantIndexOp::create(builder, loc, extent);
      linear = arith::MulIOp::create(builder, loc, linear, scale);
      linear =
          arith::AddIOp::create(builder, loc, linear, tileIdOp->getResult(axis));
    }
    return linear;
  }

  //===--------------------------------------------------------------------===//
  // Step 3: buffer assignment, with reuse
  //===--------------------------------------------------------------------===//

  /// Assign a buffer to each spill that needs one, reusing a buffer whose
  /// value has already been read by its last consumer. Fills `buffers` and
  /// `bufferOf` (an index into `buffers` per spill, or -1 for the spills that
  /// read back from an existing store).
  static void assignBuffers(ArrayRef<Spill> spills,
                            const DenseMap<Operation *, int64_t> &order,
                            SmallVectorImpl<Buffer> &buffers,
                            SmallVectorImpl<int> &bufferOf, int64_t gridTotal) {
    bufferOf.assign(spills.size(), -1);

    // Buffers whose value is still to be read, as (buffer index, position of
    // the last consumer). A buffer leaves this list once the producer being
    // placed comes after that position.
    SmallVector<std::pair<unsigned, int64_t>> live;
    SmallVector<unsigned> pool;

    for (auto [i, spill] : llvm::enumerate(spills)) {
      if (spill.existingStore)
        continue;
      int64_t producerAt = order.at(spill.value.getDefiningOp());

      // `<=`, not `<`: a buffer whose last reader is the very compute now
      // producing a spill is free for that spill. The load is emitted before
      // that compute and the store after it, so the read has happened by the
      // time the write lands -- which is the ordinary case in a chain, where
      // each compute reads the previous value and writes the next.
      for (unsigned k = 0; k < live.size();) {
        if (live[k].second <= producerAt) {
          pool.push_back(live[k].first);
          live.erase(live.begin() + k);
        } else {
          ++k;
        }
      }

      auto tileType = cast<RankedTensorType>(spill.value.getType());
      unsigned chosen = buffers.size();
      auto reusable = llvm::find_if(pool, [&](unsigned b) {
        return buffers[b].tileType == tileType;
      });
      if (reusable != pool.end()) {
        chosen = *reusable;
        pool.erase(reusable);
      } else {
        SmallVector<int64_t> viewShape(tileType.getShape());
        viewShape[0] *= gridTotal;
        buffers.push_back(Buffer{tileType, std::move(viewShape), nullptr,
                                 nullptr});
      }

      bufferOf[i] = static_cast<int>(chosen);
      int64_t lastConsumer = 0;
      for (Operation *consumer : spill.consumers)
        lastConsumer = std::max(lastConsumer, order.at(consumer));
      live.push_back({chosen, lastConsumer});
    }
  }

  //===--------------------------------------------------------------------===//
  // Step 4: give each compute its own copy of every tensor it reads
  //===--------------------------------------------------------------------===//

  /// Once the stores and loads are in, each `linalg` op is its own compute
  /// group, and the scheduler moves a group's operations into a schedule module
  /// of their own. A tensor-producing op shared by two groups therefore cannot
  /// go anywhere:
  ///
  ///   ComputeGroupExtraction.cpp:472: Assertion `op->use_empty() &&
  ///   "Operation should have no uses left"' failed.
  ///
  /// Three shared producers occur in practice, all from earlier passes that had
  /// no reason to avoid it -- one compute per kernel made sharing invisible:
  ///
  ///   - the `tensor.empty` UnaliasLinalgOuts puts on `outs`, which canonicalize
  ///     and CSE then merge across computes because it is Pure and identical;
  ///   - a `linalg.fill` over such a `tensor.empty`, for the same reason and with
  ///     the same remedy: it is Pure, so two identical fills become one, and it
  ///     is not a compute (see :func:`isCompute`), so it belongs *inside* the
  ///     group that reads it rather than passing through HBM;
  ///   - a `ktdp.load` of a kernel input read by more than one compute, which is
  ///     how a normalisation reads its values three times.
  ///
  /// All are cloned immediately before the compute that reads them, whether or
  /// not they are shared. Position matters as much as sharing does: groups are
  /// spans of the block, so a `tensor.empty` sitting between the first load and
  /// the first store belongs to the first group even when its only reader is the
  /// second. Cloning in front of the reader settles both questions at once, and
  /// the original is swept afterwards if nothing else wants it.
  ///
  /// The clone is a whole cone, not one operation: a `linalg.fill` reads both the
  /// `tensor.empty` it writes into and the scalar it writes, and leaving either
  /// of those shared moves the assertion one operation down rather than removing
  /// it. The recursion is :func:`cloneCone`, the same one the address cones use,
  /// which also means a cloned `ktdp.load` arrives with an address cone of its
  /// own.
  static LogicalResult privatizeComputeInputs(Block &entry) {
    for (Operation &op : llvm::make_early_inc_range(entry)) {
      // Generics only. A fill is privatized as part of the cone of the compute
      // that reads it; visiting it in its own right would clone its
      // `tensor.empty` in front of the *fill*, which is a position that may
      // belong to an earlier group.
      if (!isa<linalg::GenericOp>(&op))
        continue;
      for (OpOperand &operand : op.getOpOperands()) {
        Value value = operand.get();
        if (!isa<RankedTensorType>(value.getType()))
          continue;
        Operation *producer = value.getDefiningOp();
        if (!producer || producer->getBlock() != &entry)
          continue;

        if (isa<tensor::EmptyOp, linalg::FillOp, mlir::ktdp::LoadOp>(producer)) {
          OpBuilder builder(&op);
          DenseMap<Value, Value> cloned;
          operand.set(cloneCone(builder, value, cloned));
          continue;
        }
        // Every compute-to-compute edge is a `ktdp.load` by the time this runs,
        // and the entry block holds no `linalg` op other than a generic or a fill
        // (:func:`onlyGenericsAndFills`), so what is left here is a tensor
        // producer this pass has not been taught to place -- a reshape, say.
        // Reported rather than cloned blindly, because getting it wrong is an
        // assertion inside dbo-opt that names no IR the caller wrote.
        return op.emitError()
               << "HbmRoundtrip: " << producer->getName()
               << " produces a tensor a compute reads, and this pass can only "
                  "clone a tensor.empty, a linalg.fill or a ktdp.load into the "
                  "compute group that reads it";
      }
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Step 5: give each memory operation its own address cone
  //===--------------------------------------------------------------------===//

  /// Clone `value`'s defining op, and recursively everything that op reads, in
  /// front of the insertion point. Stops at block arguments -- the base
  /// addresses, which every group legitimately shares because they are the
  /// function's own inputs. `cloned` memoizes within one cone so a value read
  /// twice by the same cone is cloned once.
  static Value cloneCone(OpBuilder &builder, Value value,
                         DenseMap<Value, Value> &cloned) {
    if (isa<BlockArgument>(value))
      return value;
    auto known = cloned.find(value);
    if (known != cloned.end())
      return known->second;

    Operation *def = value.getDefiningOp();
    IRMapping mapping;
    for (Value operand : def->getOperands())
      mapping.map(operand, cloneCone(builder, operand, cloned));
    Operation *clone = builder.clone(*def, mapping);
    for (auto [original, copy] :
         llvm::zip(def->getResults(), clone->getResults()))
      cloned[original] = copy;
    return cloned[value];
  }

  /// Rebuild the address computation of every `ktdp.load` and `ktdp.store` so
  /// that no two of them share any of it.
  ///
  /// This is what the hand-written reference chains do explicitly: each compute
  /// re-reads `ktdp.get_compute_tile_id`, re-constructs its memory views and
  /// re-constructs its access tiles, sharing only the base addresses. The reason
  /// is compute-group extraction -- the scheduler moves each group's operations
  /// into a schedule module of its own, so an operation two groups read cannot
  /// be moved into either and it asserts on the leftover use. Index arithmetic
  /// is not exempt: a kernel whose descriptor offset is computed once and used
  /// by two access tiles hits it through that `arith.divsi`, not through
  /// anything tensor-shaped.
  ///
  /// Cloning per memory operation rather than per group is deliberate: it needs
  /// no notion of which group an operation belongs to, and being over-generous
  /// costs only dead IR, which the sweep afterwards removes.
  static void privatizeAddressCones(Block &entry) {
    SmallVector<Operation *> memoryOps;
    for (Operation &op : entry)
      if (isa<mlir::ktdp::LoadOp, mlir::ktdp::StoreOp>(&op))
        memoryOps.push_back(&op);

    for (Operation *memoryOp : memoryOps) {
      // The access tile only. A store's other operand is the tensor the compute
      // produced, which is the one thing that must NOT be duplicated.
      auto accessTile = isa<mlir::ktdp::LoadOp>(memoryOp)
                            ? cast<mlir::ktdp::LoadOp>(memoryOp).getAccessTile()
                            : cast<mlir::ktdp::StoreOp>(memoryOp).getAccessTile();
      OpBuilder builder(memoryOp);
      DenseMap<Value, Value> cloned;
      memoryOp->replaceUsesOfWith(accessTile,
                                  cloneCone(builder, accessTile, cloned));
    }
  }

  //===--------------------------------------------------------------------===//
  // Step 6: rewrite
  //===--------------------------------------------------------------------===//

  /// Returns whether the function was rewritten at all, so the caller knows
  /// whether the dead-op sweep has anything to do.
  FailureOr<bool> roundtrip(func::FuncOp funcOp,
                            SmallVectorImpl<std::string> &reported) {
    Block &entry = funcOp.getBody().front();

    // Nothing at all on a function whose compute this pass was not written
    // against -- see the scope note at the top of the file. Checked before
    // anything else so the function is left byte-for-byte as it arrived.
    if (!onlyGenericsAndFills(entry))
      return false;

    DenseMap<Operation *, int64_t> order;
    {
      int64_t position = 0;
      for (Operation &op : entry)
        order[&op] = position++;
    }

    SmallVector<Spill> spills = collectSpills(entry, order);
    if (spills.empty())
      return false;

    // A rank-0 spill has no dimension to slab per compute tile, and nothing in
    // this pipeline produces one (every reduce leaves at least a stick).
    for (const Spill &spill : spills) {
      if (spill.existingStore)
        continue;
      auto tileType = dyn_cast<RankedTensorType>(spill.value.getType());
      if (!tileType || tileType.getRank() == 0 ||
          !tileType.hasStaticShape()) {
        return spill.value.getDefiningOp()->emitError()
               << "HbmRoundtrip: cannot spill a value of type "
               << spill.value.getType()
               << " to HBM; a spill buffer needs a statically shaped tile of "
                  "rank >= 1";
      }
    }

    int64_t gridTotal = gridSize(funcOp);
    FailureOr<Value> tileId = linearTileId(funcOp, entry, gridTotal);
    if (failed(tileId))
      return failure();

    SmallVector<Buffer> buffers;
    SmallVector<int> bufferOf;
    assignBuffers(spills, order, buffers, bufferOf, gridTotal);

    // Arguments first, then the views over them: adding an argument invalidates
    // nothing already built, whereas building a view over an argument that does
    // not exist yet is impossible.
    Location loc = funcOp.getLoc();
    auto indexType = IndexType::get(funcOp.getContext());
    for (Buffer &buffer : buffers)
      buffer.baseArg = entry.addArgument(indexType, loc);
    funcOp.setType(FunctionType::get(funcOp.getContext(),
                                     entry.getArgumentTypes(),
                                     funcOp.getFunctionType().getResults()));

    auto memorySpace = mlir::ktdp::MemorySpaceAttr::get(
        funcOp.getContext(), mlir::ktdp::MemorySpaceKind::global,
        /*ct_id=*/-1);
    OpBuilder builder(&entry, entry.begin());
    for (Buffer &buffer : buffers)
      buffer.view = mlir::triton::ktdp::buildMemoryView(
          builder, loc, buffer.baseArg, buffer.viewShape,
          rowMajorStrides(buffer.viewShape), /*dynSizes=*/{},
          /*dynStrides=*/{}, buffer.tileType.getElementType(), memorySpace);

    for (auto [i, spill] : llvm::enumerate(spills)) {
      if (spill.existingStore) {
        // Read back from the buffer the kernel already wrote. One clone of the
        // store's access tile per consumer, never the store's own tile: see the
        // fresh-access-tile note at the top of this file.
        for (Operation *consumer : spill.consumers) {
          OpBuilder loadBuilder(consumer);
          Operation *tile = loadBuilder.clone(
              *spill.existingStore.getAccessTile().getDefiningOp());
          Value loaded = mlir::ktdp::LoadOp::create(
              loadBuilder, spill.value.getLoc(), spill.value.getType(),
              tile->getResult(0));
          consumer->replaceUsesOfWith(spill.value, loaded);
        }
        continue;
      }

      Buffer &buffer = buffers[bufferOf[i]];
      ArrayRef<int64_t> tileShape = buffer.tileType.getShape();

      Operation *producer = spill.value.getDefiningOp();
      OpBuilder storeBuilder(producer);
      storeBuilder.setInsertionPointAfter(producer);
      Value storeTile =
          buildSlabTile(storeBuilder, spill.value.getLoc(), buffer, tileShape,
                        *tileId, gridTotal);
      mlir::ktdp::StoreOp::create(storeBuilder, spill.value.getLoc(),
                                  spill.value, storeTile);

      for (Operation *consumer : spill.consumers) {
        OpBuilder loadBuilder(consumer);
        Value loadTile =
            buildSlabTile(loadBuilder, spill.value.getLoc(), buffer, tileShape,
                          *tileId, gridTotal);
        Value loaded = mlir::ktdp::LoadOp::create(
            loadBuilder, spill.value.getLoc(), spill.value.getType(), loadTile);
        consumer->replaceUsesOfWith(spill.value, loaded);
      }
    }

    if (failed(privatizeComputeInputs(entry)))
      return failure();
    privatizeAddressCones(entry);

    for (const Buffer &buffer : buffers)
      reported.push_back(describe(buffer));
    return true;
  }

  /// A fresh `ktdp.construct_access_tile` selecting this compute tile's slab of
  /// `buffer`, anchored at `tileId * tileShape[0]` along dimension 0.
  static Value buildSlabTile(OpBuilder &builder, Location loc,
                             const Buffer &buffer, ArrayRef<int64_t> tileShape,
                             Value tileId, int64_t gridTotal) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<Value> indices(tileShape.size(), zero);
    if (gridTotal != 1) {
      Value slab = arith::ConstantIndexOp::create(builder, loc, tileShape[0]);
      indices[0] = arith::MulIOp::create(builder, loc, tileId, slab);
    }
    return mlir::triton::ktdp::buildAccessTile(builder, loc, buffer.view,
                                               tileShape, indices);
  }

  /// One buffer as the launcher needs it: MLIR's own shape spelling,
  /// `12x64x64xf32`, which carries the extent of every dimension and the element
  /// type and nothing else.
  ///
  /// A string, and all of them joined into one string attribute, because that is
  /// what Python can already read: `ir.module.get_operation().get_str_attr` is an
  /// existing binding, where an array of dictionaries would need a new one
  /// written and kept in step. The width is deliberately absent -- the backend
  /// already reads a width off an element type's own spelling for the kernel's
  /// pointer arguments (`_elem_bytes` in backend/compiler.py), and a second
  /// answer to the same question is a second thing that can disagree.
  static std::string describe(const Buffer &buffer) {
    std::string spec;
    llvm::raw_string_ostream os(spec);
    for (int64_t extent : buffer.viewShape)
      os << extent << "x";
    buffer.tileType.getElementType().print(os);
    return spec;
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createHbmRoundtripPass() {
  return std::make_unique<HbmRoundtripPass>();
}
} // namespace mlir::triton::ktdp
