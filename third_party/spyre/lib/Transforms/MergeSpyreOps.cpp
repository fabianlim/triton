//===- MergeSpyreOps.cpp - Many compute ops -> one, by rule --------------===//
//
// A RULE HOST. Each rule matches a group of ops inside a compute body and
// replaces it with the device's own spelling of that group. Adding a rule is a
// pattern plus one line in the pass, and one lit case -- no Passes.td, CMake or
// pipeline change.
//
// THE NAME IS ABOUT THE TARGETS, NOT THE SUBJECT, and it is a debt rather than a
// description. What a rule MATCHES is arith and math -- a divide, a compare, a
// select, a cast -- because that is what a compute body holds before any
// intrinsic has been chosen. What it EMITS is the spyreop spelling, which is
// where "spyre" comes from, and `LowerSpyreOps` next door is the reason the pair
// reads. Should a rule's target ever be an upstream op instead -- a compare and a
// select becoming `arith.maximumf`, say, which is a spelling the device wants and
// `NormalizeForDevice` already produces by another route -- then the name is
// simply wrong and the honest one is about the subject: the compute body. Rename
// then; do not quietly widen what "spyre ops" is taken to mean.
//
// The pair this forms with LowerSpyreOps is two-tier instruction selection, and
// it is the whole reason this pass exists separately:
//
//   MergeSpyreOps   OPPORTUNISTIC, and about a GROUP. Rules run to a fixpoint and
//                   anything no rule claims is left exactly as it was. No
//                   conversion target, because "unmatched" is the normal case.
//   LowerSpyreOps   EXHAUSTIVE, and ONE op to one. A dialect conversion, so an
//                   op it should have handled and could not is reported.
//
// Merge runs FIRST, and that order is the point rather than a detail. After the
// 1:1 selection each op has already been committed to an intrinsic, and the
// multi-op shape a rule wants to see whole is gone: `1.0 / sqrt(x)` is one rule's
// three-op match on the way in, and on the way out it is `reciprocal(sqrt(x))` --
// two intrinsics that have to be un-chosen before the one right one can be. So
// the bigger patterns get first refusal and the 1:1 rules mop up.
//
// WHY THE GENERIC BODY IS THE SCOPE, and not a predicate. Every compute reaches
// this pass as a linalg.generic, so "inside a generic body" is not a guess about
// whether an op is compute -- it is where the compute IS. A rule rooted in a body
// therefore never asks the question: `arith.addi` on an element type in a body is
// compute, while the addi of a loop index or a tile address is not in a body at
// all. Used as a SCOPE this is exact, where the same test used as a per-op
// predicate is a proxy -- which is what made the integer gate in LowerSpyreOps an
// over-claim.
//
// THE TWO THINGS EVERY RULE SHARES, and the reason they are helpers here rather
// than repeated per rule:
//
//   resolveThroughBody   An operand of a body op may be a BLOCK ARGUMENT, in
//            which case the value it really names is the generic's matching `ins`
//            operand. A rule that asks about an operand's definition has to look
//            there or it sees nothing. This is what makes a rule independent of
//            whether anything folded a constant into the body -- see below.
//
//   dropping the operand a rule consumed
//            When a rule stops reading a block argument, its `ins` operand, that
//            argument and its indexing map all have to go. None of that is
//            written here: upstream's
//            populateEraseUnusedOperandsAndResultsPatterns does it, in the same
//            greedy fixpoint. So a rule is purely BODY-LOCAL -- it replaces ops
//            and never touches the operand list -- and the generic comes out with
//            one fewer input for free.
//
// THE ORDERING DEBT THIS PAYS OFF. Matching on the body value alone made the
// reciprocal depend on FoldDataMovementGenerics, whose elementwise fusion happens
// to fold a splat constant into the body and is the only thing in the pipeline
// that does. That was never a property of the problem: `tl.full([M, S], 1.0)`
// arrives as a splat `ins` operand with the numerator a block argument, and
// resolveThroughBody sees the splat whether or not anything folded it. The rule
// matches both forms, so the pass has no ordering constraint against that one at
// all -- which is the difference between a written-down accident and no accident.
//
// The mechanism is upstream's -- the greedy driver, and the operand erasure --
// and the POLICY is ours: which groups of ops the device has one op for, and on
// what evidence.
//
// WHAT MAY BE ADDED HERE, so the pass does not become a junk drawer. A rule
// belongs here only if all three hold.
//
//   1. THE GROUP IS WHAT IS SELECTABLE: no member of it could have been selected
//      on its own. That is the criterion, and NOT the op count -- a rule may well
//      come out two ops long. `arith.cmpf` is the clean case: its result is an
//      `i1`, which no spyreop op produces and the scheduler will not take in a
//      body, so the compare is not selectable by itself at all. What its CONSUMER
//      is decides what it becomes -- a `uitofp` makes the pair one
//      `spyreop.compare`, while an `arith.select` makes the pair a `compare` fed
//      to a `spyreop.select`, still two ops and still one indivisible choice.
//      Where a single op IS independently selectable, the rewrite is a lowering
//      (LowerSpyreOps) or a normalization (NormalizeForDevice), and both already
//      have a home.
//   2. The group and what replaces it compute the same value. A rule is a
//      PREFERENCE between spellings, not a repair; one that changes the computed
//      value is not a rule.
//   3. The preference names the downstream behaviour it is for -- the tool, the
//      component, what it does -- so a reader can check whether it still holds.
//
// `--debug-only=merge-spyre-ops` traces the decisions rather than the control
// flow: one line per candidate root a rule looked at and what it concluded, with
// the operand form named when a match was declined.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "merge-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_MERGESPYREOPS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// The scope, and the one lookup every rule needs
//===----------------------------------------------------------------------===//

/// The `linalg.generic` whose body `op` sits in, or null if it is not in one.
///
/// This is the scope test, and every rule below starts with it. Not a claim about
/// the op -- see WHY THE GENERIC BODY IS THE SCOPE in the header.
linalg::GenericOp computeBodyOf(Operation *op) {
  auto generic = op->getParentOfType<linalg::GenericOp>();
  if (!generic)
    return nullptr;
  // A generic reached through some other region in between is not this op's
  // compute body. Nothing in this tree produces that shape, and a rule reading
  // the wrong operand list would be silent, so it is checked rather than assumed.
  return op->getBlock() == generic.getBlock() ? generic : nullptr;
}

/// What `v` names from OUTSIDE `generic`'s body: the matching `ins` operand when
/// `v` is one of the body's input block arguments, and `v` itself otherwise.
///
/// This is what lets a rule ask about an operand's DEFINITION without caring how
/// the value reached the body. The same `1.0` is a scalar `arith.constant` above
/// the generic if something folded it in, and a splat `ins` operand if nothing
/// did; resolved, both answer the same question.
///
/// SOUND ONLY BECAUSE OF WHAT CALLERS ASK. Every element of the body value maps
/// to the same tensor only for a value that is uniform across the operand, so a
/// caller may ask a resolved value whether it is a SPLAT or a scalar constant --
/// which is exactly what `m_OneFloat` tests -- and may not ask it anything about a
/// particular element. A non-splat constant tensor resolves here too and
/// correctly matches nothing.
///
/// The `outs` block arguments are deliberately not resolved: an `outs` of a
/// `tensor.empty` has no defined value to name, so forwarding it would invite a
/// rule to read one.
Value resolveThroughBody(linalg::GenericOp generic, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner() != generic.getBlock())
    return v;
  unsigned n = arg.getArgNumber();
  if (n >= static_cast<unsigned>(generic.getNumDpsInputs()))
    return v;
  return generic.getDpsInputs()[n];
}

/// Whether spyreop's scalar float intrinsics accept this operand type. The same
/// question LowerSpyreOps asks, and deliberately the same answer: a rule must not
/// select an intrinsic for a type the 1:1 lowering would have refused.
bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type>(type);
}

/// The spyreop predicate computing the same thing as `p`, or nothing when none
/// does.
///
/// ONLY THE ORDERED PREDICATES MAP, and that is a correctness requirement rather
/// than a limitation of this table. `spyreop.compare` documents every predicate as
/// ordered in the IEEE-754 sense -- "the result is zero if either operand is NaN,
/// `notequal` included". So arith's ordered six have exact counterparts, while its
/// UNORDERED six differ from them on precisely the NaN input: `une` is true where
/// either operand is NaN and `spyreop.compare <notequal>` is zero there. Mapping
/// `une` onto it would change the computed value, which rule 2 in the header
/// forbids, so the unordered predicates are declined and their `arith.cmpf`
/// survives.
///
/// `ord`, `uno`, `true` and `false` have no counterpart of any kind: they ask
/// about NaN-ness or about nothing, and a comparison intrinsic answers neither.
std::optional<spyreop::ComparePredicate>
spyrePredicateFor(arith::CmpFPredicate p) {
  switch (p) {
  case arith::CmpFPredicate::OEQ:
    return spyreop::ComparePredicate::Equal;
  case arith::CmpFPredicate::ONE:
    return spyreop::ComparePredicate::NotEqual;
  case arith::CmpFPredicate::OGT:
    return spyreop::ComparePredicate::GreaterThan;
  case arith::CmpFPredicate::OGE:
    return spyreop::ComparePredicate::GreaterEqual;
  case arith::CmpFPredicate::OLT:
    return spyreop::ComparePredicate::LesserThan;
  case arith::CmpFPredicate::OLE:
    return spyreop::ComparePredicate::LesserEqual;
  default:
    return std::nullopt;
  }
}

/// One line per decision, naming the root op and what came of it. Terse because
/// the greedy driver asks each rule about each candidate once per sweep.
void traceDecline(Operation *root, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": no merge (" << why << ")\n");
}

void traceMerge(Operation *root, const llvm::Twine &what) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": " << what << "\n");
}

//===----------------------------------------------------------------------===//
// Rule 1: arith.divf with a numerator of one -> spyreop.reciprocal
//===----------------------------------------------------------------------===//

/// `1.0 / x` is two ops -- the constant and the divide -- and the device does it
/// in one. Replacing them with the unary intrinsic takes the float immediate out
/// of the program entirely, which is what this rule is for: a float immediate
/// reaching a Spyre compute unit is not read back as it was written, so a divide
/// by a rounded one is not the divide that was written. Nothing downstream
/// refuses it -- dbo-opt has no diagnostic for a float immediate operand -- so the
/// rule is what protects the kernel rather than a diagnostic being what reports
/// it. test/fixtures/reduce/meta.py records the observation, and
/// `reduce/softmax_on_stick` is the kernel that depends on it.
///
/// The numerator is read THROUGH the body, so both forms match: a scalar constant
/// hoisted above the generic, and the splat `ins` operand that `tl.full` actually
/// produces. In the second case the block argument goes unused and upstream's
/// erasure takes the operand with it, leaving one generic reading one input --
/// which is the whole of what this rule has to do about the operand list.
struct DivFOneToReciprocal : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    linalg::GenericOp generic = computeBodyOf(op);
    if (!generic)
      // Silent: the overwhelming majority of ops in a module are not here.
      return failure();
    if (!isSpyreOpScalarType(op.getType())) {
      traceDecline(op, "no spyreop intrinsic for this type");
      return failure();
    }

    Value numerator = resolveThroughBody(generic, op.getLhs());
    if (!matchPattern(numerator, m_OneFloat())) {
      traceDecline(op, isa<BlockArgument>(numerator)
                           ? "numerator resolves to a block argument, so it "
                             "names no value this rule can read"
                           : "numerator is not a constant 1.0");
      return failure();
    }

    traceMerge(op, "-> spyreop.reciprocal; the numerator is left to dead-op "
                   "elimination, and its `ins` operand, if it had one, to "
                   "upstream's unused-operand erasure");
    rewriter.replaceOpWithNewOp<spyreop::Reciprocal>(op, op.getType(),
                                                     op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 2: arith.cmpf feeding arith.uitofp -> spyreop.compare
//===----------------------------------------------------------------------===//

/// A comparison whose answer is wanted as a NUMBER rather than as a flag --
/// `(m != 0)` used multiplicatively, the shape a mask arrives in -- is two ops in
/// arith and one on the device.
///
/// The two are an indivisible choice, which is rule 1 in the header. `arith.cmpf`
/// alone produces an `i1`, a type no spyreop op produces and the scheduler will
/// not take in a compute body, so the compare cannot be selected on its own: what
/// consumes the `i1` is what says which device op the pair is. Here the consumer
/// is the widening cast, and `spyreop.compare` is documented as giving its answer
/// "in the width compared rather than as a boolean" -- which is exactly
/// compare-then-cast, so the pair collapses with nothing left over.
///
/// ROOTED ON THE CAST, not on the compare. The consumer is the op that identifies
/// the rule, and rooting there means the match reads DOWN a def-use edge it
/// already holds rather than searching users. The compare is left to dead-op
/// elimination; when it has another reader it simply stays, and both readers are
/// correct, so nothing here asks about its use count.
///
/// `uitofp` and not `sitofp`: an `i1` interpreted as unsigned is 0 or 1, which is
/// the mask wanted. Interpreted as SIGNED it is 0 or -1, so `sitofp` gives
/// -1.0 where the predicate holds and is a different computation -- declined
/// rather than folded in, per rule 2 in the header.
struct CmpFUIToFPToCompare : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    linalg::GenericOp generic = computeBodyOf(op);
    if (!generic)
      return failure();
    if (!isSpyreOpScalarType(op.getType())) {
      traceDecline(op, "no spyreop intrinsic for this result type");
      return failure();
    }

    // Resolved through the body, because a compare could in principle have been
    // computed outside the generic and carried in -- in which case there is no
    // compare here to merge with, and the resolved value is not a cmpf.
    auto cmp =
        resolveThroughBody(generic, op.getIn()).getDefiningOp<arith::CmpFOp>();
    if (!cmp) {
      traceDecline(op, "operand is not an arith.cmpf");
      return failure();
    }
    // The compared width is the intrinsic's whole type, by
    // SameOperandsAndResultType: comparing f32 and casting to f16 is a narrowing
    // this one op cannot express.
    if (cmp.getLhs().getType() != op.getType()) {
      traceDecline(op, "the compared type and the cast's result type differ, "
                       "which one spyreop.compare cannot express");
      return failure();
    }

    std::optional<spyreop::ComparePredicate> predicate =
        spyrePredicateFor(cmp.getPredicate());
    if (!predicate) {
      traceDecline(op, llvm::Twine("spyreop.compare has no counterpart for "
                                   "predicate '") +
                           arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                           "'");
      return failure();
    }

    traceMerge(op, llvm::Twine("arith.cmpf '") +
                       arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                       "' + arith.uitofp -> spyreop.compare; the cmpf is left "
                       "to dead-op elimination");
    rewriter.replaceOpWithNewOp<spyreop::Compare>(
        op, op.getType(), cmp.getLhs(), cmp.getRhs(),
        spyreop::ComparePredicateAttr::get(op.getContext(), *predicate));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct MergeSpyreOpsPass
    : public mlir::triton::spyre::impl::MergeSpyreOpsBase<MergeSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // One line per rule: the only edit a new rule needs outside its own pattern.
    patterns.add<DivFOneToReciprocal, CmpFUIToFPToCompare>(ctx);

    // Upstream's, and load-bearing rather than tidying: a rule that stops reading
    // a block argument leaves an `ins` operand, that argument and its indexing
    // map behind, and this is what removes all three. In the same fixpoint, so a
    // rule never has to see the intermediate state.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createMergeSpyreOpsPass() {
  return std::make_unique<MergeSpyreOpsPass>();
}

} // namespace mlir::triton::spyre
