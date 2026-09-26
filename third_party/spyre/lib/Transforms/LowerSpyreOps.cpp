//===- LowerSpyreOps.cpp - Select spyreop intrinsics from compute bodies --===//
//
// INSTRUCTION SELECTION for the Spyre device: arith and math ops become the
// spyreop intrinsic that does the same thing. A RULE HOST -- adding a case is
// one pattern plus one line in the pass and one lit case.
//
// KTIR -> KTIR, which is why this sits in Transforms/ rather than in
// Conversion/TritonToKTIR/: that directory's criterion is a `tt` source
// dialect, and nothing here reads a tt op. The input is KTIR whose computes are
// linalg.generic, and the output is the same KTIR with intrinsics in the
// bodies.
//
// ONE PASS FOR ALL SELECTION, and the two kinds of rule are not two tiers:
//
//   one op to one   math.sqrt -> spyreop.sqrt, and so on. Type-driven.
//   a GROUP to one  `1.0 / x` -> spyreop.reciprocal; a compare and a cast ->
//                   spyreop.compare. The group is what is selectable: no
//                   member of it could have been selected alone.
//
// They share one greedy pattern set, and a group rule wins where both could
// apply because it is more specific: a `divf` with a constant-one numerator is
// claimed by the reciprocal rule and the realdiv rule never sees it. No order
// is declared anywhere. Splitting these across two passes is what an earlier
// shape did, and it bought a standing question -- which pass claims this op --
// for nothing.
//
// EVERYTHING UNMATCHED FLOWS THROUGH, and that is deliberate. There is no
// conversion target and nothing is reported: an op with no device form reaches
// the backend, which is the component that actually knows what it can take, and
// it refuses there. So this pass has no notion of an illegal input -- it
// selects what it can and leaves the rest exactly as it found it. A type or a
// predicate this file does not handle is a silent pass-through by design; see
// WHAT IS NOT SELECTED below for the list and what each one costs.
//
// WHAT THIS PASS RELIES ON ITS PREDECESSOR FOR. A rule matches ops in ONE body,
// and ConvertElementwiseToLinalg gives every tensor-level op a body of its own
// -- so a group spanning two tensor ops arrives spread over two generics unless
// something fused them. FuseComputeAndDataMovement is that something, and its
// `i1` clause exists for exactly this: the compare rule sees its pair only
// because that pass brought them together, and the reciprocal's constant is a
// scalar in the body for the same reason. Stated as a contract between adjacent
// passes rather than left to be discovered -- and `resolveThroughBody` below
// keeps the reciprocal working on the unfused form too, so for that rule the
// dependency is about whether it FIRES, never about whether it is correct.
//
// THE GENERIC BODY IS THE SCOPE for a group rule, and not a predicate. Every
// compute reaches this pass as a linalg.generic, so "inside a body" is not a
// guess about whether an op is compute -- it is where the compute is. A rule
// rooted in a body never asks: an `arith.addi` on an element type in a body is
// compute, and the addi of a loop index or a tile address is not in a body at
// all. The integer 1:1 rules use the same test as a per-op PREDICATE, which is
// a proxy and is known to be one; see isInsideLinalgGeneric.
//
// The mechanism is upstream's -- the greedy driver, and the unused-operand
// erasure -- and the POLICY is ours: which ops and which groups the device has
// a form for, and on what evidence.
//
// WHAT IS NOT SELECTED, and what each costs. None of these is a diagnostic, per
// EVERYTHING UNMATCHED FLOWS THROUGH above; Passes.td carries the same list
// with the reasoning and this is the short form.
//
//   A 1:1 rule does not fire and the op goes to the backend as arith:
//     - a float width with no intrinsic (f64, bf16)
//     - an integer width with no intrinsic, or integer add/mul outside a body
//     - anything still tensor-typed, which only ConvertElementwiseToLinalg not
//       having run can produce
//
//   A group rule declines and its members stay as they were:
//     - a divide whose numerator is not a constant one, or whose constant is
//       the denominator: the realdiv rule takes it, which is right
//     - a numerator resolving to a NON-SPLAT constant tensor: reading an
//       operand through the body is sound only for a uniform value
//     - an UNORDERED cmpf predicate. spyreop.compare is ordered for every
//       predicate, `notequal` included, so `une` and its siblings differ from
//       it on exactly the NaN input, and mapping them across would change the
//       computed value. Declined for correctness, not for coverage.
//     - `ord`/`uno` and the constant predicates: no counterpart at all
//     - `arith.sitofp` where `uitofp` was wanted: an `i1` read as signed is 0
//       or -1, so the cast gives -1.0 where the predicate holds
//     - a compare at one width cast to another: spyreop.compare has
//       SameOperandsAndResultType and cannot do both
//
//   The compare declines are the ones to know: each leaves an `i1` inside a
//   body, which the backend will not take either. It is refused there rather
//   than here, which is the point of the rule above -- but the diagnostic names
//   neither this pass nor the predicate that caused it.
//
// `--debug-only=lower-spyre-ops` traces the group rules' decisions rather than
// the control flow: one line per candidate looked at and what came of it, with
// the operand form named when a match was declined. The 1:1 rules say nothing
// -- there is one per op, and the output IR shows whether it fired.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "lower-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// Shared questions
//===----------------------------------------------------------------------===//

/// Whether spyreop's scalar float intrinsics accept this operand type.
bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type>(type);
}

/// The bit width of `type` if it is a scalar integer, or 0 otherwise -- e.g.
/// for a not-yet-scalarized tensor of integers.
unsigned getScalarIntBitWidth(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy ? intTy.getWidth() : 0;
}

/// Whether this op is (transitively) inside a linalg.generic body.
///
/// A PROXY, used only by the integer 1:1 rules, which have no better test:
/// plain scalar integer add and mul are also loop indices, offsets and tile
/// addressing, and there is no exact test separating those from scalarized
/// integer compute. The proxy over-claims -- `tl.arange(0, N) * stride` is an
/// `arith.muli` that will be inside a body once scalarized -- and it holds
/// today only because the tensor-of-pointers `tt.load` path is unimplemented.
/// Stated rather than glossed. The group rules use the enclosing generic as a
/// SCOPE instead, through computeBodyOf, which asks a different and exact
/// question.
bool isInsideLinalgGeneric(Operation *op) {
  return op->getParentOfType<linalg::GenericOp>() != nullptr;
}

/// The `linalg.generic` whose body `op` sits directly in, or null.
///
/// The scope for a group rule. Not a claim about the op -- see THE GENERIC BODY
/// IS THE SCOPE in the header.
linalg::GenericOp computeBodyOf(Operation *op) {
  auto generic = op->getParentOfType<linalg::GenericOp>();
  if (!generic)
    return nullptr;
  // A generic reached through some other region in between is not this op's
  // compute body. Nothing in this tree produces that shape, and a rule reading
  // the wrong operand list would be silent, so it is checked rather than
  // assumed.
  return op->getBlock() == generic.getBlock() ? generic : nullptr;
}

/// What `v` names from OUTSIDE `generic`'s body: the matching `ins` operand
/// when `v` is one of the body's input block arguments, and `v` itself
/// otherwise.
///
/// This is what lets a rule ask about an operand's DEFINITION without caring
/// how the value reached the body. The same `1.0` is a scalar `arith.constant`
/// above the generic once FuseComputeAndDataMovement has folded the splat in,
/// and a splat `ins` operand if that pass has not run; resolved, both answer
/// the same question.
///
/// SOUND ONLY BECAUSE OF WHAT CALLERS ASK. A body value corresponds to the same
/// tensor element only for a value uniform across the operand, so a caller may
/// ask a resolved value whether it is a SPLAT or a scalar constant -- which is
/// exactly what `m_OneFloat` tests -- and may not ask it anything about a
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

/// The spyreop predicate computing the same thing as `p`, or nothing when none
/// does.
///
/// ONLY THE ORDERED PREDICATES MAP, and that is a correctness requirement
/// rather than a limitation of this table. `spyreop.compare` documents every
/// predicate as ordered in the IEEE-754 sense -- "the result is zero if either
/// operand is NaN, `notequal` included". So arith's ordered six have exact
/// counterparts, while its UNORDERED six differ from them on precisely the NaN
/// input: `une` is true where either operand is NaN and `spyreop.compare
/// <notequal>` is zero there. Mapping `une` onto it would change the computed
/// value.
///
/// `ord`, `uno` and the two constant predicates have no counterpart of any
/// kind: they ask about NaN-ness or about nothing, and a comparison intrinsic
/// answers neither.
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

void traceDecline(Operation *root, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": no group match (" << why
                          << ")\n");
}

void traceMatch(Operation *root, const llvm::Twine &what) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": " << what << "\n");
}

//===----------------------------------------------------------------------===//
// One op to one: the unary float math ops
//===----------------------------------------------------------------------===//

/// math.sqrt/exp/rsqrt -> the matching spyreop intrinsic.
///
/// No body scope and no discriminator: these ops only ever appear in real
/// floating-point compute, never in address or index arithmetic, so every
/// scalar occurrence is one to select, inside a generic body or not. An operand
/// type with no intrinsic does not match and flows through.
template <typename Source, typename Target>
struct SelectUnaryFloat : public OpRewritePattern<Source> {
  using OpRewritePattern<Source>::OpRewritePattern;

  LogicalResult matchAndRewrite(Source op,
                                PatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.template replaceOpWithNewOp<Target>(op, op.getType(),
                                                 op.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A group to one: arith.divf with a numerator of one -> spyreop.reciprocal
//===----------------------------------------------------------------------===//

/// `1.0 / x` is two ops -- the constant and the divide -- and the device does
/// it in one. Replacing them with the unary intrinsic takes the float immediate
/// out of the program entirely, which is what this rule is for: a float
/// immediate reaching a Spyre compute unit is not read back as it was written,
/// so a divide by a rounded one is not the divide that was written. Nothing
/// downstream refuses it, so the rule is what protects the kernel rather than a
/// diagnostic being what reports it. test/fixtures/reduce/meta.py records the
/// observation, and `reduce/softmax_on_stick` is the kernel that depends on it.
///
/// MORE SPECIFIC THAN SelectArithDivF below, and that is how it wins: both
/// could match, this one claims the op, and the realdiv rule never sees it.
///
/// The numerator is read THROUGH the body, so both forms match: the scalar
/// constant FuseComputeAndDataMovement hoists above the generic, and the splat
/// `ins` operand `tl.full` produces before it has run. In the second case the
/// block argument goes unused and upstream's erasure takes the operand, its
/// block argument and its indexing map with it -- the whole of what this rule
/// has to do about the operand list.
struct SelectReciprocal : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    linalg::GenericOp generic = computeBodyOf(op);
    if (!generic)
      return failure(); // Silent: most ops in a module are not in a body.
    if (!isSpyreOpScalarType(op.getType()))
      return failure();

    Value numerator = resolveThroughBody(generic, op.getLhs());
    if (!matchPattern(numerator, m_OneFloat())) {
      traceDecline(op, isa<BlockArgument>(numerator)
                           ? "numerator resolves to a block argument, so it "
                             "names no value this rule can read"
                           : "numerator is not a constant 1.0");
      return failure();
    }

    traceMatch(op, "-> spyreop.reciprocal; the numerator is left to dead-op "
                   "elimination, and its `ins` operand, if it had one, to "
                   "upstream's unused-operand erasure");
    rewriter.replaceOpWithNewOp<spyreop::Reciprocal>(op, op.getType(),
                                                     op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// One op to one: arith.divf -> spyreop.realdiv
//===----------------------------------------------------------------------===//

/// Unconditional on the numerator. A constant one is not a different lowering,
/// it is a group the device has one op for, and SelectReciprocal claims it
/// first by being more specific. A divide reaching this rule already means no
/// group rule wanted it.
struct SelectArithDivF : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::RealDiv>(op, op.getType(), op.getLhs(),
                                                  op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A group to one: arith.cmpf feeding arith.uitofp -> spyreop.compare
//===----------------------------------------------------------------------===//

/// A comparison whose answer is wanted as a NUMBER rather than as a flag -- `(m
/// != 0)` used multiplicatively, the shape a mask arrives in -- is two ops in
/// arith and one on the device.
///
/// The two are an indivisible choice. `arith.cmpf` alone produces an `i1`, a
/// type no spyreop op produces and the backend will not take in a compute body,
/// so the compare cannot be selected on its own: what consumes the `i1` is what
/// says which device op the pair is. Here the consumer is the widening cast,
/// and `spyreop.compare` is documented as giving its answer "in the width
/// compared rather than as a boolean" -- which is exactly compare-then-cast, so
/// the pair collapses with nothing left over.
///
/// ROOTED ON THE CAST, not on the compare. The consumer is the op that
/// identifies the rule, and rooting there means the match reads DOWN a def-use
/// edge it already holds rather than searching users. The compare is left to
/// dead-op elimination; when it has another reader it stays, and both readers
/// are correct, so nothing here asks about its use count.
///
/// `uitofp` and not `sitofp`: an `i1` read as unsigned is 0 or 1, which is the
/// mask wanted. Read as SIGNED it is 0 or -1, so `sitofp` gives -1.0 where the
/// predicate holds and is a different computation.
struct SelectCompare : public OpRewritePattern<arith::UIToFPOp> {
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
    // SameOperandsAndResultType: comparing f32 and casting to f16 is a
    // narrowing this one op cannot express.
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

    traceMatch(op, llvm::Twine("arith.cmpf '") +
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
// One op to one: the integer ops
//===----------------------------------------------------------------------===//

/// arith.addi -> spyreop.addi32toi32 / addi64toi64, inside a generic body only.
///
/// The body test here is the PROXY described at isInsideLinalgGeneric, not the
/// exact scope the group rules use. A width with no intrinsic flows through.
struct SelectArithAddI : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op))
      return failure();
    unsigned width = getScalarIntBitWidth(op.getType());
    if (width == 32)
      rewriter.replaceOpWithNewOp<spyreop::AddI32ToI32>(
          op, op.getType(), op.getLhs(), op.getRhs());
    else if (width == 64)
      rewriter.replaceOpWithNewOp<spyreop::AddI64ToI64>(
          op, op.getType(), op.getLhs(), op.getRhs());
    else
      return failure();
    return success();
  }
};

/// arith.muli -> spyreop.muli32toi32, inside a generic body only. Same proxy,
/// and only one width has an intrinsic.
struct SelectArithMulI : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op) || getScalarIntBitWidth(op.getType()) != 32)
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::MulI32ToI32>(op, op.getType(),
                                                      op.getLhs(), op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// After selection: an i1 left in a compute body
//===----------------------------------------------------------------------===//

/// Refuse every `i1` value still inside a compute body once selection has had
/// its chance.
///
/// THE ONE EXCEPTION TO FLOWS-THROUGH, and it is a different kind of thing
/// rather than a carve-out. That rule is about a CAPABILITY gap: an f64
/// `math.sqrt` has no intrinsic today, the IR is valid, and a future device or
/// a future rule may well do it -- so the backend is the right place to judge.
/// An `i1` in a compute body is not a capability gap, it is UNREPRESENTABLE: no
/// spyreop op produces or consumes that type, so no rule anyone could add would
/// ever select it. That is a property of the dialect, not of a device
/// generation.
///
/// And this is the only place that can say WHY. Reaching the backend, the
/// failure names an op several lowerings below the one the author wrote, and
/// nothing about the predicate that caused it. Here the predicate is in hand.
///
/// AFTER THE FIXPOINT, NOT DURING IT, and not in the fusion pass either. Before
/// selection an `i1` between a compare and its consumer is the expected shape
/// -- it is exactly what the compare rule matches -- so the fusion pass that
/// removes the TENSOR form cannot refuse the scalar one. Only once no rule has
/// claimed it is its presence evidence.
LogicalResult rejectSurvivingBooleans(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk([&](linalg::GenericOp generic) {
    Block *body = generic.getBlock();
    if (!body)
      return;
    // RESULTS ONLY, not block arguments. An i1 arriving as an `ins` operand is
    // the TENSOR form crossing into a body, and removing that is the fusion
    // pass's clause -- and it is indistinguishable here from an unfused
    // compare, whose own generic yields a `tensor<i1>` that the next generic
    // reads as an i1 argument. Refusing it would report IR a pass ahead of this
    // one was meant to reshape.
    SmallVector<Value> values;
    for (Operation &op : *body)
      values.append(op.getResults().begin(), op.getResults().end());

    for (Value v : values) {
      if (!getElementTypeOrSelf(v.getType()).isInteger(1))
        continue;
      // Nor one that LEAVES the body. Yielded, it is again the tensor form, and
      // before fusion has run every compare sits in a generic of its own
      // yielding exactly that.
      //
      // What the two tests leave is an i1 this pass PRODUCED AND KEPT: made
      // inside one body and read inside the same body. That is precisely a
      // group this pass had the chance to select and did not, which is why it
      // is this pass's finding and not its predecessor's.
      if (llvm::any_of(v.getUsers(), [](Operation *user) {
            return isa<linalg::YieldOp>(user);
          }))
        continue;
      result = failure();

      InFlightDiagnostic diag = mlir::emitError(v.getLoc());
      diag << "lower-spyre-ops: an i1 value survives inside a compute body, "
              "which the Spyre device has no form for at all -- no spyreop "
              "intrinsic produces or consumes that type, so no selection rule "
              "can ever remove it";

      // The actionable half. A compare is how an i1 gets into a body in
      // practice, and which predicate it used is the whole of what an author
      // can change.
      if (auto cmp = v.getDefiningOp<arith::CmpFOp>()) {
        StringRef pred = arith::stringifyCmpFPredicate(cmp.getPredicate());
        if (spyrePredicateFor(cmp.getPredicate()))
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' does have a spyreop.compare counterpart, so this compare "
                 "was "
                 "selectable and something about its READER was not: "
                 "spyreop.compare answers in the width compared, so the reader "
                 "must be an arith.uitofp at that same width";
        else
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' has no spyreop.compare counterpart. That intrinsic is "
                 "ORDERED for every predicate, `notequal` included -- it "
                 "answers "
                 "zero when either operand is NaN -- so an unordered predicate "
                 "would compute a different value, and `ord`/`uno` ask a "
                 "question it does not answer at all. Where NaN operands are "
                 "not "
                 "expected, the ordered spelling is selectable";
      }

      for (Operation *user : v.getUsers())
        diag.attachNote(user->getLoc())
            << "read here, by '" << user->getName() << "'";
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreOpsPass
    : public mlir::triton::spyre::impl::LowerSpyreOpsBase<LowerSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // One line per rule, group rules and 1:1 rules in one set: specificity
    // rather than a declared order is what decides between two that could both
    // match. See ONE PASS FOR ALL SELECTION in the header.
    patterns.add<SelectReciprocal, SelectCompare>(ctx);
    patterns.add<SelectArithDivF, SelectArithAddI, SelectArithMulI>(ctx);
    patterns.add<SelectUnaryFloat<math::SqrtOp, spyreop::Sqrt>,
                 SelectUnaryFloat<math::ExpOp, spyreop::Exp>,
                 SelectUnaryFloat<math::RsqrtOp, spyreop::RSqrt>>(ctx);

    // Upstream's, and load-bearing rather than tidying: a group rule that stops
    // reading a block argument leaves an `ins` operand, that argument and its
    // indexing map behind, and this removes all three. In the same fixpoint, so
    // a rule never has to see the intermediate state.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

    // The greedy driver, not a dialect conversion. Nothing here is illegal: an
    // op with no device form flows through to the backend, which is the
    // component that knows what it can take. A failure from the driver would
    // mean the rewrite diverged, not that an op went unhandled.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // The one thing this pass does report. Not a selection failure -- see
    // rejectSurvivingBooleans on why an unrepresentable TYPE is a different
    // kind of thing from an op the device happens not to do.
    if (failed(rejectSurvivingBooleans(module)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}

} // namespace mlir::triton::spyre
