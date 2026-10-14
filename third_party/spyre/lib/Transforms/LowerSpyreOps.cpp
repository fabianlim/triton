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
//   one op to one   math.sqrt -> spyreop.sqrt, and so on. `arith.divf`
//                   picks its target from the numerator: `1.0 / x` ->
//                   spyreop.reciprocal, anything else -> spyreop.realdiv.
//   a GROUP to one  a compare and a cast -> spyreop.compare. The group is what
//                   is selectable: no member of it could have been selected
//                   alone.
//   a REQUEST       a body whose ops all carry `tts.spyre_intrinsic` attributes of one
//                   `id` -> the op the hint's `target` names, applied to the
//                   call operands in call order, replacing the whole body. The
//                   author asked for that op with `tl.spyre_intrinsic`, and the body
//                   is its fallback; see matchCallSite and selectCallSite.
//
// Two phases, in this order. Requests first, by a walk over the generics: a
// body holding a call site becomes its intrinsic, and a hint left anywhere
// after that is refused. Then the 1:1 and group rules, which share one greedy
// pattern set; no two of them are rooted on the same op, so no order is
// declared between them. No hinted op is left when the second phase starts,
// so no rule there has to recognize one: a fallback's `math.exp` is gone
// before SelectUnaryFloat could see it.
//
// EVERYTHING UNMATCHED FLOWS THROUGH, WITH TWO EXCEPTIONS. There is no
// conversion target: an op with no device form reaches the backend, which is
// the component that actually knows what it can take, and it refuses there. So
// this pass has no notion of an illegal input -- it selects what it can and
// leaves the rest exactly as it found it. A type or a predicate this file does
// not handle is a silent pass-through by design; see WHAT IS NOT SELECTED below
// for the list and what each one costs. The exceptions are an `i1` left inside
// a compute body, which rejectSurvivingBooleans reports, and a
// `tts.spyre_intrinsic` attribute that survives selection, which rejectSurvivingHints
// reports: the author asked for the intrinsic by name, so running the fallback
// instead would ignore the request without saying so.
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
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets the pass:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// A NaN input gives an unspecified result on its lane.
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
//     - `ord`/`uno` and the constant predicates `false`/`true`: no
//       counterpart. They ask about NaN-ness or about nothing, not about an
//       ordering
//     - `arith.sitofp` where `uitofp` was wanted: an `i1` read as signed is 0
//       or -1, so the cast gives -1.0 where the predicate holds
//     - a compare at one width cast to another: spyreop.compare has
//       SameOperandsAndResultType and cannot do both
//
//   The compare declines are the ones to know: each leaves an `i1` inside a
//   body, so each is the exception above rather than a pass-through --
//   rejectSurvivingBooleans reports it here, naming the predicate.
//
// `--debug-only=lower-spyre-ops` traces the group rules' decisions rather than
// the control flow: one line per candidate looked at and what came of it, with
// the operand form named when a match was declined. The 1:1 rules say nothing
// -- there is one per op, and the output IR shows whether it fired.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "Dialect/TTS/IR/Dialect.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

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
/// SCOPE instead, requiring the generic to be their immediate parent.
bool isInsideLinalgGeneric(Operation *op) {
  return op->getParentOfType<linalg::GenericOp>() != nullptr;
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
  OpOperand *operand = generic.getMatchingOpOperand(arg);
  return generic.isDpsInput(operand) ? operand->get() : v;
}

/// The spyreop predicate computing the same thing as `p`, or nothing:
///
///   - an ordering maps from both spellings, e.g. `oeq` and `ueq` -> `equal`
///     (they differ only on NaN; see NO NaN REACHES A COMPUTE BODY);
///   - `ord`/`uno` ask about NaN and `false`/`true` about nothing, so they have
///     no counterpart.
std::optional<spyreop::ComparePredicate>
spyrePredicateFor(arith::CmpFPredicate p) {
  switch (p) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return spyreop::ComparePredicate::Equal;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return spyreop::ComparePredicate::NotEqual;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return spyreop::ComparePredicate::GreaterThan;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return spyreop::ComparePredicate::GreaterEqual;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return spyreop::ComparePredicate::LesserThan;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
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

/// Matches a floating-point comparison feeding a single-result consumer
/// directly inside a linalg.generic body. The consumer and compared values
/// must have the same scalar f16 or f32 type. Returns null on a mismatch;
/// predicate support remains the caller's responsibility.
arith::CmpFOp matchComparedInput(Operation *consumer, Value input) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer->getParentOp());
  if (!generic)
    return nullptr;
  Type resultType = consumer->getResult(0).getType();
  if (!isSpyreOpScalarType(resultType)) {
    traceDecline(consumer, "no spyreop intrinsic for this result type");
    return nullptr;
  }
  auto cmp = resolveThroughBody(generic, input).getDefiningOp<arith::CmpFOp>();
  if (!cmp) {
    traceDecline(consumer, "operand is not an arith.cmpf");
    return nullptr;
  }
  // Device comparison and selection require identical input and result types.
  if (cmp.getLhs().getType() != resultType) {
    traceDecline(consumer, "the compared type and the consumer's result type "
                           "differ, which one spyreop intrinsic cannot express");
    return nullptr;
  }
  return cmp;
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
// One op to one: arith.divf -> spyreop.reciprocal or spyreop.realdiv
//===----------------------------------------------------------------------===//

/// `1.0 / x` -> `spyreop.reciprocal x`; any other `a / b` ->
/// `spyreop.realdiv a, b`.
///
/// The numerator is read through the enclosing body when there is one, so a
/// one passed in as a splat `ins` operand matches as well as a scalar
/// constant. Its unused block argument is then erased upstream.
struct SelectArithDivF : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();

    Value numerator = op.getLhs();
    if (linalg::GenericOp generic = dyn_cast<linalg::GenericOp>(op->getParentOp()))
      numerator = resolveThroughBody(generic, numerator);
    if (matchPattern(numerator, m_OneFloat())) {
      traceMatch(op, "numerator is 1.0 -> spyreop.reciprocal");
      rewriter.replaceOpWithNewOp<spyreop::Reciprocal>(op, op.getType(),
                                                       op.getRhs());
      return success();
    }

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
    auto cmp = matchComparedInput(op, op.getIn());
    if (!cmp)
      return failure();

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
// A group to one: arith.cmpf feeding arith.select -> spyreop.select
//===----------------------------------------------------------------------===//

/// A zero test that `spyreop.select` performs itself, found by
/// selectFromZeroTest. `condition` is the scalar already in the body.
struct ZeroTestSelect {
  Value condition;
  bool exchangeValues;
};

/// Finds a comparison the select can drop, because `spyreop.select` already
/// tests its condition against zero:
///
///   - `m != 0` (`one`/`une`) -> condition `m`, values in order;
///   - `m == 0` (`oeq`/`ueq`) -> condition `m`, values exchanged;
///   - zero on either side, as a scalar or a uniform input tensor, and -0.0,
///     e.g. `0 != m`, or `%z` fed by `dense<0.0>`.
///
/// Returns nothing for the rest, which keep their compare:
///
///   - an ordering, e.g. `m > 0` is false for a negative `m` that "not zero"
///     calls true;
///   - a non-uniform constant, e.g. `dense<[0.0, 1.0]>`.
std::optional<ZeroTestSelect> selectFromZeroTest(linalg::GenericOp generic,
                                                 arith::CmpFOp cmp) {
  bool exchangeValues;
  switch (cmp.getPredicate()) {
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    exchangeValues = false;
    break;
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    exchangeValues = true;
    break;
  default:
    return std::nullopt;
  }
  if (matchPattern(resolveThroughBody(generic, cmp.getRhs()), m_AnyZeroFloat()))
    return ZeroTestSelect{cmp.getLhs(), exchangeValues};
  if (matchPattern(resolveThroughBody(generic, cmp.getLhs()), m_AnyZeroFloat()))
    return ZeroTestSelect{cmp.getRhs(), exchangeValues};
  return std::nullopt;
}

/// Lowers `arith.cmpf` feeding `arith.select` (e.g. `tl.where(a > b, p, q)`)
/// in one of two ways, chosen by selectFromZeroTest:
///
///   - a zero test -> `spyreop.select %m, %p, %q` alone;
///   - otherwise -> `%c = spyreop.compare <greaterthan> %a, %b` then
///     `spyreop.select %c, %p, %q`.
///
/// Rooted on the select, because the cmpf's reader decides the device op. The
/// cmpf is left for dead-code removal. Declines, leaving the arith ops, when:
///
///   - the selected type has no `spyreop.select`, e.g. i32 or f64;
///   - the condition is not a cmpf, e.g. an `i1` function argument;
///   - the compared and selected widths differ, e.g. f32 compared, f16 picked;
///   - a kept compare has no counterpart, e.g. `ord`.
///
/// A declined cmpf is then reported by rejectSurvivingBooleans.
struct SelectWhere : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cmp = matchComparedInput(op, op.getCondition());
    if (!cmp)
      return failure();
    auto generic = cast<linalg::GenericOp>(op->getParentOp());
    Type selected = op.getType();

    std::optional<ZeroTestSelect> zeroTest = selectFromZeroTest(generic, cmp);
    StringRef predicateName = arith::stringifyCmpFPredicate(cmp.getPredicate());
    std::optional<spyreop::ComparePredicate> predicate =
        spyrePredicateFor(cmp.getPredicate());
    if (!zeroTest && !predicate) {
      traceDecline(op, llvm::Twine("spyreop.compare has no counterpart for "
                                   "predicate '") +
                           predicateName + "'");
      return failure();
    }

    // All rejection checks precede mutation: a failed match must not leave
    // partially constructed device operations in the body.
    Value condition = zeroTest ? zeroTest->condition : Value{};
    if (!zeroTest) {
      traceMatch(op, llvm::Twine("arith.cmpf '") + predicateName +
                         "' + arith.select -> spyreop.compare feeding "
                         "spyreop.select");
      condition = spyreop::Compare::create(
          rewriter, op.getLoc(), selected, cmp.getLhs(), cmp.getRhs(),
          spyreop::ComparePredicateAttr::get(op.getContext(), *predicate));
    } else {
      traceMatch(op,
                 llvm::Twine("arith.cmpf '") + predicateName +
                     "' tests against zero, which spyreop.select does to its "
                     "own condition -> spyreop.select alone");
    }

    // An opposite zero test reaches the same condition value, so the exchange
    // is what keeps it meaning the same thing.
    Value trueValue = op.getTrueValue(), falseValue = op.getFalseValue();
    if (zeroTest && zeroTest->exchangeValues)
      std::swap(trueValue, falseValue);

    rewriter.replaceOpWithNewOp<spyreop::Select>(op, selected, condition,
                                                 trueValue, falseValue);
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
    if (!isInsideLinalgGeneric(op) ||
        getScalarIntBitWidth(op.getType()) != 32)
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::MulI32ToI32>(op, op.getType(),
                                                      op.getLhs(), op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A request to one: a body hinted `tts.spyre_intrinsic` -> the op it targets
//===----------------------------------------------------------------------===//

/// What the call-site rule needs from a body it replaces.
struct CallSiteMatch {
  DictionaryAttr hint;
  /// Call operand k -> the `ins` block argument holding it, in call order.
  SmallVector<BlockArgument> inputs;
  SmallVector<Operation *> members;
};

/// Whether block arguments `a` and `b` of `generic` read one element of one
/// value: the same `ins` operand through the same indexing map. Two ops of a
/// call site reading one call operand from two generics give the fused body two
/// such block arguments unless the fusion merges them, which it does for the
/// `ins` it sees are equal; a body that kept both still has one input there.
bool readSameInput(linalg::GenericOp generic, BlockArgument a,
                   BlockArgument b) {
  OpOperand *operandA = generic.getMatchingOpOperand(a);
  OpOperand *operandB = generic.getMatchingOpOperand(b);
  return operandA->get() == operandB->get() &&
         generic.getMatchingIndexingMap(operandA) ==
             generic.getMatchingIndexingMap(operandB);
}

/// What `generic`'s body, the call site `hint` names, reads for each of its
/// call operands, or failure, reported, when the body does not say.
///
/// FuseComputeAndDataMovement has already checked that the body holds one call
/// site, whole, and nothing else but constants, so what is left to find is the
/// inputs. Not by the block arguments' order, which after
/// ConvertElementwiseToLinalg and fusion is whatever order the `ins` ended up
/// in -- `b / a` converts to `ins(%b, %a)`, block argument 0 being call operand
/// 1 -- but from the binding LowerTTSMarkers wrote when it was exact: each
/// member's `operands`, entry j naming the call operand that member's operand j
/// is.
///
/// On a non-commutative member the binding is positional: operand j is call
/// operand `operands[j]`. On a commutative member it is read positionally when
/// every position it lists holds an `ins` block argument, and otherwise as a
/// set: the member's operands that are `ins` block arguments, whatever their
/// positions, are the call operands it lists, one each. A set reading assigns
/// only when the member lists one call operand, possibly more than once, since
/// then every assignment is the same one; listing two different ones, it is
/// refused rather than guessed. So a commutative member either binds by the
/// positions it was tagged with, or binds every input it reads to the one call
/// operand it names, and in both cases its bindings are checked against the
/// other members' like any.
///
/// Gathered over every member, and refused, naming the target and the call
/// operand, when it
///
///   - binds call operand k to an operand that is not an `ins` block argument
///     of this generic, which nothing between the two passes makes it;
///   - binds, on a commutative member read as a set, a number of call operands
///     other than the number of `ins` block arguments the member reads, or two
///     different call operands;
///   - binds call operand k to two block arguments reading different values,
///     two reading one value through one map being one input (readSameInput);
///   - binds no block argument to some call operand k below `arity`, which is
///     a fallback that does not read that operand, or whose reading folded
///     away: the target would have nothing to take it from.
///
/// A bound block argument whose `ins` operand is a constant is used like any
/// other: the binding says it is call operand k, so it is a call operand that
/// folded to a constant after LowerTTSMarkers refused the ones constant there,
/// and the target applied to it still computes what was asked. An unbound one
/// -- a constant the fusion did not fold into the body -- is not read for the
/// target at all, and goes with the members' other unused inputs.
FailureOr<CallSiteMatch> matchCallSite(linalg::GenericOp generic,
                                       DictionaryAttr hint) {
  StringRef target = triton::tts::getSpyreIntrinsicTarget(hint);
  int64_t arity = triton::tts::getSpyreIntrinsicArity(hint);
  Block *body = generic.getBlock();
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  auto refuse = [&](Location loc) {
    return mlir::emitError(loc)
           << "lower-spyre-ops: tl.spyre_intrinsic(\"" << target << "\"): ";
  };

  CallSiteMatch match;
  match.hint = hint;
  match.inputs.assign(arity, BlockArgument());
  for (Operation &op : body->without_terminator())
    if (!op.hasTrait<OpTrait::ConstantLike>())
      match.members.push_back(&op);

  auto isInput = [&](Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner() == body &&
           generic.isDpsInput(generic.getMatchingOpOperand(arg));
  };

  for (Operation *op : match.members) {
    DictionaryAttr opHint = triton::tts::getSpyreIntrinsic(op);
    DenseI64ArrayAttr operands =
        opHint ? triton::tts::getSpyreIntrinsicOperands(opHint) : nullptr;
    if (!operands)
      continue;

    // (operand index, call operand) per bound operand of `op`: by position,
    // or, on a commutative op whose listed positions do not all hold inputs,
    // by set -- see the function comment.
    SmallVector<std::pair<unsigned, int64_t>> binding;
    SmallVector<int64_t> listed;
    bool positional = true;
    for (auto [j, k] : llvm::enumerate(operands.asArrayRef())) {
      if (k < 0)
        continue;
      binding.emplace_back(j, k);
      listed.push_back(k);
      positional &= isInput(op->getOperand(j));
    }
    if (!positional && op->hasTrait<OpTrait::IsCommutative>()) {
      SmallVector<unsigned> inputs;
      for (auto [j, v] : llvm::enumerate(op->getOperands()))
        if (isInput(v))
          inputs.push_back(j);
      if (inputs.size() != listed.size()) {
        refuse(op->getLoc())
            << "call operand " << listed.front() << " is bound on commutative '"
            << op->getName() << "', which binds " << listed.size()
            << " call operand(s) and reads " << inputs.size()
            << " input(s) of its compute body";
        return failure();
      }
      if (!llvm::all_equal(listed)) {
        refuse(op->getLoc())
            << "call operand " << listed.front() << " is bound on commutative '"
            << op->getName()
            << "', whose operands are not in the order its binding names, and "
               "which binds more than one call operand, so which input is "
               "which call operand is not known";
        return failure();
      }
      binding.clear();
      for (unsigned j : inputs)
        binding.emplace_back(j, listed.front());
    }

    for (auto [j, k] : binding) {
      Value v = op->getOperand(j);
      if (!isInput(v)) {
        refuse(op->getLoc())
            << "call operand " << k << " is operand " << j << " of '"
            << op->getName()
            << "', which is not an input of its compute body -- a compiler "
               "bug, since nothing between the call site's inlining and its "
               "selection makes it one";
        return failure();
      }
      auto arg = cast<BlockArgument>(v);
      BlockArgument &bound = match.inputs[k];
      if (!bound)
        bound = arg;
      else if (bound != arg && !readSameInput(generic, bound, arg)) {
        refuse(op->getLoc())
            << "call operand " << k
            << " is read from two different inputs of its compute body, block "
               "arguments "
            << bound.getArgNumber() << " and " << arg.getArgNumber();
        return failure();
      }
    }
  }

  for (auto [k, bound] : llvm::enumerate(match.inputs))
    if (!bound) {
      refuse(generic.getLoc())
          << "call operand " << k
          << " is read by no op left of the fallback, so there is no input to "
             "pass the target for it";
      return failure();
    }
  if (yield->getNumOperands() != 1) {
    refuse(generic.getLoc())
        << "the call site's body yields " << yield->getNumOperands()
        << " values, and a request has one result";
    return failure();
  }
  return match;
}

/// Replaces a body that is one call site with the op its hint targets.
///
/// Matched on the generic and not on a member, because the call site is the
/// BODY: which op the fallback happens to end in says nothing. What the members
/// computed -- the fallback, casts included -- is discarded whole, and the body
/// becomes the target applied to its inputs, in call order, at the type the
/// body yields. The inputs' unused siblings, a captured constant among them,
/// are left to the unused-operand cleanup in the rule phase.
///
/// The target is built by name, so nothing here knows what it takes: its own
/// verifier is asked, and a target that refuses this operand or type is erased
/// again, its diagnostic emitted, and the body left for rejectSurvivingHints.
///
/// The `tts` attributes of the op whose value was yielded move to the target,
/// the hint aside: that op produced the call site's result, so a `tts.pin` on
/// the result is on it.
void selectCallSite(RewriterBase &rewriter, linalg::GenericOp generic,
                    const CallSiteMatch &match) {
  StringRef target = triton::tts::getSpyreIntrinsicTarget(match.hint);
  traceMatch(generic, "body is call site -> " + llvm::Twine(target));

  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  Value result = yield->getOperand(0);
  rewriter.setInsertionPoint(yield);
  OperationState state(result.getLoc(), target);
  state.addOperands(ArrayRef<BlockArgument>(match.inputs));
  state.addTypes(result.getType());
  Operation *selected = rewriter.create(state);
  if (failed(mlir::verify(selected))) {
    rewriter.eraseOp(selected);
    return;
  }
  if (Operation *producer = result.getDefiningOp()) {
    triton::tts::copyTTSAttrs(producer, selected);
    selected->removeDiscardableAttr(
        triton::tts::TTSDialect::kSpyreIntrinsicAttrName);
  }
  rewriter.modifyOpInPlace(
      yield, [&] { yield->setOperand(0, selected->getResult(0)); });
  // Reverse program order, so each op's users are gone before it is.
  for (Operation *op : llvm::reverse(match.members))
    rewriter.eraseOp(op);
}

/// The request phase: every body that is one call site becomes its target.
/// A walk and not a fixpoint, since a body holds at most one call site and
/// replacing it makes no other. A body matchCallSite refuses is reported there,
/// with the reason, and left hinted, so rejectSurvivingHints reports the
/// request as unselected as well.
void selectCallSites(ModuleOp module) {
  SmallVector<std::pair<linalg::GenericOp, CallSiteMatch>> matches;
  module.walk([&](linalg::GenericOp generic) {
    DictionaryAttr hint = triton::tts::getBodyCallSite(*generic.getBlock());
    if (!hint)
      return;
    FailureOr<CallSiteMatch> match = matchCallSite(generic, hint);
    if (succeeded(match))
      matches.emplace_back(generic, std::move(*match));
  });
  IRRewriter rewriter(module.getContext());
  for (auto &[generic, match] : matches)
    selectCallSite(rewriter, generic, match);
}

/// Reports every call site selection left in place, once per `id`, naming the
/// intrinsic: the author asked for it by name, so running its fallback instead
/// would ignore that request without saying so. After the request phase, so it
/// is the one check that no `tts.spyre_intrinsic` attribute reaches the rule phase.
LogicalResult rejectSurvivingHints(ModuleOp module) {
  llvm::SmallDenseSet<int64_t> reported;
  module.walk([&](Operation *op) {
    DictionaryAttr hint = triton::tts::getSpyreIntrinsic(op);
    if (!hint || !reported.insert(triton::tts::getSpyreIntrinsicId(hint)).second)
      return;
    mlir::emitError(op->getLoc())
        << "lower-spyre-ops: tl.spyre_intrinsic(\""
        << triton::tts::getSpyreIntrinsicTarget(hint)
        << "\") was not selected. The request is explicit, so its fallback is "
           "not used instead";
  });
  return success(reported.empty());
}

//===----------------------------------------------------------------------===//
// After selection: an i1 left in a compute body
//===----------------------------------------------------------------------===//

/// Reports an `i1` that selection left in a compute body, because no spyreop
/// op takes that type, so the backend would refuse it without naming the
/// predicate. Runs after the fixpoint, since until then such an `i1` is what
/// the group rules match.
///
/// Reported, when made and read in the same body without being yielded:
///
///   - `cmpf ogt` read by an i32 `arith.select`;
///   - `cmpf ord` read by `arith.uitofp` (no counterpart);
///   - `cmpf oeq` read by `arith.sitofp` (the compare rule wants `uitofp`).
///
/// Not reported, because each is the tensor form crossing a body boundary,
/// which FuseComputeAndDataMovement removes:
///
///   - an `i1` block argument, e.g. `^bb0(%cond: i1, ...)`;
///   - an `i1` that is yielded, e.g. `linalg.yield %c : i1`.
LogicalResult rejectSurvivingBooleans(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk([&](linalg::GenericOp generic) {
    Block *body = generic.getBlock();
    if (!body)
      return;
    // Op results only: an i1 block argument is the tensor form crossing into
    // the body, which is not this pass's finding.
    SmallVector<Value> values;
    for (Operation &op : *body)
      values.append(op.getResults().begin(), op.getResults().end());

    for (Value v : values) {
      if (!getElementTypeOrSelf(v.getType()).isInteger(1))
        continue;
      // Nor one that is yielded: that is the tensor form leaving the body.
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

      // Name the predicate: it is what an author can change.
      if (auto cmp = v.getDefiningOp<arith::CmpFOp>()) {
        StringRef pred = arith::stringifyCmpFPredicate(cmp.getPredicate());
        if (spyrePredicateFor(cmp.getPredicate()))
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' does have a spyreop.compare counterpart, so this compare "
                 "was selectable and something about its READER was not: the "
                 "reader must be an arith.uitofp to, or an arith.select of, "
                 "the width compared, and that width must be f16 or f32";
        else
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' has no spyreop.compare counterpart: `ord` and `uno` "
                 "ask whether an operand is NaN, and `false`/`true` ask "
                 "nothing, while that intrinsic only compares two values";
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

    // The request phase. A hint it leaves is refused here rather than handed
    // on, since the rule phase would select the fallback's ops one by one.
    selectCallSites(module);
    if (failed(rejectSurvivingHints(module))) {
      signalPassFailure();
      return;
    }

    // The rule phase. One line per rule, group and 1:1 rules in one set, each
    // rooted on a different op. See ONE PASS FOR ALL SELECTION in the header.
    RewritePatternSet patterns(ctx);
    patterns.add<SelectCompare, SelectWhere>(ctx);
    patterns.add<SelectArithDivF, SelectArithAddI, SelectArithMulI>(ctx);
    patterns.add<SelectUnaryFloat<math::SqrtOp, spyreop::Sqrt>,
                 SelectUnaryFloat<math::ExpOp, spyreop::Exp>,
                 SelectUnaryFloat<math::RsqrtOp, spyreop::RSqrt>>(ctx);

    // Upstream's, and load-bearing rather than tidying: a rule that stops
    // reading a block argument leaves an `ins` operand, that argument and its
    // indexing map behind, and this removes all three -- the request phase's
    // replacements included. In the same fixpoint, so a rule never has to see
    // the intermediate state.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

    // The greedy driver, not a dialect conversion. Nothing here is illegal: an
    // op with no device form flows through to the backend, which is the
    // component that knows what it can take. A failure from the driver would
    // mean the rewrite diverged, not that an op went unhandled.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // The other thing this pass reports, besides a surviving hint. An `i1` is
    // not a selection failure at all -- see rejectSurvivingBooleans on why an
    // unrepresentable TYPE is a different kind of thing from an op the device
    // happens not to do.
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
