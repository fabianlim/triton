//===- SpyreOpLoweringRules.cpp - The spyreop lowering rules -------------===//
//
// kSpyreOpLoweringRules, every rule LowerSpyreOps lowers by and FuseGenerics
// fuses for, and the guards and builds its entries name. See
// SpyreOpLoweringRules.h.
//
// A new rule is one build function and one entry in kSpyreOpLoweringRules,
// plus a guard when it has a condition no verifier sees, with no new pattern
// in either pass.
//
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets a group:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// NO RULE CHECKS A TYPE. Which types a rule lowers is what the spyreop ops it
// builds accept: ruleMatches runs the rule's build into a scratch block whose
// arguments have the operands' types (probeBuild), verifies every op it made
// against that op's own definition, and erases them. A rule whose ops do not
// verify does not match, so two rules on one anchor that build ops of
// different types, such as addi32 and addi64, need no guard to tell them
// apart. A guard is left for what no verifier sees: a constant operand, or a
// predicate's meaning.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir::triton::spyre {

//===----------------------------------------------------------------------===//
// Shared questions
//===----------------------------------------------------------------------===//

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

namespace {

template <typename OpT> constexpr StringLiteral opName() {
  return OpT::getOperationName();
}

//===----------------------------------------------------------------------===//
// Guards
//===----------------------------------------------------------------------===//

/// The reciprocal: the numerator is a scalar constant 1.0.
bool guardReciprocal(const GroupCandidate &candidate, std::string &why) {
  if (matchPattern(cast<arith::DivFOp>(candidate.anchor).getLhs(),
                   m_OneFloat()))
    return true;
  why = "the numerator is not 1.0";
  return false;
}

std::string noCounterpart(arith::CmpFOp cmp) {
  return (llvm::Twine("spyreop.compare has no counterpart for predicate '") +
          arith::stringifyCmpFPredicate(cmp.getPredicate()) + "'")
      .str();
}

/// compare-cast and compare-select: the predicate has a spyreop.compare
/// counterpart.
bool guardCompare(const GroupCandidate &candidate, std::string &why) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  if (spyrePredicateFor(cmp.getPredicate()))
    return true;
  why = noCounterpart(cmp);
  return false;
}

/// A zero test that `spyreop.select` performs itself, found by
/// selectFromZeroTest. `conditionOperand` is the index among the compare's
/// operands of the one that is not zero, the scalar already in the body.
struct ZeroTestSelect {
  unsigned conditionOperand;
  bool exchangeValues;
};

/// Finds a comparison the select can drop, because `spyreop.select` already
/// tests its condition against zero:
///
///   - `m != 0` (`one`/`une`) -> condition `m`, values in order;
///   - `m == 0` (`oeq`/`ueq`) -> condition `m`, values exchanged;
///   - zero, a scalar constant, on either side, and -0.0, e.g. `0 != m`.
///
/// Returns nothing for the rest, which keep their compare:
///
///   - an ordering, e.g. `m > 0` is false for a negative `m` that "not zero"
///     calls true;
///   - an operand that is not a scalar constant zero, e.g. a block argument.
std::optional<ZeroTestSelect> selectFromZeroTest(arith::CmpFOp cmp) {
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
  if (matchPattern(cmp.getRhs(), m_AnyZeroFloat()))
    return ZeroTestSelect{0, exchangeValues};
  if (matchPattern(cmp.getLhs(), m_AnyZeroFloat()))
    return ZeroTestSelect{1, exchangeValues};
  return std::nullopt;
}

/// compare-select-zero-test: the compare is a zero test spyreop.select
/// performs itself (selectFromZeroTest).
bool guardZeroTest(const GroupCandidate &candidate, std::string &why) {
  if (selectFromZeroTest(cast<arith::CmpFOp>(candidate.anchor)))
    return true;
  why = "the compare is not a zero test spyreop.select performs itself";
  return false;
}

//===----------------------------------------------------------------------===//
// Builds
//===----------------------------------------------------------------------===//

/// sqrt, exp, rsqrt: `Target`(a0).
template <typename Target>
Value buildUnary(OpBuilder &b, const GroupCandidate &candidate,
                 ValueRange operands) {
  Operation *op = candidate.anchor;
  return Target::create(b, op->getLoc(), op->getResult(0).getType(),
                        operands[0]);
}

/// `Target`(a0, a1), for realdiv and the integer rules.
template <typename Target>
Value buildBinary(OpBuilder &b, const GroupCandidate &candidate,
                  ValueRange operands) {
  Operation *op = candidate.anchor;
  return Target::create(b, op->getLoc(), op->getResult(0).getType(),
                        operands[0], operands[1]);
}

/// reciprocal: `spyreop.reciprocal`(a1).
Value buildReciprocal(OpBuilder &b, const GroupCandidate &candidate,
                      ValueRange operands) {
  Operation *op = candidate.anchor;
  return spyreop::Reciprocal::create(b, op->getLoc(),
                                     op->getResult(0).getType(), operands[1]);
}

spyreop::ComparePredicateAttr comparePredicate(arith::CmpFOp cmp) {
  return spyreop::ComparePredicateAttr::get(
      cmp.getContext(), *spyrePredicateFor(cmp.getPredicate()));
}

/// compare-cast: the reader, an arith.uitofp, becomes `spyreop.compare` of
/// the compare's operands under its predicate's counterpart, at the reader's
/// type.
Value buildCompareCast(OpBuilder &b, const GroupCandidate &candidate,
                       ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *reader = candidate.readers.front();
  return spyreop::Compare::create(b, reader->getLoc(),
                                  reader->getResult(0).getType(), operands[0],
                                  operands[1], comparePredicate(cmp));
}

/// compare-select-zero-test: the reader, an arith.select, becomes a
/// `spyreop.select` on the compare's nonzero operand, with the values
/// exchanged when the test is `== 0`. Operands: a0, a1, then the select's
/// condition, r1 and r2.
Value buildZeroTestSelect(OpBuilder &b, const GroupCandidate &candidate,
                          ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *select = candidate.readers.front();
  ZeroTestSelect zeroTest = *selectFromZeroTest(cmp);
  // An opposite zero test reaches the same condition value, so the exchange
  // is what keeps it meaning the same thing.
  Value trueValue = operands[3], falseValue = operands[4];
  if (zeroTest.exchangeValues)
    std::swap(trueValue, falseValue);
  return spyreop::Select::create(b, select->getLoc(),
                                 select->getResult(0).getType(),
                                 operands[zeroTest.conditionOperand], trueValue,
                                 falseValue);
}

/// compare-select: the reader, an arith.select, becomes a `spyreop.select`
/// whose condition is a `spyreop.compare` of the compare's operands, placed
/// before it.
Value buildCompareSelect(OpBuilder &b, const GroupCandidate &candidate,
                         ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *select = candidate.readers.front();
  Type selected = select->getResult(0).getType();
  Value condition =
      spyreop::Compare::create(b, select->getLoc(), selected, operands[0],
                               operands[1], comparePredicate(cmp));
  return spyreop::Select::create(b, select->getLoc(), selected, condition,
                                 operands[3], operands[4]);
}

//===----------------------------------------------------------------------===//
// The table
//===----------------------------------------------------------------------===//

constexpr StringLiteral kUIToFPReader[] = {opName<arith::UIToFPOp>()};
constexpr StringLiteral kSelectReader[] = {opName<arith::SelectOp>()};

/// Every rule LowerSpyreOps lowers by, the rules lowering one op first, then
/// the rules with a reader. Among the rules on one anchor a higher benefit is
/// tried first, and rules of equal benefit on one anchor never both match one
/// candidate. The rules with a reader are also tried in this order by
/// matchLowerableGroup, so rules with the same anchor and the same readers are
/// listed by descending benefit.
///
/// No op is both a rule's reader and the anchor of a rule without readers:
/// benefit orders only patterns rooted on the same op, so such an overlap
/// would be decided by the order the driver visits ops.
const SpyreOpLoweringRule kSpyreOpLoweringRules[] = {
    {"sqrt", opName<math::SqrtOp>(), {}, 1, nullptr,
     buildUnary<spyreop::Sqrt>},
    {"exp", opName<math::ExpOp>(), {}, 1, nullptr,
     buildUnary<spyreop::Exp>},
    {"rsqrt", opName<math::RsqrtOp>(), {}, 1, nullptr,
     buildUnary<spyreop::RSqrt>},
    {"reciprocal", opName<arith::DivFOp>(), {}, 2, guardReciprocal,
     buildReciprocal},
    {"realdiv", opName<arith::DivFOp>(), {}, 1, nullptr,
     buildBinary<spyreop::RealDiv>},
    // The integer rules over-claim: an integer add or multiply in a body is
    // taken for compute, which `tl.arange(0, N) * stride` scalarized into a
    // body would not be.
    {"addi32", opName<arith::AddIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::AddI32ToI32>},
    {"addi64", opName<arith::AddIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::AddI64ToI64>},
    {"muli32", opName<arith::MulIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::MulI32ToI32>},
    {"compare-cast", opName<arith::CmpFOp>(), kUIToFPReader, 1, guardCompare,
     buildCompareCast},
    {"compare-select-zero-test", opName<arith::CmpFOp>(), kSelectReader, 2,
     guardZeroTest, buildZeroTestSelect},
    {"compare-select", opName<arith::CmpFOp>(), kSelectReader, 1, guardCompare,
     buildCompareSelect},
};

/// Whether every op `rule.build` makes for `candidate` passes its own
/// verifier, on operands of the types `operands` has. The build runs into a
/// detached block whose arguments stand for `operands`, with a builder that
/// notifies no rewriter, so no real IR is touched, and the block and every op
/// in it are destroyed on return. The verifier's diagnostics are caught by a
/// handler scoped to the check, and the first becomes `why`.
bool probeBuild(const SpyreOpLoweringRule &rule,
                const GroupCandidate &candidate, ValueRange operands,
                std::string &why) {
  MLIRContext *ctx = candidate.anchor->getContext();
  Block scratch;
  for (Value operand : operands)
    scratch.addArgument(operand.getType(), operand.getLoc());
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(&scratch);
  rule.build(builder, candidate, scratch.getArguments());

  std::string diagnostic;
  ScopedDiagnosticHandler handler(ctx, [&](Diagnostic &diag) {
    if (diagnostic.empty())
      diagnostic = diag.str();
    return success();
  });
  for (Operation &op : scratch)
    if (failed(verify(&op, /*verifyRecursively=*/false))) {
      why = "'" + op.getName().getStringRef().str() +
            "' does not take these types: " + diagnostic;
      return false;
    }
  return true;
}

} // namespace

SmallVector<Value> ruleOperands(const SpyreOpLoweringRule &rule,
                                const GroupCandidate &candidate) {
  SmallVector<Value> operands(candidate.anchor->getOperands());
  if (!rule.readers.empty())
    llvm::append_range(operands, candidate.readers.front()->getOperands());
  return operands;
}

ArrayRef<SpyreOpLoweringRule> spyreOpLoweringRules() {
  return kSpyreOpLoweringRules;
}

bool ruleMatches(const SpyreOpLoweringRule &rule,
                 const GroupCandidate &candidate, std::string &why) {
  // A rule with readers asks for them one for one; a rule without asks
  // nothing of them.
  if (!rule.readers.empty() &&
      candidate.readers.size() != rule.readers.size()) {
    why = candidate.readers.empty()
              ? std::string("nothing reads the anchor")
              : "the anchor has " + std::to_string(candidate.readers.size()) +
                    " readers, and a group takes exactly " +
                    (rule.readers.size() == 1
                         ? std::string("one")
                         : std::to_string(rule.readers.size()));
    return false;
  }
  if (candidate.anchor->getName().getStringRef() != rule.anchor) {
    why = ("the anchor is not an " + rule.anchor).str();
    return false;
  }
  for (auto [reader, name] : llvm::zip(candidate.readers, rule.readers))
    if (reader->getName().getStringRef() != name) {
      why = "the reader, '" + reader->getName().getStringRef().str() +
            "', is not an " + name.str();
      return false;
    }
  if (rule.guard && !rule.guard(candidate, why))
    return false;
  return probeBuild(rule, candidate, ruleOperands(rule, candidate), why);
}

FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why) {
  // Each distinct reason once: the rules on one reader count, or on one
  // anchor op, decline for the same one.
  SmallVector<std::string> reasons;
  for (const SpyreOpLoweringRule &rule : kSpyreOpLoweringRules) {
    if (rule.readers.empty())
      continue;
    std::string reason;
    if (ruleMatches(rule, candidate, reason))
      return GroupMatch{rule.name, {candidate.anchor}};
    if (!llvm::is_contained(reasons, reason))
      reasons.push_back(reason);
  }
  if (why)
    *why = llvm::join(reasons, "; ");
  return failure();
}

} // namespace mlir::triton::spyre
