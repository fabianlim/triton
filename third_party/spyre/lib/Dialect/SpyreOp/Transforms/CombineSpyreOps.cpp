//===- CombineSpyreOps.cpp - spyreop -> spyreop peepholes ----------------===//
//
// A PATTERN HOST for rewrites that replace one spyreop intrinsic with the more
// specialized spyreop intrinsic computing the same thing. Adding a case is one
// pattern plus one lit case -- no Passes.td, CMake or pipeline change.
//
// The split this pass exists to draw. LowerSpyreOps decides WHICH OP a scalar
// arith/math computation becomes -- one rule per op, no policy. Whether a
// division by a constant one should ship as a division at all is a different
// question: it is about which of two spyreop spellings the device prefers, and
// it is answerable only once the operand is in the form the preference is stated
// over. Two questions, two passes, each placed where it is individually correct.
// Fused into the lowering they share one position, and moving the lowering moves
// the preference with it.
//
// The mechanism is upstream's greedy driver; the POLICY -- which spellings are
// preferred, and on what downstream evidence -- is ours, and Passes.td holds it
// along with the rule for what may be added here.
//
// THE ONE TENANT, and why it is not a fold. `realdiv(1.0, x)` -> `reciprocal(x)`
// is not an algebraic simplification: both ops exist, both compute the same
// value, and the unary one is preferred because it carries no float immediate.
// A float immediate reaching a Spyre compute unit is not read back as it was
// written, and nothing downstream refuses it -- dbo-opt has no diagnostic for
// one -- so the rewrite is what protects the kernel rather than a diagnostic
// being what reports it. test/fixtures/reduce/meta.py records the observation.
//
// THE NUMERATOR IS NOT ERASED HERE. The greedy driver removes an op that becomes
// trivially dead, and an arith.constant is Pure, so the 1.0 goes by ordinary
// dead-op elimination. That is the whole reason this is a rewrite pattern rather
// than a conversion pattern: a dialect conversion would leave the constant
// standing, which is what made the version of this rewrite inside LowerSpyreOps
// carry a hand-rolled `hasOneUse()` guard and an explicit eraseOp. A numerator
// shared with another reader now keeps working with no special case, because
// nothing here is asking.
//
// WHAT THIS PASS DECLINES, and why the decline is correct rather than a gap.
// The match needs a constant VISIBLE FROM THE DIVIDE. A splat 1.0 passed as an
// `ins` operand of the enclosing linalg.generic arrives in the body as a BLOCK
// ARGUMENT, which is not a constant, so the realdiv survives. Converting that
// form would mean dropping an operand, its block argument and its indexing map
// from the generic -- a rewrite of the generic, not of a spyreop op, which is
// rule 1 in Passes.td. What establishes the visible form is
// FoldDataMovementGenerics, whose elementwise fusion folds the splat's producer
// into the body; measured on reduce/softmax_on_stick, that pass is the
// difference between the two forms. Hence the ordering constraint in Passes.td,
// and hence this pass sitting after the whole physicalization half of the
// `spyrecode` stage rather than beside the lowering that creates its input.
//
// `--debug-only=combine-spyre-ops` traces the decisions rather than the control
// flow: one line per realdiv considered, saying whether its numerator matched
// and, when it did not, which form the numerator had.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Transforms/Passes.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "combine-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyreop {
#define GEN_PASS_DEF_COMBINESPYREOPS
#include "Dialect/SpyreOp/Transforms/Passes.h.inc"
} // namespace mlir::triton::spyreop

namespace {

//===----------------------------------------------------------------------===//
// spyreop.realdiv(1.0, x) -> spyreop.reciprocal(x)
//===----------------------------------------------------------------------===//

/// Replaces a division by a constant-one numerator with the unary intrinsic, so
/// no float immediate reaches the device.
///
/// Matched with `m_OneFloat`, which accepts a scalar float constant or a splat.
/// Both spellings are kept deliberately: the op is scalar by the time it is a
/// spyreop, so the numerator is a scalar constant in practice, but the matcher
/// asking about the VALUE rather than about the shape is what makes the pattern
/// independent of the enclosing tensor form -- and independent of whether the
/// constant was hoisted above a linalg.generic or written beside the divide.
struct RealDivOneToReciprocal
    : public OpRewritePattern<mlir::spyreop::RealDiv> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::spyreop::RealDiv op,
                                PatternRewriter &rewriter) const override {
    Value numerator = op.getOperand(0);
    if (!matchPattern(numerator, m_OneFloat())) {
      // Named by the form the numerator HAS, not merely declined: a reader of a
      // trace that kept a realdiv needs to know which case they are in, and only
      // one of the cases is the precondition's.
      //
      // For a block argument that means naming its OWNER, because the two block
      // arguments a numerator can be are not the same finding: one belonging to a
      // linalg.generic is the splat-`ins` form the precondition is about, and one
      // belonging to a function is just a value, with nothing to fold. Matched on
      // the op's NAME rather than by casting to it, so this stays true to the
      // library's one rule -- no dialect but spyreop is linked, which is what
      // keeps a pattern here from growing into a rewrite of its consumer.
      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "] realdiv at " << op.getLoc()
                     << " keeps its binary form: numerator is ";
        if (auto arg = dyn_cast<BlockArgument>(numerator)) {
          Operation *owner = arg.getOwner()->getParentOp();
          StringRef ownerName =
              owner ? owner->getName().getStringRef() : "an unparented block";
          llvm::dbgs() << "a block argument of " << ownerName
                       << ", so not a constant this pattern can read";
          if (ownerName == "linalg.generic")
            llvm::dbgs() << " -- the splat `ins` form, see the precondition in "
                            "Passes.td";
        } else {
          llvm::dbgs() << "not a constant 1.0";
        }
        llvm::dbgs() << "\n";
      });
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] realdiv at " << op.getLoc()
                            << " -> reciprocal; its 1.0 is left to dead-op "
                               "elimination\n");
    rewriter.replaceOpWithNewOp<mlir::spyreop::Reciprocal>(op, op.getType(),
                                                           op.getOperand(1));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CombineSpyreOpsPass
    : public mlir::triton::spyreop::impl::CombineSpyreOpsBase<
          CombineSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<RealDivOneToReciprocal>(ctx);

    // The greedy driver rather than a dialect conversion, for the numerator:
    // it erases an op that becomes trivially dead, which is what makes the
    // constant go without this pass asking about its uses.
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyreop {

std::unique_ptr<OperationPass<ModuleOp>> createCombineSpyreOpsPass() {
  return std::make_unique<CombineSpyreOpsPass>();
}

} // namespace mlir::triton::spyreop
