#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_INDEXDOMAIN_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_INDEXDOMAIN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::triton::ktdp {

/// True for a cast that is transparent when asking *which* value an index
/// derives from. Only the root's identity matters, not the value reaching it.
/// NOT a test for whether an expression may be re-emitted in a wider type --
/// use isValuePreservingIntCast for that.
bool isIdentityTracingIntCast(Operation *op);

/// True for a cast that cannot change the number an expression denotes when the
/// expression is re-emitted in 64-bit `index`.
bool isValuePreservingIntCast(Operation *op);

/// Walk a def-chain of index arithmetic back to the single BlockArgument it
/// derives from, or null if it does not reduce to one.
BlockArgument traceToMLIRBlockArg(Value v);

/// Re-emit `v` with every step performed in `index`, or return `v` unchanged
/// when that cannot be done value-preservingly, or when there is nothing to
/// lift. Emits nothing in either of those cases.
Value rebuildInIndexDomain(OpBuilder &b, Location loc, Value v);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_INDEXDOMAIN_H
