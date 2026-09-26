// Declarations for transforms on the spyreop dialect's own intrinsics.  One of
// five Passes.h under third_party/spyre, and the second of the three in a
// dialect's own namespace (mlir::triton::spyreop, as Dialect/KTDP/Transforms/ is
// in mlir::triton::ktdp and Dialect/TTS/Transforms/ in mlir::triton::tts) -- it
// earns the dialect's name because the dialect is its subject.
// Conversion/TritonToKTIR/ holds those that cross a dialect boundary and
// Transforms/ the rest, both in mlir::triton::spyre.  The criterion and the
// per-pass contracts are in the Passes.td beside this file.
//
// A NAMING HAZARD this namespace carries, shared with mlir::triton::ktdp: the
// dialect itself is mlir::spyreop, so inside mlir::triton::spyreop the name
// `spyreop` resolves to the enclosing namespace and `spyreop::Reciprocal` does
// not compile.  Spell dialect ops mlir::spyreop::... in any code that sits
// inside this namespace.  The .cpp keeps its patterns in an anonymous namespace
// at file scope, where the short form works, so the hazard is confined to the
// GEN_PASS_DEF block and the create* definition.

#ifndef TRITON_SPYRE_DIALECT_SPYREOP_TRANSFORMS_PASSES_H
#define TRITON_SPYRE_DIALECT_SPYREOP_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::spyreop {

#define GEN_PASS_DECL
#include "Dialect/SpyreOp/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Dialect/SpyreOp/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createCombineSpyreOpsPass();

} // namespace mlir::triton::spyreop

#endif // TRITON_SPYRE_DIALECT_SPYREOP_TRANSFORMS_PASSES_H
