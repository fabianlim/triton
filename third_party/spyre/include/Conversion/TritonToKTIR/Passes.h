// Declarations for the TTIR -> KTIR conversions, in namespace
// mlir::triton::spyre.  One of five Passes.h under third_party/spyre: this one
// for passes that cross a dialect boundary, one per dialect whose own
// abstractions are a pass's subject -- Dialect/KTDP/Transforms/,
// Dialect/SpyreOp/Transforms/ and Dialect/TTS/Transforms/, each in that
// dialect's own namespace -- and Transforms/ for the rest.  The criterion and
// the per-pass contracts are in the Passes.td beside this file.

#ifndef TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H
#define TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir::triton::spyre {

#define GEN_PASS_DECL
#include "Conversion/TritonToKTIR/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToKTIR/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerDescriptorMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerScalarLoadPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerInterTilePass();
std::unique_ptr<OperationPass<ModuleOp>> createDistributeWorkPass(
    llvm::ArrayRef<int64_t> grid = {});

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H
