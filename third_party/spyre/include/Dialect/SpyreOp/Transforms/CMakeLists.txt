# The -name= value decides the generated symbol names:
# registerSpyreOpTransformsPasses() for the whole group, and the GEN_PASS_*
# macro spellings the .cpp files use. "SpyreOpTransforms" rather than bare
# "SpyreOp" so the symbol cannot collide with a registration function the
# ktir-mlir-frontend submodule may one day generate for the dialect itself.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name=SpyreOpTransforms)
add_public_tablegen_target(SpyreOpTransformsIncGen)
