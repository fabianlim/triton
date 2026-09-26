# Transforms whose subject is the spyreop dialect's own intrinsics -- one
# spelling of a computation rewritten into another spelling of the same one, both
# spyreop.  The dialect comes from the ktir-mlir-frontend submodule and defines
# no canonicalizer of its own, so this is where such a peephole can live without
# a submodule change; Passes.td has the reasoning.
#
# LINK_LIBS derived from actual #includes plus Passes.td's dependentDialects.
# The absences are the content: no TritonIR, no ktdp, no linalg, no tensor, no
# arith, no math.  A pattern here has spyreop on both sides, so the only dialect
# this library reaches is spyreop -- which is exactly rule 1 in Passes.td, held
# by the build rather than only by the prose.  A link edge appearing here is a
# signal that a pattern has outgrown the pass.
add_triton_library(SpyreOpTransforms
  CombineSpyreOps.cpp

  DEPENDS
  SpyreOpTransformsIncGen

  LINK_LIBS PUBLIC
  KTIR::SpyreOpDialect
  MLIRIR
  MLIRPass
  # For applyPatternsGreedily, which is also what erases the numerator this pass
  # leaves dead rather than erasing it by hand.
  MLIRTransformUtils
)
