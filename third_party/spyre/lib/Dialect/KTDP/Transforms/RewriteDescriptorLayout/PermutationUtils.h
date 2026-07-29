#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::triton::ktdp {

enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2 };

/// Apply one coordinate op to a static (compile-time) logical extent.
/// Returns a non-kDynamic int64 on success, or std::nullopt when the result
/// is dynamic (i.e. needs a runtime SSA value).
inline std::optional<int64_t> applyStatic(int64_t logical, CoordOp op,
                                          int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    if (logical == mlir::ShapedType::kDynamic)
      return std::nullopt;
    return logical;
  case CoordOp::FloorDiv:
    if (logical == mlir::ShapedType::kDynamic)
      return std::nullopt;
    return arg == 0 ? std::optional<int64_t>(std::nullopt)
                    : std::optional<int64_t>((logical + arg - 1) / arg);
  case CoordOp::Mod:
    return arg;
  }
  return std::nullopt;
}

/// Compute physical static extents from logical static extents via a coord map.
/// Returns true on success (all physical extents are static).
inline bool applyCoordMap(llvm::ArrayRef<int64_t> logSizes,
                          llvm::ArrayRef<int64_t> physSrc,
                          llvm::ArrayRef<int64_t> physOp,
                          llvm::ArrayRef<int64_t> physArg,
                          llvm::SmallVectorImpl<int64_t> &out) {
  unsigned physRank = physSrc.size();
  out.resize(physRank);
  for (unsigned k = 0; k < physRank; ++k) {
    auto sz = applyStatic(logSizes[physSrc[k]],
                          static_cast<CoordOp>(physOp[k]), physArg[k]);
    if (!sz)
      return false;
    out[k] = *sz;
  }
  return true;
}

/// Compute the permutation that reorders opTileDims from physical to canonical
/// axis order. Returns empty vector if already identity (no transpose needed).
inline llvm::SmallVector<int64_t> computeTransposePerm(
    llvm::ArrayRef<int> opTileDims,
    llvm::ArrayRef<int64_t> dimRoles,
    llvm::ArrayRef<int64_t> canonicalAxes) {
  unsigned nTile = opTileDims.size();
  llvm::SmallVector<int64_t> perm(nTile, -1);
  llvm::SmallVector<bool> used(nTile, false);

  // First pass: match parallel dims (role >= 0) — unique role values.
  for (unsigned c = 0; c < nTile; ++c) {
    int64_t canonRole = canonicalAxes[c];
    if (canonRole == -1) continue;
    for (unsigned j = 0; j < nTile; ++j) {
      if (!used[j] && dimRoles[opTileDims[j]] == canonRole) {
        perm[j] = (int64_t)c;
        used[j] = true;
        break;
      }
    }
  }
  // Second pass: match reduction dims (role == -1) left-to-right.
  for (unsigned c = 0; c < nTile; ++c) {
    if (canonicalAxes[c] != -1) continue;
    for (unsigned j = 0; j < nTile; ++j) {
      if (!used[j] && dimRoles[opTileDims[j]] == -1) {
        perm[j] = (int64_t)c;
        used[j] = true;
        break;
      }
    }
  }
  // Check if identity.
  bool isIdentity = true;
  for (unsigned j = 0; j < nTile; ++j)
    if (perm[j] != (int64_t)j) { isIdentity = false; break; }
  return isIdentity ? llvm::SmallVector<int64_t>{} : perm;
}

/// Invert a permutation vector.
inline llvm::SmallVector<int64_t> invertPerm(llvm::ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> inv(perm.size());
  for (unsigned i = 0; i < perm.size(); ++i)
    inv[perm[i]] = i;
  return inv;
}

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H
