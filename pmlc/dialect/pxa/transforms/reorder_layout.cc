// Copyright 2020 Intel Corporation

#include <numeric>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetOperations.h"

#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

namespace {

/// Pass that performs optimization of data layouts in order to align order
/// of elements in memory with order of iteration over these elements.
/// Aim of this optimization is to decrease distance between accesses to
/// same memory, which should result in lowering number of cache misses
/// and allow to leverage implicit prefetching of consecutive cachelines.
/// Overall flow of the algorithm is:
/// 1. Gather memory that is worth reordering. Several criteria influence
///    whether memory are be considered:
///    - memory is not local to `affine.parallel`;
///    - ratio of elements read to total size of memory is greater than 1;
///    - for now, only memory with single read operation is be considered;
///    - in future, whether memory is constant and can be reordered during
///      compilation.
/// 2. Select new layout by analyzing access patterns: affine maps of read
///    operations, loop order, vectorization, etc.
/// 3. Try to modify producer(s) of data to store into optimized layout
///    or create separate "reorder" operation.
///
/// Selecting of new layout is implemented by analyzing affine maps of read
/// operations, with constraints on loop variables, expanding number of
/// dimensions to reduce number of variables used in each dimension and
/// permuting those dimensions to better align with order of loops.
/// See: `generateLayoutChange`, `expandAffineMap`, `sortAffineMap`.
///
/// As an example given input with following structure:
/// ```mlir
///   %1 = alloc() : memref<1x58x58x64xf32>
///   %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) {
///     %7 = pxa.reduce assign %6, %1[0, %arg3 + 1, %arg4 + 1, %arg5]
///         : memref<1x58x58x64xf32>
///     // (d0, d1, d2) -> (0, d0 + 1, d1 + 1, d2)
///   }
///   %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 7, 8) {
///     %7 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (3, 3, 4) {
///       %12 = affine.parallel (%arg9, %arg10) = (0, 0) to (8, 2) {
///         %14 = pxa.vector_load %2[
///             0, %arg3 + %arg6, %arg4 * 8 + %arg7 + %arg9,
///             %arg8 * 16 + %arg10 * 8] : memref<1x58x58x64xf32>, vector<8xf32>
///         // (d0, d1, d2, d3, d4, d5, d6)
///         //     -> (0, d0 + d2, d1 * 8 + d3 + d4, d5 * 16 + d6 * 8)
///       }
///     }
///   }
/// ```
/// Expected output of the pass would be similar to:
/// ```mlir
///   %1 = alloc() : memref<1x58x4x58x2x8xf32>
///   %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) {
///     %7 = pxa.reduce assign %6, %1[
///         0, %arg3 + 1, %arg5 floordiv 16, %arg4 + 1, %arg5 floordiv 8 % 2,
///         %arg5 % 8] : memref<1x58x4x58x2x8xf32>
///     // (d0, d1, d2) ->
///     //     (0, d0 + 1, d2 floordiv 16, d1 + 1, d2 floordiv 8 % 2, d2 % 8)
///   }
///   %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 7, 8) {
///     %7 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (3, 3, 4) {
///       %12 = affine.parallel (%arg9, %arg10) = (0, 0) to (8, 2) {
///         %14 = pxa.vector_load %2[
///             0, %arg3 + %arg6, %arg8, %arg4 * 8 + %arg7 + %arg9, %arg10, 0]
///             : memref<1x58x4x58x2x8xf32>, vector<8xf32>
///         // (d0, d1, d2, d3, d4, d5, d6) ->
///         //     (0, d0 + d2, d5, d1 * 8 + d3 + d4, d6, 0)
///       }
///     }
///   }
/// ```
/// Please note above snippets are not valid IR, they have been heavily
/// formated for readability, uninteresting portions have been erased and
/// a lot of syntactic noise has been stripped.
class ReorderLayoutPass final : public ReorderLayoutBase<ReorderLayoutPass> {
public:
  void runOnFunction();
};

/// Structure holding information about single read operation.
struct MemoryReadDesc {
  // Operation this structure describes.
  PxaReadOpInterface readOp;
  // Affine map of read operation.
  mlir::AffineMap readMap;
  // Vectorization of read operation.
  mlir::SmallVector<uint64_t, 4> readVector;
  // Lower bounds of input dimensions in `readMap`.
  mlir::SmallVector<uint64_t, 6> dimensionLowerBounds;
  // Upper bounds of input dimensions in `readMap`.
  mlir::SmallVector<uint64_t, 6> dimensionUpperBounds;
  // Iteration order for input dimensions in `readMap`
  // (from least to most frequent).
  mlir::SmallVector<unsigned, 6> iterationOrder;
};

/// Structure describing memory and its usage.
struct MemoryUsageDesc {
  // IR value representing memory.
  mlir::Value value;
  // Shape of memory.
  mlir::SmallVector<uint64_t, 4> shape;
  // List of read descriptions accessing memory.
  mlir::SmallVector<MemoryReadDesc, 1> reads;
};

/// Gathers information about specified read operation.
MemoryReadDesc gatherReadDesc(PxaReadOpInterface op);

/// Returns MemoryUsageDesc initialized with information about `memory`,
/// without any information about its usage.
MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory);

/// Returns whether it is worth to change layout for given read operation.
/// In general it checks whether number of elements read is more than number
/// of elements memory holds.
bool isReadWorthLayoutChange(MemoryReadDesc &desc, uint64_t totalSize);

/// Returns whether any of read operations on memory make layout change
/// worth it.
bool isMemoryWorthLayoutChange(MemoryUsageDesc &desc);

/// Structure describing layout change in terms of affine map from previous
/// layout to new one. Additionally it holds shape and vectorization
/// of reordered memory as extracting this information from affine map
/// is not trivial.
struct ReorderDesc {
  // Map from original memory layout into target layout.
  mlir::AffineMap reorderMap;
  // Shape of memory after performing layout change.
  mlir::SmallVector<uint64_t, 6> reorderedShape;
  // Vectorization of memory after layout change. It has same number of
  // elements as original vectorization, but vectorized dimensions may
  // be moved to rightmost place.
  mlir::SmallVector<uint64_t, 6> reorderedVector;
};

/// Generates layout change that tries to match order of elements in memory
/// to order of iteration.
/// Returns llvm::None when better layout cannot be found.
///
/// New layout is selected by performing two transformations:
/// 1. Expanding number of dimensions to try to reduce number of loop variables
///    each dimension depends on. Additionally vectorized dimensions
///    are separated as non-empty dimensions not depending on any loop variable.
/// 2. Permuting/sorting separated dimensions in order of loops whose variables
///    are used in each dimension.
mlir::Optional<ReorderDesc>
generateLayoutChange(const MemoryUsageDesc &memoryDesc);

/// Inserts layout change into IR, replacing previous read and write operations.
/// If some write cannot be replaced, separate operation performing "reorder"
/// will be inserted.
///
/// Overall flow of the algorithm:
/// 1. Create memory with new shape.
/// 2. Replace all read operations to refer to new memory with modified
///    affine maps.
/// 3. Try to replace all write operations to refer to new memory with
///    modified affine maps.
/// 4. If replacing fails, insert explicit operation immediately after
///    `affine.parallel` writing to old memory, that copies it into memory
///     with new layout.
/// 5. If all writes were replaced and old memory is no longer used, remove it.
void foldOrCreateReorder(const MemoryUsageDesc &memoryDesc,
                         const ReorderDesc &reorderDesc);

// ============================================================================
// Helper affine map transformations
// ============================================================================

/// Simplify affine map given integral constraints. Returns the same map if it
/// cannot be simplified further.
///
/// Input:
///   map         = (d0, d1) -> (d0 + d1,
///                              (d0 * 16 + d1 * 8) floordiv 8 floordiv 2,
///                              (d0 * 16 + d1 * 8) floordiv 8 % 2, 0)
///   constraints = {0 <= d1 < 2, 0 <= d0 < 6}
///
/// Output:
///   (d0, d1) -> (d0 + d1, d0, d1, 0)
mlir::AffineMap simplifyConstrainedAffineMap(mlir::AffineMap map,
                                             mlir::IntegerSet constraints);

/// Expand affine map dimensions based on integral constraints and vector shape.
/// Aim of this transformation is to separate vector dimension and reduce
/// number of input dimensions each result dimension depends on.
///
/// Input:
///   map (A)     = (d0, d1) -> (d0 + d1, d0 * 16 + d1 * 8)
///   shape       = <7, 96>
///   vector      = <1, 8>
///   constraints = {0 <= d1 < 2, 0 <= d0 < 6}
/// Output:
///   reorder map (B)  = (d0, d1) -> (d0, d1 floordiv 8 floordiv 2,
///                                   d1 floordiv 8 % 2, 0)
///   reordered shape  = <7, 6, 2, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to expanded space composition
///       A o B can be used (with simplification).
/// A o B = (d0, d1) -> (d0 + d1, d0, d1, 0)
ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<uint64_t> shape,
                            mlir::ArrayRef<uint64_t> vector,
                            mlir::IntegerSet constraints);

/// Create affine permutation map that sorts resulting space dimensions in order
/// of increasing schedule.
/// Vectorized dimensions are alway put last.
/// In basic case dimension latest in schedule and used in expression determines
/// the order.
/// If two dimensions have same input dimension as appearing latest in schedule,
/// remaining dimensions specify their order.
/// If two output dimensions use exactly the same input dimensions in their
/// expressions, original order is preserved (stable sort).
///
/// Input:
///   map (A)    = (d0, d1) -> (d0 + d1, d0, d1, 0)
///   shape      = <7, 6, 2, 8>
///   vector     = <1, 1, 1, 8>
///   schedule   = <1, 0>
/// Output:
///   reorder map (B)  = (d0, d1, d2, d3) -> (d2, d0, d1, d3)
///   reordered shape  = <2, 7, 6, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to sorted space composition
///       A o B can be used.
///   A o B = (d0, d1) -> (d1, d0 + d1, d0, 0)
ReorderDesc sortAffineMap(mlir::AffineMap map, mlir::ArrayRef<uint64_t> shape,
                          mlir::ArrayRef<uint64_t> vector,
                          mlir::ArrayRef<unsigned> schedule);

/// Tile affine map using integral constraints to optimize specified schedule.
/// Returns llvm::None if current affine map is already optimal.
/// In essence this function first performs expansion of dimensions, then
/// sorts them according to schedule.
///
/// Input:
///   map (A)     = (d0, d1) -> (d0 + d1, d0 * 16 + d1 * 8)
///   shape       = <7, 96>
///   vector      = <1, 8>
///   constraints = {0 <= d1 < 2, 0 <= d0 < 6}
///   schedule    = <1, 0>
/// Output:
///   reorder map (B)  = (d0, d1) -> (d1 floordiv 8 % 2, d0,
///                                   d1 floordiv 8 floordiv 2, 0)
///   reordered shape  = <2, 7, 6, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to tiled space composition
///       A o B can be used (with simplification).
///   A o B = (d0, d1) -> (d1, d0 + d1, d0, 0)
mlir::Optional<ReorderDesc> tileAffineMap(mlir::AffineMap map,
                                          mlir::ArrayRef<uint64_t> shape,
                                          mlir::ArrayRef<uint64_t> vector,
                                          mlir::IntegerSet constraints,
                                          mlir::ArrayRef<unsigned> schedule);

// =============================================================================
// Implementation
// =============================================================================

void ReorderLayoutPass::runOnFunction() {
  mlir::FuncOp func = getFunction();
  // Gather read operations with memory not local to loop.
  mlir::DenseMap<mlir::Value, mlir::SmallVector<PxaReadOpInterface, 1>>
      valueReadListMap;
  for (auto parallelOp : func.getOps<mlir::AffineParallelOp>()) {
    mlir::Region &body = parallelOp.getLoopBody();
    parallelOp.walk([&](PxaReadOpInterface read) {
      mlir::AffineMap map = read.getAffineMap();
      mlir::Value memRef = read.getMemRef();
      mlir::Region *memParent = memRef.getParentRegion();
      // Skip memory local to `affine.parallel`.
      if (memParent == &body || body.isProperAncestor(memParent))
        return;
      valueReadListMap[memRef].push_back(read);
    });
  }
  // Gather descriptions and reject memory not worth changing layout.
  std::vector<MemoryUsageDesc> worthLayoutChange;
  for (auto &valueReadListPair : valueReadListMap) {
    // For now ignore memory with more than one usage to simplify the algorithm.
    if (valueReadListPair.second.size() != 1)
      continue;
    MemoryUsageDesc memoryDesc = getEmptyUsageDesc(valueReadListPair.first);
    for (PxaReadOpInterface read : valueReadListPair.second)
      memoryDesc.reads.emplace_back(gatherReadDesc(read));

    if (isMemoryWorthLayoutChange(memoryDesc))
      worthLayoutChange.emplace_back(std::move(memoryDesc));
  }

  for (MemoryUsageDesc &memoryDesc : worthLayoutChange) {
    IVLOG(3, "Optimizing layout for " << mlir::debugString(memoryDesc.value));
    IVLOG(3, "map: " << mlir::debugString(memoryDesc.reads.front().readMap));
    // Select new layout.
    mlir::Optional<ReorderDesc> optReorderDesc =
        generateLayoutChange(memoryDesc);
    if (!optReorderDesc.hasValue()) {
      IVLOG(3, "Layout already optimal");
      continue;
    }
    ReorderDesc &reorderDesc = optReorderDesc.getValue();
    IVLOG(3, "Optimized layout: " << mlir::debugString(reorderDesc.reorderMap));
    // Materialize layout change.
    foldOrCreateReorder(memoryDesc, optReorderDesc.getValue());
  }
}

MemoryReadDesc gatherReadDesc(PxaReadOpInterface op) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<uint64_t, 4> readVec(shapeRef.size(), 1);
  mlir::SmallVector<uint64_t, 6> dimensionLowerBounds;
  mlir::SmallVector<uint64_t, 6> dimensionUpperBounds;
  mlir::SmallVector<unsigned, 6> iterationOrder;
  mlir::SmallVector<mlir::BlockArgument, 6> argsOrder;
  unsigned idx = 0;
  for (mlir::Value dimVal : op.getMapOperands()) {
    auto arg = dimVal.dyn_cast<mlir::BlockArgument>();
    mlir::Operation *parent = arg.getOwner()->getParentOp();
    if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(parent)) {
      mlir::AffineExpr lower =
          parallelOp.getLowerBoundsValueMap().getResult(arg.getArgNumber());
      if (auto lowerConst = lower.dyn_cast<mlir::AffineConstantExpr>())
        dimensionLowerBounds.push_back(lowerConst.getValue());
      else
        dimensionLowerBounds.push_back(0);

      mlir::AffineExpr upper =
          parallelOp.getUpperBoundsValueMap().getResult(arg.getArgNumber());
      if (auto upperConst = upper.dyn_cast<mlir::AffineConstantExpr>())
        dimensionUpperBounds.push_back(upperConst.getValue());
      else
        dimensionUpperBounds.push_back(-1);
    }
    auto iterationIt = iterationOrder.begin();
    auto ordIt = argsOrder.begin();
    while (ordIt != argsOrder.end()) {
      mlir::Operation *ordArgParent = ordIt->getOwner()->getParentOp();
      if (ordArgParent != parent && parent->isProperAncestor(ordArgParent))
        break;
      if (ordArgParent == parent && arg.getArgNumber() < ordIt->getArgNumber())
        break;
      iterationIt++;
      ordIt++;
    }
    argsOrder.insert(ordIt, arg);
    iterationOrder.insert(iterationIt, idx);
    idx++;
  }
  return MemoryReadDesc{op,
                        op.getAffineMap(),
                        std::move(readVec),
                        std::move(dimensionLowerBounds),
                        std::move(dimensionUpperBounds),
                        std::move(iterationOrder)};
}

MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory) {
  auto memoryType = memory.getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shapeRef = memoryType.getShape();
  mlir::SmallVector<uint64_t, 4> shape;
  for (const int64_t &sh : shapeRef)
    shape.push_back(static_cast<uint64_t>(sh));
  return MemoryUsageDesc{memory, shape};
}

bool isReadWorthLayoutChange(MemoryReadDesc &desc, uint64_t totalSize) {
  // Read using only one variable can't really be reordered.
  if (desc.readMap.getNumInputs() == 1)
    return false;
  uint64_t readCount = 1;
  for (unsigned idx = 0; idx < desc.dimensionLowerBounds.size(); ++idx) {
    const uint64_t &lower = desc.dimensionLowerBounds[idx];
    const uint64_t &upper = desc.dimensionUpperBounds[idx];
    readCount *= upper - lower;
  }
  if (auto shapedResult =
          desc.readOp.getValue().getType().dyn_cast<mlir::ShapedType>())
    readCount *= shapedResult.getNumElements();
  return readCount > totalSize;
}

bool isMemoryWorthLayoutChange(MemoryUsageDesc &desc) {
  uint64_t totalSize =
      std::accumulate(desc.shape.begin(), desc.shape.end(),
                      /*init=*/(uint64_t)1, std::multiplies<uint64_t>());
  for (MemoryReadDesc &readDesc : desc.reads) {
    if (isReadWorthLayoutChange(readDesc, totalSize))
      return true;
  }
  return false;
}

mlir::Optional<ReorderDesc>
generateLayoutChange(const MemoryUsageDesc &memoryDesc) {
  const MemoryReadDesc &readDesc = memoryDesc.reads.front();
  // TODO: Fill constraints with upper and lower bounds of read dimensions.
  mlir::IntegerSet constraints;
  return tileAffineMap(readDesc.readMap, memoryDesc.shape, readDesc.readVector,
                       constraints, readDesc.iterationOrder);
}

void foldOrCreateReorder(const MemoryUsageDesc &memoryDesc,
                         const ReorderDesc &reorderDesc) {
  // TODO: Implement.
  // Create new memory with desired shape.
  // Try to replace operations writing to old memory with new modified memory
  // and transformed affine map. If this fails insert explicit reorder
  // operation. Replace read operations with new memory, new vector shape and
  // transformed affine map.
}

mlir::AffineMap simplifyConstrainedAffineMap(mlir::AffineMap map,
                                             mlir::IntegerSet constraints) {
  // TODO: Implement, identity for now.
  return map;
}

ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<uint64_t> shape,
                            mlir::ArrayRef<uint64_t> vector,
                            mlir::IntegerSet constraints) {
  // TODO: Implement, identity for now.
  auto reorderMap = mlir::AffineMap::getMultiDimIdentityMap(map.getNumResults(),
                                                            map.getContext());
  mlir::SmallVector<uint64_t, 6> reorderedShape(shape.begin(), shape.end());
  mlir::SmallVector<uint64_t, 6> reorderedVector(vector.begin(), vector.end());
  return ReorderDesc{reorderMap, reorderedShape, reorderedVector};
}

ReorderDesc sortAffineMap(mlir::AffineMap map, mlir::ArrayRef<uint64_t> shape,
                          mlir::ArrayRef<uint64_t> vector,
                          mlir::ArrayRef<unsigned> schedule) {
  // Small trick with order induced by norm for sorting.
  // For schedule <s0, s1, s2, .., s*n*>, each expression can be thought
  // as boolean vector, where i-th coordinate signifies wheter expression uses
  // i-th dimension from schedule.
  //
  // To transform such vector into norm with desired properties follwoing can
  // be used:
  // 1. Reverse values to the left of rightmost "1", ie:
  //    <a, b, c, 1, 0...> -> <c, b, a, 1, 0...>
  // 2. Negate values to the left of rightmost 1, ie:
  //    <c, b, a, 1, 0...> -> <~c, ~b, ~a, 1, 0...>
  // Next this vector can be simply reinterpreted as binary number giving
  // desired norm.
  // To handle vectorized dimensions just set all bits to one giving largest
  // representable number.
  // As a side-effect more than 31 dimensions cannot be handled with uint32_t
  // and constant dimensions always have lowest norm.
  // TODO: Add some fallback for larger number of dimensions.
  mlir::SmallVector<uint32_t, 6> scheduleNorms;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (vector[i] != 1)
      scheduleNorms.push_back(static_cast<uint32_t>(-1));
    uint32_t norm = 0;
    mlir::AffineExpr expr = map.getResult(i);
    unsigned shMax = schedule.size();
    for (; shMax > 0; --shMax) {
      unsigned dim = schedule[shMax - 1];
      if (!expr.isFunctionOfDim(dim))
        continue;
      norm = 1;
      break;
    }
    for (unsigned sh = 0; sh < shMax; ++sh) {
      unsigned dim = schedule[sh];
      norm = (norm << 1) | !expr.isFunctionOfDim(dim);
    }
    scheduleNorms.push_back(norm);
  }

  mlir::SmallVector<unsigned, 6> dimsPermutation;
  for (unsigned i = 0; i < map.getNumResults(); ++i)
    dimsPermutation.push_back(i);

  std::stable_sort(dimsPermutation.begin(), dimsPermutation.end(),
                   [&](const unsigned &a, const unsigned &b) {
                     return scheduleNorms[a] < scheduleNorms[b];
                   });

  auto reorderMap =
      mlir::AffineMap::getPermutationMap(dimsPermutation, map.getContext());
  mlir::SmallVector<uint64_t, 6> reorderedShape;
  mlir::SmallVector<uint64_t, 6> reorderedVector;
  for (unsigned perm : dimsPermutation) {
    reorderedShape.push_back(shape[perm]);
    reorderedVector.push_back(vector[perm]);
  }
  return ReorderDesc{reorderMap, reorderedShape, reorderedVector};
}

mlir::Optional<ReorderDesc> tileAffineMap(mlir::AffineMap map,
                                          mlir::ArrayRef<uint64_t> shape,
                                          mlir::ArrayRef<uint64_t> vector,
                                          mlir::IntegerSet constraints,
                                          mlir::ArrayRef<unsigned> schedule) {
  ReorderDesc expansion = expandAffineMap(map, shape, vector, constraints);
  mlir::AffineMap expanded = map.compose(expansion.reorderMap);
  mlir::AffineMap expandedSimple =
      simplifyConstrainedAffineMap(expanded, constraints);
  ReorderDesc sort = sortAffineMap(expandedSimple, expansion.reorderedShape,
                                   expansion.reorderedVector, schedule);
  // Only sorting can change actual layout, expansion preserves indices after
  // linearization to 1D.
  if (sort.reorderMap.isIdentity())
    return llvm::None;

  return ReorderDesc{expansion.reorderMap.compose(sort.reorderMap),
                     sort.reorderedShape, sort.reorderedVector};
}

} // namespace

std::unique_ptr<mlir::Pass> createReorderLayoutPass() {
  return std::make_unique<ReorderLayoutPass>();
}

} // namespace pmlc::dialect::pxa
