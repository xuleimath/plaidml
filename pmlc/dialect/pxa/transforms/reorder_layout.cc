// Copyright 2020, Intel Corporation

#include <numeric>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/pxa/analysis/affine_constraints.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/layout_utils.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

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
  mlir::SmallVector<int64_t, 4> readVector;
  // Constraints for dimensions of `readMap`.
  mlir::FlatAffineConstraints dimensionConstraints;
  // Iteration order for input dimensions in `readMap`
  // (from least to most frequent).
  mlir::SmallVector<unsigned, 6> iterationOrder;
};

/// Structure holding information about single write operation.
struct MemoryWriteDesc {
  mlir::SmallVector<int64_t, 4> writeVector;
};

/// Structure describing memory and its usage.
struct MemoryUsageDesc {
  // IR value representing memory.
  mlir::Value value;
  // Shape of memory.
  mlir::SmallVector<int64_t, 4> shape;
  // List of descriptions of reads accessing memory.
  mlir::SmallVector<MemoryReadDesc, 1> reads;
  // List of descriptions of writes accessing memory.
  mlir::SmallVector<MemoryWriteDesc, 1> writes;
  // Number of elements in memory.
  int64_t count;
};

/// Gathers information about specified read operation.
MemoryReadDesc gatherReadDesc(PxaReadOpInterface op);

/// Gathers information about specified write operation.
MemoryWriteDesc gatherWriteDesc(PxaReduceOpInterface op);

/// Returns MemoryUsageDesc initialized with information about `memory`,
/// without any information about its usage.
MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory);

/// Returns whether it is worth to change layout for given read operation.
/// In general it checks whether number of elements read is more than number
/// of elements memory holds.
bool isReadWorthLayoutChange(MemoryReadDesc &desc, int64_t totalSize);

/// Returns whether any of read operations on memory make layout change
/// worth it.
bool isMemoryWorthLayoutChange(MemoryUsageDesc &desc);

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
mlir::Optional<ReorderDesc> generateLayoutChange(MemoryUsageDesc &memoryDesc);

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
void foldOrCreateReorder(MemoryUsageDesc &memoryDesc, ReorderDesc &reorderDesc,
                         bool allowReorder);

// ============================================================================
// Helper affine map transformations
// ============================================================================

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
ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> vector,
                            mlir::FlatAffineConstraints &constraints);

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
mlir::AffineMap sortAffineMap(mlir::AffineMap map,
                              mlir::ArrayRef<int64_t> vector,
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
mlir::Optional<ReorderDesc>
tileAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
              mlir::ArrayRef<int64_t> vector,
              mlir::FlatAffineConstraints constraints,
              mlir::ArrayRef<unsigned> schedule);

// =============================================================================
// Implementation
// =============================================================================

void ReorderLayoutPass::runOnFunction() {
  mlir::FuncOp func = getFunction();
  // Gather read operations with memory not local to loop.
  mlir::DenseMap<mlir::Value, mlir::SmallVector<PxaReadOpInterface, 1>>
      valueReadListMap;
  mlir::DenseMap<mlir::Value, mlir::SmallVector<PxaReduceOpInterface, 1>>
      valueReduceListMap;
  for (auto parallelOp : func.getOps<mlir::AffineParallelOp>()) {
    mlir::Region &body = parallelOp.getLoopBody();
    parallelOp.walk([&](PxaReadOpInterface read) {
      mlir::Value memRef = read.getMemRef();
      mlir::Value indirectDef = getIndirectDef(memRef);
      // Skip memory local to `affine.parallel`.
      if (!parallelOp.isDefinedOutsideOfLoop(indirectDef))
        return;
      valueReadListMap[indirectDef].push_back(read);
    });
    parallelOp.walk([&](PxaReduceOpInterface reduce) {
      mlir::Value memRef = reduce.getMemRef();
      mlir::Value indirectDef = getIndirectDef(memRef);
      // Skip memory local to `affine.parallel`.
      if (!parallelOp.isDefinedOutsideOfLoop(indirectDef))
        return;
      valueReduceListMap[indirectDef].push_back(reduce);
    });
  }
  // Gather descriptions and reject memory not worth changing layout.
  std::vector<MemoryUsageDesc> worthLayoutChange;
  for (auto &valueReadListPair : valueReadListMap) {
    mlir::Value memory = valueReadListPair.first;
    MemoryUsageDesc memoryDesc = getEmptyUsageDesc(memory);
    for (PxaReadOpInterface read : valueReadListPair.second)
      memoryDesc.reads.emplace_back(gatherReadDesc(read));
    for (PxaReduceOpInterface reduce : valueReduceListMap.lookup(memory))
      memoryDesc.writes.emplace_back(gatherWriteDesc(reduce));

    if (isMemoryWorthLayoutChange(memoryDesc))
      worthLayoutChange.emplace_back(std::move(memoryDesc));
  }
  // Generate new layout and materialize it by converting operations
  // or creating reorder.
  for (MemoryUsageDesc &memoryDesc : worthLayoutChange) {
    IVLOG(3, "Optimizing layout for " << mlir::debugString(memoryDesc.value));
    // Select new layout.
    mlir::Optional<ReorderDesc> optReorderDesc =
        generateLayoutChange(memoryDesc);
    if (!optReorderDesc.hasValue()) {
      IVLOG(3, "Could not select more optimal layout");
      continue;
    }
    ReorderDesc &reorderDesc = optReorderDesc.getValue();
    IVLOG(3, "Optimized layout: " << mlir::debugString(reorderDesc.reorderMap));
    // Materialize layout change.
    foldOrCreateReorder(memoryDesc, reorderDesc, /*allowReorder=*/true);
  }
}

MemoryReadDesc gatherReadDesc(PxaReadOpInterface op) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<int64_t, 4> readVec(shapeRef.size(), 1);
  if (auto vecRead = mlir::dyn_cast<PxaVectorLoadOp>(op.getOperation())) {
    auto vecType = vecRead.getType().cast<mlir::VectorType>();
    mlir::ArrayRef<int64_t> vecShape = vecType.getShape();
    for (unsigned idx = 0; idx < vecShape.size(); ++idx)
      readVec[readVec.size() - vecShape.size() + idx] = vecShape[idx];
  }
  mlir::AffineMap readMap = op.getAffineMap();
  mlir::Operation::operand_range mapOperands = op.getMapOperands();
  mlir::FlatAffineConstraints dimensionConstraints =
      gatherAffineMapConstraints(mlir::AffineValueMap(readMap, mapOperands));
  mlir::SmallVector<unsigned, 6> iterationOrder;
  mlir::SmallVector<mlir::BlockArgument, 6> argsOrder;
  unsigned idx = 0;
  for (mlir::Value dimVal : mapOperands) {
    auto arg = dimVal.dyn_cast<mlir::BlockArgument>();
    mlir::Operation *parent = arg.getOwner()->getParentOp();
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

  auto topLevelParallel = op.getParentOfType<mlir::AffineParallelOp>();
  while (auto nextLevelParallel =
             topLevelParallel.getParentOfType<mlir::AffineParallelOp>())
    topLevelParallel = nextLevelParallel;
  // Fixup for sub-group size equal to 1. Backend will subgroup along
  // first parallel dimension, so it needs to be moved to the front.
  if (getIntegerTag(topLevelParallel, subgroupSizeTag(), 1) == 1 &&
      iterationOrder.size() >= 2) {
    unsigned subGrouped = iterationOrder.front();
    iterationOrder.erase(iterationOrder.begin());
    iterationOrder.push_back(subGrouped);
  }

  return MemoryReadDesc{op, op.getAffineMap(), std::move(readVec),
                        std::move(dimensionConstraints),
                        std::move(iterationOrder)};
}

MemoryWriteDesc gatherWriteDesc(PxaReduceOpInterface op) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<int64_t, 4> reduceVec(shapeRef.size(), 1);
  if (auto vecReduce = mlir::dyn_cast<PxaVectorReduceOp>(op.getOperation())) {
    auto vecType = vecReduce.getVectorType();
    mlir::ArrayRef<int64_t> vecShape = vecType.getShape();
    for (unsigned idx = 0; idx < vecShape.size(); ++idx)
      reduceVec[reduceVec.size() - vecShape.size() + idx] = vecShape[idx];
  }
  return MemoryWriteDesc{std::move(reduceVec)};
}

MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory) {
  auto memoryType = memory.getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shapeRef = memoryType.getShape();
  mlir::SmallVector<int64_t, 4> shape(shapeRef.begin(), shapeRef.end());
  auto desc = MemoryUsageDesc{memory, shape};
  desc.count = std::accumulate(shapeRef.begin(), shapeRef.end(),
                               /*init=*/(int64_t)1, std::multiplies<int64_t>());
  return desc;
}

bool isReadWorthLayoutChange(MemoryReadDesc &desc, int64_t totalSize) {
  // Read using only one variable can't really be reordered.
  if (desc.readMap.getNumInputs() == 1)
    return false;
  return true;
  // TODO: Below code will reject weights for convolutions, we should
  //       reorder constant data even if it has low in-thread reuse.
  // uint64_t readCount = 1;
  // for (unsigned idx = 0; idx < desc.dimensionConstraints.getNumDimIds();
  // ++idx) {
  //   mlir::Optional<int64_t> lower =
  //   desc.dimensionConstraints.getConstantLowerBound(idx);
  //   mlir::Optional<int64_t> upper =
  //   desc.dimensionConstraints.getConstantUpperBound(idx);
  //
  //   readCount *= (upper.getValue() - lower.getValue());
  // }
  // if (auto shapedResult =
  //         desc.readOp.getValue().getType().dyn_cast<mlir::ShapedType>())
  //   readCount *= shapedResult.getNumElements();
  // return readCount > totalSize;
}

bool isMemoryWorthLayoutChange(MemoryUsageDesc &desc) {
  for (MemoryReadDesc &readDesc : desc.reads) {
    if (isReadWorthLayoutChange(readDesc, desc.count))
      return true;
  }
  return false;
}

mlir::LogicalResult selectCommonVectorization(MemoryUsageDesc &memoryDesc,
                                              mlir::ArrayRef<int64_t> &result) {
  bool isResultUnit = false;
  auto isUnitVector = [](mlir::ArrayRef<int64_t> vec) {
    return std::all_of(vec.begin(), vec.end(),
                       [](int64_t val) { return val == 1; });
  };

  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    mlir::ArrayRef<int64_t> readVector = readDesc.readVector;
    if (result.empty() || isResultUnit) {
      result = readVector;
      isResultUnit = isUnitVector(readVector);
      continue;
    }
    if (isUnitVector(readVector))
      continue;
    if (!std::equal(result.begin(), result.end(), readVector.begin()))
      return mlir::failure();
  }
  for (MemoryWriteDesc &writeDesc : memoryDesc.writes) {
    mlir::ArrayRef<int64_t> writeVector = writeDesc.writeVector;
    if (result.empty() || isResultUnit) {
      result = writeVector;
      isResultUnit = isUnitVector(writeVector);
      continue;
    }
    if (isUnitVector(writeVector))
      continue;
    if (!std::equal(result.begin(), result.end(), writeVector.begin()))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::Optional<ReorderDesc> generateLayoutChange(MemoryUsageDesc &memoryDesc) {
  mlir::Optional<ReorderDesc> selectedReorder = llvm::None;
  mlir::ArrayRef<int64_t> commonVector;
  if (mlir::failed(selectCommonVectorization(memoryDesc, commonVector))) {
    IVLOG(3, "  Inconsistent vectorization between reads and writes");
    return llvm::None;
  }
  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    if (!isReadWorthLayoutChange(readDesc, memoryDesc.count))
      continue;
    mlir::Optional<ReorderDesc> reorder =
        tileAffineMap(readDesc.readMap, memoryDesc.shape, commonVector,
                      readDesc.dimensionConstraints, readDesc.iterationOrder);
    if (!reorder.hasValue())
      return llvm::None;
    if (!selectedReorder.hasValue()) {
      selectedReorder = reorder;
      continue;
    }
    if (selectedReorder.getValue().reorderMap !=
        reorder.getValue().reorderMap) {
      IVLOG(3, "  Inconsistent layout between reads");
      return llvm::None;
    }
  }
  return selectedReorder;
}

void foldOrCreateReorder(MemoryUsageDesc &memoryDesc, ReorderDesc &reorderDesc,
                         bool allowReorder) {
  if (mlir::succeeded(convertMemoryLayout(memoryDesc.value, reorderDesc)))
    return;
  if (!allowReorder) {
    IVLOG(3,
          "  Failed to change layout in-place, separate reorder not allowed");
    return;
  }
  IVLOG(3, "  Failed to change layout in-place, inserting reorder");
  mlir::DenseSet<mlir::Value> memoryToReorder;
  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    PxaReadOpInterface readOp = readDesc.readOp;
    mlir::Value readMem = readOp.getMemRef();
    memoryToReorder.insert(readMem);
  }
  for (mlir::Value reorderMem : memoryToReorder) {
    mlir::OpBuilder builder(reorderMem.getContext());
    builder.setInsertionPointAfterValue(reorderMem);
    // TODO: It should be fused location of all reads.
    reorderMemoryLayoutForReading(reorderMem.getLoc(), builder, reorderDesc,
                                  reorderMem);
  }
}

void expandAffineExpr(mlir::AffineExpr expr, mlir::AffineExpr dimExpr,
                      int64_t dimSize, int64_t vecSize,
                      mlir::FlatAffineConstraints &constraints,
                      mlir::SmallVectorImpl<mlir::AffineExpr> &expansionExprs,
                      mlir::SmallVectorImpl<int64_t> &expandedShape,
                      mlir::SmallVectorImpl<int64_t> &expandedVec) {
  auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  if (vecSize != 1) {
    expandAffineExpr(expr.floorDiv(vecSize), dimExpr.floorDiv(vecSize),
                     ceilDiv(dimSize, vecSize), 1, constraints, expansionExprs,
                     expandedShape, expandedVec);
    expansionExprs.push_back(dimExpr % vecSize);
    expandedShape.push_back(vecSize);
    expandedVec.push_back(vecSize);
    return;
  }
  if (expr.getKind() == mlir::AffineExprKind::Add) {
    auto addExpr = expr.cast<mlir::AffineBinaryOpExpr>();
    mlir::AffineExpr lhsExpr = addExpr.getLHS();
    mlir::AffineExpr rhsExpr = addExpr.getRHS();
    mlir::Optional<int64_t> lhsUpperBound = getUpperBound(lhsExpr, constraints);
    mlir::Optional<int64_t> rhsUpperBound = getUpperBound(rhsExpr, constraints);

    // Pattern e*i* + e*j*, where e*i* % N == 0 and e*j* < N.
    mlir::Optional<bool> caseRhsSmaller = rhsUpperBound.map(
        [&](int64_t val) { return lhsExpr.isMultipleOf(val + 1); });
    // Pattern e*i* + e*j*, where e*i* < N and e*j* % N == 0.
    mlir::Optional<bool> caseLhsSmaller = lhsUpperBound.map(
        [&](int64_t val) { return rhsExpr.isMultipleOf(val + 1); });

    if (caseRhsSmaller.getValueOr(false)) {
      int64_t divisor = rhsUpperBound.getValue() + 1;
      expandAffineExpr(lhsExpr.floorDiv(divisor), dimExpr.floorDiv(divisor),
                       ceilDiv(dimSize, divisor), vecSize, constraints,
                       expansionExprs, expandedShape, expandedVec);
      expandAffineExpr(rhsExpr, dimExpr % divisor, divisor, vecSize,
                       constraints, expansionExprs, expandedShape, expandedVec);
      return;
    }
    if (caseLhsSmaller.getValueOr(false)) {
      int64_t divisor = lhsUpperBound.getValue() + 1;
      expandAffineExpr(rhsExpr.floorDiv(divisor), dimExpr.floorDiv(divisor),
                       ceilDiv(dimSize, divisor), vecSize, constraints,
                       expansionExprs, expandedShape, expandedVec);
      expandAffineExpr(lhsExpr, dimExpr % divisor, divisor, vecSize,
                       constraints, expansionExprs, expandedShape, expandedVec);
      return;
    }
  }

  expansionExprs.push_back(dimExpr);
  expandedShape.push_back(dimSize);
  expandedVec.push_back(vecSize);
}

ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                            mlir::ArrayRef<int64_t> vector,
                            mlir::FlatAffineConstraints &constraints) {
  mlir::SmallVector<mlir::AffineExpr, 6> expansionExprs;
  mlir::SmallVector<int64_t, 6> expandedShape;
  mlir::SmallVector<int64_t, 6> expandedVec;
  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    mlir::AffineExpr dimExpr = mlir::getAffineDimExpr(idx, map.getContext());
    expandAffineExpr(map.getResult(idx), dimExpr, shape[idx], vector[idx],
                     constraints, expansionExprs, expandedShape, expandedVec);
  }
  auto reorderMap = mlir::AffineMap::get(map.getNumResults(), 0, expansionExprs,
                                         map.getContext());
  return ReorderDesc{reorderMap, expandedShape, expandedVec};
}

ReorderDesc sortAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                          mlir::ArrayRef<int64_t> vector,
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
  mlir::SmallVector<int64_t, 6> sortedShape;
  mlir::SmallVector<int64_t, 6> sortedVec;
  for (unsigned perm : dimsPermutation) {
    sortedShape.push_back(shape[perm]);
    sortedVec.push_back(vector[perm]);
  }
  return ReorderDesc{reorderMap, sortedShape, sortedVec};
}

mlir::Optional<ReorderDesc>
tileAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
              mlir::ArrayRef<int64_t> vector,
              mlir::FlatAffineConstraints constraints,
              mlir::ArrayRef<unsigned> schedule) {
  ReorderDesc expand = expandAffineMap(map, shape, vector, constraints);
  mlir::AffineMap expanded = expand.reorderMap.compose(map);
  mlir::AffineMap expandedSimple =
      simplifyMapWithConstraints(expanded, constraints);
  mlir::ArrayRef<int64_t> expandedShape = expand.reorderedShape;
  mlir::ArrayRef<int64_t> expandedVector = expand.reorderedVector;
  ReorderDesc sort =
      sortAffineMap(expandedSimple, expandedShape, expandedVector, schedule);
  // Only sorting can change actual layout, expansion preserves indices after
  // linearization to 1D.
  if (sort.reorderMap.isIdentity())
    return llvm::None;

  return ReorderDesc{sort.reorderMap.compose(expand.reorderMap),
                     sort.reorderedShape, sort.reorderedVector};
}

} // namespace

std::unique_ptr<mlir::Pass> createReorderLayoutPass() {
  return std::make_unique<ReorderLayoutPass>();
}

} // namespace pmlc::dialect::pxa
