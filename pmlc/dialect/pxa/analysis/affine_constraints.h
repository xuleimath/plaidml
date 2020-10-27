// Copyright 2020, Intel Corporation
#pragma once

#include <utility>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"

namespace pmlc::dialect::pxa {

mlir::LogicalResult
addAffineParallelIVDomain(mlir::AffineParallelOp parallelOp, unsigned idx,
                          mlir::FlatAffineConstraints &constraints);

mlir::Optional<int64_t> getLowerBound(mlir::AffineExpr expr,
                                      mlir::FlatAffineConstraints &constraints);

mlir::Optional<int64_t> getUpperBound(mlir::AffineExpr expr,
                                      mlir::FlatAffineConstraints &constraints);

std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>
getLowerUpperBounds(mlir::AffineExpr expr,
                    mlir::FlatAffineConstraints &constraints);

mlir::AffineExpr
simplifyExprWithConstraints(mlir::AffineExpr expr,
                            mlir::FlatAffineConstraints &constraints);

/// Simplifies affine map given integral constraints.
/// Returns the same map if it cannot be simplified further.
mlir::AffineMap
simplifyMapWithConstraints(mlir::AffineMap map,
                           mlir::FlatAffineConstraints &constraints);

mlir::AffineValueMap simplifyMapWithConstraints(mlir::AffineValueMap map);

mlir::FlatAffineConstraints
gatherAffineMapConstraints(mlir::AffineValueMap map);

} // namespace pmlc::dialect::pxa
