// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/Pass/Pass.h"

#include "pmlc/util/ids.h"

namespace pmlc::transforms {

#define GEN_PASS_CLASSES
#include "pmlc/transforms/passes.h.inc"

} // namespace pmlc::transforms
