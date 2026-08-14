// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_TRANSFORMS_PARSER_SYMBOLIC_EXECUTION_H
#define P4MLIR_TRANSFORMS_PARSER_SYMBOLIC_EXECUTION_H

#include <optional>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR {

using StackId = mlir::Attribute;

struct StackAccess {
    StackId key;
    size_t size; // Size is stored for OOB checks
    unsigned count = 1;  // number of .next accesses to this stack
};

struct StackNumbering {
    llvm::DenseMap<mlir::Value, unsigned> valueIds;
    unsigned nextId = 0;
    unsigned forValue(mlir::Value base) {
        auto [it, inserted] = valueIds.try_emplace(base, nextId);
        if (inserted) ++nextId;
        return it->second;
    }
};

using ValueMap = llvm::DenseMap<StackId, mlir::TypedAttr>;
using AccessMap = llvm::DenseMap<P4HIR::ParserStateOp, llvm::SmallVector<StackAccess>>;

class IndexMap {
 public:
    bool isOOBFor(const StackAccess &access) const;
    bool isOOBForAny(llvm::ArrayRef<StackAccess> accesses) const;
    IndexMap advanced(llvm::ArrayRef<StackAccess> accesses) const;
    IndexMap restrictTo(llvm::ArrayRef<StackAccess> relevant) const;
    unsigned indexOf(StackId key) const;
    mlir::ArrayAttr encode(mlir::MLIRContext *context) const;

 private:
    llvm::DenseMap<StackId, unsigned> data_;
};

using VisitedKey = std::pair<P4HIR::ParserStateOp, mlir::ArrayAttr>;

VisitedKey makeVisitedKey(mlir::MLIRContext *context, P4HIR::ParserStateOp state,
                          const IndexMap &indexMap, const ValueMap &valueMap);

struct SymbolicState {
    P4HIR::ParserStateOp state;
    unsigned callIndex;
    IndexMap indexMap;
    ValueMap entryValueMap;
    P4HIR::ParserStateOp cloneOp;
};

struct SymbolicResult {
    llvm::SmallVector<SymbolicState> states;
    llvm::DenseMap<VisitedKey, std::optional<unsigned>> visitedMap;
    AccessMap accesses;
    llvm::DenseSet<StackId> indexVars;

    mlir::FailureOr<std::optional<unsigned>> lookupSuccessor(
        P4HIR::ParserStateOp state, const IndexMap &indexMap,
        const ValueMap &valueMap) const;
};

using RelevantStacksProvider =
    llvm::function_ref<llvm::ArrayRef<StackAccess>(P4HIR::ParserStateOp)>;

StackId makeStackId(mlir::Builder &attrBuilder, int64_t base, llvm::ArrayRef<int64_t> path);
bool stackIdLess(StackId lhs, StackId rhs);
std::string renderStackId(StackId id);
mlir::FailureOr<size_t> stackSizeOf(mlir::Type type);
mlir::FailureOr<StackId> getStackId(mlir::Value value, StackNumbering &numbering);

mlir::TypedAttr foldToConstAttr(mlir::Value value, const ValueMap &valueMap,
                                StackNumbering &numbering);
mlir::FailureOr<llvm::APSInt> foldToConstInt(mlir::Value value, const ValueMap &valueMap,
                                             StackNumbering &numbering);

ValueMap interpretState(P4HIR::ParserStateOp state, ValueMap valueMap,
                        llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess,
                        StackNumbering &numbering);

void collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out,
                        StackNumbering &numbering);
llvm::DenseSet<StackId> collectIndexVars(P4HIR::ParserOp parser, StackNumbering &numbering);
ValueMap restrictValueMap(const ValueMap &valueMap, const llvm::DenseSet<StackId> &keep);

void canonicalizeParserVariables(P4HIR::ParserOp parser, StackNumbering &numbering);

mlir::FailureOr<llvm::SmallVector<StackAccess>> computeStackAccesses(
    P4HIR::ParserStateOp state, StackNumbering &numbering);

SymbolicResult runSymbolicExecution(
    P4HIR::ParserOp parser, AccessMap stateAccesses, StackNumbering &numbering,
    RelevantStacksProvider getRelevantStacks,
    unsigned symbolicExecutionLimit,
    const llvm::DenseSet<P4HIR::ParserStateOp> &skipStates = {},
    const llvm::DenseSet<StackId> &extraIndexVars = {});

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_PARSER_SYMBOLIC_EXECUTION_H
