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

class StackId {
 public:
    StackId() = default;

    static StackId make(mlir::MLIRContext *context, int64_t base,
                        llvm::ArrayRef<int64_t> path);

    bool operator==(const StackId &rhs) const { return attribute == rhs.attribute; }
    bool operator!=(const StackId &rhs) const { return attribute != rhs.attribute; }
    bool operator<(const StackId &rhs) const;

    std::string render() const;

    mlir::Attribute asAttribute() const { return attribute; }
    explicit operator bool() const { return static_cast<bool>(attribute); }

 private:
    friend struct llvm::DenseMapInfo<StackId>;
    explicit StackId(mlir::DenseI64ArrayAttr attr) : attribute(attr) {}
    mlir::DenseI64ArrayAttr attribute;
};

}  // namespace P4::P4MLIR

namespace llvm {
template <>
struct DenseMapInfo<P4::P4MLIR::StackId> {
    static P4::P4MLIR::StackId getEmptyKey() {
        return P4::P4MLIR::StackId(mlir::DenseI64ArrayAttr(
            DenseMapInfo<mlir::Attribute>::getEmptyKey().getImpl()));
    }
    static P4::P4MLIR::StackId getTombstoneKey() {
        return P4::P4MLIR::StackId(mlir::DenseI64ArrayAttr(
            DenseMapInfo<mlir::Attribute>::getTombstoneKey().getImpl()));
    }
    static unsigned getHashValue(const P4::P4MLIR::StackId &id) {
        return DenseMapInfo<mlir::Attribute>::getHashValue(id.attribute);
    }
    static bool isEqual(const P4::P4MLIR::StackId &lhs, const P4::P4MLIR::StackId &rhs) {
        return lhs == rhs;
    }
};
}  // namespace llvm

namespace P4::P4MLIR {

struct StackAccess {
    StackId key;
    size_t size; // Size is stored for OOB checks
    unsigned count = 1;  // number of .next accesses to this stack
};

class ValueMap {
 public:
    using MapType = llvm::DenseMap<StackId, mlir::TypedAttr>;
    using iterator = MapType::iterator;
    using const_iterator = MapType::const_iterator;

    ValueMap() = default;

    ValueMap restrictTo(const llvm::DenseSet<StackId> &keep) const;
    mlir::ArrayAttr encode(mlir::MLIRContext *context) const;

    mlir::TypedAttr &operator[](StackId key) { return data[key]; }
    iterator find(StackId key) { return data.find(key); }
    const_iterator find(StackId key) const { return data.find(key); }
    void erase(StackId key) { data.erase(key); }
    bool empty() const { return data.empty(); }
    auto insert(const std::pair<StackId, mlir::TypedAttr> &entry) { return data.insert(entry); }

    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

 private:
    MapType data;
};

class StackNumbering {
 public:
    unsigned forValue(mlir::Value base) {
        auto [it, inserted] = valueIds.try_emplace(base, nextId);
        if (inserted) ++nextId;
        return it->second;
    }

    void copyNumbering(mlir::Value original, mlir::Value cloned);

    mlir::FailureOr<StackId> getStackId(mlir::Value value);

    mlir::TypedAttr foldToConstAttr(mlir::Value value, const ValueMap &valueMap);
    mlir::FailureOr<llvm::APSInt> foldToConstInt(mlir::Value value, const ValueMap &valueMap);

    ValueMap interpretState(P4HIR::ParserStateOp state, ValueMap valueMap,
                            llvm::function_ref<void(mlir::Operation *, const ValueMap &)> onAccess);

    void collectVarsInIndex(mlir::Value value, llvm::DenseSet<StackId> &out);
    llvm::DenseSet<StackId> collectIndexVars(P4HIR::ParserOp parser);

    void canonicalizeParserVariables(P4HIR::ParserOp parser);

    mlir::FailureOr<llvm::SmallVector<StackAccess>> computeStackAccesses(
        P4HIR::ParserStateOp state);

 private:
    llvm::DenseMap<mlir::Value, unsigned> valueIds;
    unsigned nextId = 0;
};

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
    llvm::DenseMap<StackId, unsigned> data;
};

using VisitedKey = std::pair<P4HIR::ParserStateOp, mlir::ArrayAttr>;

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

    static VisitedKey makeVisitedKey(mlir::MLIRContext *context, P4HIR::ParserStateOp state,
                                     const IndexMap &indexMap, const ValueMap &valueMap);

    mlir::FailureOr<std::optional<unsigned>> lookupSuccessor(
        P4HIR::ParserStateOp state, const IndexMap &indexMap,
        const ValueMap &valueMap) const;
};

using RelevantStacksProvider =
    llvm::function_ref<llvm::ArrayRef<StackAccess>(P4HIR::ParserStateOp)>;

class SymbolicExecution {
 public:
    SymbolicExecution(P4HIR::ParserOp parser, AccessMap stateAccesses,
                      StackNumbering &numbering,
                      RelevantStacksProvider getRelevantStacks,
                      unsigned limit,
                      const llvm::DenseSet<P4HIR::ParserStateOp> &skipStates = {},
                      const llvm::DenseSet<StackId> &extraIndexVars = {});
    SymbolicResult run();

 private:
    llvm::SmallVector<P4HIR::ParserStateOp> resolveSuccessors(
        P4HIR::ParserStateOp state, const ValueMap &valueMap);
    ValueMap buildResolveValueMap(P4HIR::ParserStateOp state,
                                  const ValueMap &valueMap,
                                  const IndexMap &indexMap);

    P4HIR::ParserOp parser;
    AccessMap stateAccesses;
    StackNumbering &numbering;
    RelevantStacksProvider getRelevantStacks;
    unsigned limit;
    llvm::DenseSet<P4HIR::ParserStateOp> skipStates;
    llvm::DenseSet<StackId> extraIndexVars;
};

}  // namespace P4::P4MLIR

#endif  // P4MLIR_TRANSFORMS_PARSER_SYMBOLIC_EXECUTION_H
