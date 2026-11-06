// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Builders.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-wrap-extract-next"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_WRAPEXTRACTNEXT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

static mlir::Value getFieldStack(mlir::Value fieldValue, llvm::StringRef fieldName) {
    auto fieldRef = fieldValue.getDefiningOp<P4HIR::StructFieldRefOp>();
    if (!fieldRef || fieldRef.getFieldName() != fieldName) return {};
    auto inputType = mlir::cast<P4HIR::ReferenceType>(fieldRef.getInput().getType());
    if (!mlir::isa<P4HIR::HeaderStackType>(inputType.getObjectType())) return {};
    return fieldRef.getInput();
}

static P4HIR::AssignOp findCopyOut(P4HIR::VariableOp variable) {
    for (auto *variableUser : variable->getUsers())
        if (auto read = mlir::dyn_cast<P4HIR::ReadOp>(variableUser))
            for (auto *readUser : read->getUsers())
                if (auto assign = mlir::dyn_cast<P4HIR::AssignOp>(readUser))
                    if (assign.getValue() == read.getResult()) return assign;
    return {};
}

static void wrapExtract(mlir::Value header) {
    auto variable = header.getDefiningOp<P4HIR::VariableOp>();
    if (!variable) return;
    auto assign = findCopyOut(variable);
    if (!assign) return;

    mlir::Value target = assign.getRef();
    if (auto memberRef = target.getDefiningOp<P4HIR::StructFieldRefOp>())
        if (memberRef.getInput().getDefiningOp<P4HIR::ArrayElementRefOp>())
            target = memberRef.getInput();
    auto elementRef = target.getDefiningOp<P4HIR::ArrayElementRefOp>();
    if (!elementRef) return;
    auto indexRead = elementRef.getIndex().getDefiningOp<P4HIR::ReadOp>();
    mlir::Value stack =
        getFieldStack(elementRef.getInput(), P4HIR::HeaderStackType::dataFieldName);
    if (!stack || !indexRead ||
        getFieldStack(indexRead.getRef(), P4HIR::HeaderStackType::nextIndexFieldName) != stack)
        return;

    mlir::Block *block = assign->getBlock();
    mlir::Operation *first = assign;
    for (mlir::Operation *anchor : {elementRef.getInput().getDefiningOp(),
                                    indexRead.getRef().getDefiningOp(), variable.getOperation()}) {
        if (anchor->getBlock() != block) return;
        if (anchor->isBeforeInBlock(first)) first = anchor;
    }
    if (auto *stackDefinition = stack.getDefiningOp();
        stackDefinition && stackDefinition->getBlock() == block &&
        !stackDefinition->isBeforeInBlock(first))
        return;

    auto end = std::next(assign->getIterator());
    for (auto &operation : llvm::make_range(first->getIterator(), end)) {
        for (auto *user : operation.getUsers()) {
            auto *ancestor = block->findAncestorOpInBlock(*user);
            if (!ancestor || ancestor->isBeforeInBlock(first) || assign->isBeforeInBlock(ancestor))
                return;
        }
    }

    mlir::OpBuilder builder(first);
    auto wrapper = P4HIR::HeaderStackExtractNextOp::create(builder, assign.getLoc(), stack);
    mlir::Block *body = builder.createBlock(&wrapper.getBody());
    body->getOperations().splice(body->end(), block->getOperations(), first->getIterator(), end);
}

struct WrapExtractNext : public impl::WrapExtractNextBase<WrapExtractNext> {
    void runOnOperation() override {
        llvm::SmallVector<mlir::Value> headers;
        getOperation()->walk([&](mlir::Operation *operation) {
            if (auto extract = mlir::dyn_cast<P4CoreLib::PacketExtractOp>(operation))
                headers.push_back(extract.getHdr());
            else if (auto extractVariable =
                         mlir::dyn_cast<P4CoreLib::PacketExtractVariableOp>(operation))
                headers.push_back(extractVariable.getHdr());
        });
        for (mlir::Value header : headers) wrapExtract(header);
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createWrapExtractNextPass() {
    return std::make_unique<WrapExtractNext>();
}

}  // namespace P4::P4MLIR
