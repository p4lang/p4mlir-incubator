#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir//Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir//Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_OpInterfaces.h"

using namespace mlir;
using namespace P4::P4MLIR::BMv2IR;

LogicalResult ParserStateOp::verify() {
    // Check that only allowed ops are used as transition keys, transitions and parser ops

    for (auto &block : getTransitionKeys()) {
        for (auto &op : block) {
            if (!isa<AllowedTransitionKey>(op)) {
                return emitError("Op not allowed as transition key");
            }
        }
    }

    for (auto &block : getTransitions()) {
        for (auto &op : block) {
            if (!isa<TransitionOp>(op)) return emitError("Op not allowed as transition");
        }
    }

    for (auto &block : getParserOps()) {
        for (auto &op : block) {
            if (!isa<AllowedParserOp>(op)) return emitError("Op not allowed as parser op");
        }
    }
    return success();
}

static mlir::ModuleOp getParentModule(Operation *from) {
    if (auto moduleOp = from->getParentOfType<mlir::ModuleOp>()) return moduleOp;

    from->emitOpError("could not find parent module op");
    return nullptr;
}

LogicalResult SymToValueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the decl attribute was specified.
    auto declAttr = (*this)->getAttrOfType<SymbolRefAttr>(getDeclAttrName());
    if (!declAttr) return emitOpError("requires a 'decl' symbol reference attribute");

    auto decl = symbolTable.lookupSymbolIn(getParentModule(*this), declAttr);
    if (!decl) return emitOpError("cannot resolve symbol '") << declAttr << "' to declaration";

    if (!mlir::isa<BMv2IR::HeaderInstanceOp>(decl))
        return emitOpError("invalid symbol reference: ") << decl << ", expected header instance";

    return mlir::success();
}

LogicalResult ConditionalOp::verify() {
    // Check that the condition region yields a boolean value
    auto yieldOp = cast<BMv2IR::YieldOp>(getConditionRegion().front().getTerminator());
    if (yieldOp->getNumOperands() != 1 || !isa<P4HIR::BoolType>(yieldOp->getOperand(0).getType()))
        return emitOpError("Condition region should yield a boolean");
    // Check that the then and else symbols refer to either tables, conditionals or
    // (TODO) action calls
    auto moduleOp = getParentModule(*this);
    auto checkSym = [&](SymbolRefAttr ref) -> LogicalResult {
        auto thenDecl = SymbolTable::lookupSymbolIn(moduleOp, ref);
        if (!thenDecl) return emitOpError("cannot resolve symbol ") << thenDecl << "\n";
        if (!isa<BMv2IR::TableOp, BMv2IR::ConditionalOp>(thenDecl))
            return emitOpError("symbol resolves to invalid op");
        return success();
    };

    if (failed(checkSym(getThenRef()))) return failure();

    auto elseRef = getElseRefAttr();
    if (!elseRef) return success();
    return checkSym(elseRef);
}

LogicalResult TableOp::verify() {
    auto actions = getActions();
    auto nextTables = llvm::map_to_vector(
        getNextTablesAttr(), [](Attribute a) { return cast<BMv2IR::ActionTableAttr>(a); });
    auto moduleOp = getParentModule(*this);
    for (auto a : actions) {
        auto actionRef = cast<SymbolRefAttr>(a);
        auto op = SymbolTable::lookupSymbolIn(moduleOp, actionRef);
        if (!op) return emitOpError("cannot resolve symbol ") << op << "\n";
        if (auto actionOp = dyn_cast<P4HIR::FuncOp>(op); !actionOp || !actionOp.getAction())
            return emitOpError("symbols resolves to invalid op: ") << op << "\n";

        auto it = llvm::find_if(nextTables, [&](ActionTableAttr at) {
            return at.getAction().getLeafReference() == actionRef.getLeafReference();
        });
        if (it == nextTables.end())
            return emitOpError("can't find next_table entry for action: ") << actionRef << "\n";
        auto tableSymRef = it->getTable();
        if (tableSymRef) {
            auto tableDecl = SymbolTable::lookupSymbolIn(moduleOp, it->getTable());
            if (!tableDecl) return emitError("cannot resolve symbol ") << tableDecl << "\n";
            if (!isa<BMv2IR::TableOp>(tableDecl))
                return emitOpError("expected table op, got ") << tableDecl << "\n";
        }
    }
    // Check that we only have at most one LPM key
    auto numLPM = llvm::count_if(getKeys(), [](Attribute a) {
        return cast<TableKeyAttr>(a).getMatchType() == BMv2IR::TableMatchKind::LPM;
    });
    if (numLPM > 1) return emitOpError("Only one LPM table key allowed");
    return success();
}

void SymToValueOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getDecl().getLeafReference());
}

LogicalResult V1SwitchOp::verify() {
    auto moduleOp = getParentModule(*this);
    auto checkArg = [&]<typename... AllowedTys>(SymbolRefAttr ref) -> LogicalResult {
        auto defOp = SymbolTable::lookupSymbolIn(moduleOp, ref);
        if (!defOp) return emitOpError("cannot resolve symbol");
        if (!isa<AllowedTys...>(defOp)) return emitOpError("unexpected type");
        return success();
    };
    if (failed(checkArg.operator()<P4HIR::ParserOp, BMv2IR::ParserOp>(getParser())))
        return failure();
    if (failed(checkArg.operator()<P4HIR::ControlOp>(getVerifyChecksum()))) return failure();
    if (failed(checkArg.operator()<P4HIR::ControlOp, BMv2IR::PipelineOp>(getIngress())))
        return failure();
    if (failed(checkArg.operator()<P4HIR::ControlOp, BMv2IR::PipelineOp>(getEgress())))
        return failure();
    if (failed(checkArg.operator()<P4HIR::ControlOp>(getComputeChecksum()))) return failure();
    if (failed(checkArg.operator()<P4HIR::ControlOp>(getDeparser()))) return failure();
    return success();
}

void BMv2IRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.cpp.inc"  // NOLINT
        >();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.cpp.inc"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.cpp.inc"  // NOLINT
