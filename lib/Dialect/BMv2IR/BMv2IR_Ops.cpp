#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
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

    if (!isa<BMv2IR::HeaderInstanceOp>(decl))
        return emitOpError("invalid symbol reference: ") << decl << ", expected header instance";

    return success();
}

LogicalResult FieldOp::verify() {
    auto op = SymbolTable::lookupSymbolIn(getParentModule(*this), getHeaderInstance());
    if (!op) return emitOpError("Can't find op for symbol reference");

    auto instance = dyn_cast<BMv2IR::HeaderInstanceOp>(op);
    if (!instance) return emitOpError("Field op should refer to an header instance");

    auto fieldName = getFieldMember();
    return llvm::TypeSwitch<Type, LogicalResult>(instance.getHeaderType())
        .Case([&](BMv2IR::HeaderType ty) -> LogicalResult {
            if (!ty.hasField(fieldName)) return emitOpError("Invalid field name");
            return success();
        })
        .Case([&](P4HIR::HeaderType ty) -> LogicalResult {
            if (!ty.getFieldType(fieldName)) return emitOpError("Invalid field name");
            return success();
        });
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
    auto nextTablesPair = dyn_cast<ArrayAttr>(getNextTables());
    if (!nextTablesPair) return success();
    auto nextTables = llvm::map_to_vector(
        nextTablesPair, [](Attribute a) { return cast<BMv2IR::ActionTableAttr>(a); });
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
            if (!isa<BMv2IR::TableOp, BMv2IR::ConditionalOp>(tableDecl))
                return emitOpError("expected table or conditional op, got ") << tableDecl << "\n";
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
        if (!isa<AllowedTys...>(defOp)) return emitOpError("unexpected type") << defOp;
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
    if (failed(checkArg.operator()<P4HIR::ControlOp, BMv2IR::DeparserOp>(getDeparser())))
        return failure();
    return success();
}

static FailureOr<V1SwitchOp> getPackageInstantiationFromParentModule(Operation *op) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp) return op->emitError("No module parent");
    // TODO: consider adding an interface to support different targets transparently
    V1SwitchOp packageInstantiateOp = nullptr;
    auto walkRes = moduleOp.walk([&](V1SwitchOp v1switch) {
        if (packageInstantiateOp != nullptr) return WalkResult::interrupt();
        packageInstantiateOp = v1switch;
        return WalkResult::advance();
    });
    if (walkRes.wasInterrupted())
        return op->emitError("Expected only a single package instantiation");
    if (!packageInstantiateOp)
        return op->emitError("Expected package instantiation op in the module");
    return packageInstantiateOp;
}

HeaderInstanceOp HeaderUnionInstanceOp::getInstanceByName(StringRef name) {
    auto type = dyn_cast<P4HIR::HeaderUnionType>(getUnionType());
    if (!type) return nullptr;
    auto moduleOp = getParentModule(*this);
    assert(moduleOp && "No parent module");
    auto index = type.getFieldIndex(name);
    assert(index.has_value() && "No index for field name");
    auto ref = cast<SymbolRefAttr>(getHeaders()[index.value()]);
    return cast<HeaderInstanceOp>(SymbolTable::lookupSymbolIn(moduleOp, ref));
}

namespace P4::P4MLIR::BMv2IR {

StringAttr getUniqueNameInParentModule(Operation *op, Twine base) {
    auto name = StringAttr::get(op->getContext(), base);
    unsigned counter = 0;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Expected module op");
    auto uniqueName = SymbolTable::generateSymbolName<256>(
        name,
        [&](StringRef candidate) {
            return SymbolTable::lookupSymbolIn(moduleOp, candidate) != nullptr;
        },
        counter);
    return StringAttr::get(op->getContext(), uniqueName);
}

using P4HIR_ControlOp = P4::P4MLIR::P4HIR::ControlOp;

FailureOr<bool> isTopLevelControl(P4HIR_ControlOp controlOp) {
    auto packageInstantiateOp = getPackageInstantiationFromParentModule(controlOp);
    if (failed(packageInstantiateOp)) return failure();
    auto symToCheck = controlOp.getSymName();
    return symToCheck == packageInstantiateOp->getIngress().getLeafReference() ||
           symToCheck == packageInstantiateOp->getEgress().getLeafReference();
}

FailureOr<bool> isDeparserControl(P4HIR_ControlOp controlOp) {
    auto packageInstantiateOp = getPackageInstantiationFromParentModule(controlOp);
    if (failed(packageInstantiateOp)) return failure();
    auto symToCheck = controlOp.getSymName();
    return symToCheck == packageInstantiateOp->getDeparser().getLeafReference();
}

FailureOr<bool> isCalculationControl(P4HIR_ControlOp controlOp) {
    auto packageInstantiateOp = getPackageInstantiationFromParentModule(controlOp);
    if (failed(packageInstantiateOp)) return failure();
    auto symToCheck = controlOp.getSymName();
    return symToCheck == packageInstantiateOp->getVerifyChecksum().getLeafReference() ||
           symToCheck == packageInstantiateOp->getComputeChecksum().getLeafReference();
}

bool isHitOrMissIf(Operation *op) {
    auto ifOp = dyn_cast<P4HIR::IfOp>(op);
    if (!ifOp) return false;
    auto extractOp = ifOp.getCondition().getDefiningOp<P4HIR::StructExtractOp>();
    if (!extractOp) return false;
    auto applyOp = extractOp.getInput().getDefiningOp<P4HIR::TableApplyOp>();
    if (!applyOp) return false;

    auto fieldName = extractOp.getFieldName();
    return fieldName == "hit" || fieldName == "miss";
}

FailureOr<P4HIR::IntAttr> getTrueMask(MLIRContext *ctx, unsigned width) {
    // TODO: how do we represent values that don't fit in int64_t ?
    if (llvm::divideCeil(width, 8) > sizeof(int64_t)) return failure();
    int64_t one = 1;
    auto res = (one << width) - 1;
    return getWithWidth(ctx, res, width);
}

FailureOr<P4HIR::IntAttr> getWithWidth(MLIRContext *ctx, int64_t val, unsigned width) {
    auto type = P4HIR::BitsType::get(ctx, width, false);
    return P4HIR::IntAttr::get(type, val);
}

}  // namespace P4::P4MLIR::BMv2IR

llvm::FailureOr<unsigned> P4::P4MLIR::BMv2IR::TableKeyAttr::getWidth(ModuleOp moduleOp) {
    auto headerDef = SymbolTable::lookupSymbolIn(moduleOp, getHeader());
    auto header = dyn_cast<HeaderInstanceOp>(headerDef);
    auto headerTy = dyn_cast<HeaderType>(header.getHeaderType());
    if (!headerTy) return failure();
    auto fieldInfo = headerTy.getField(getFieldName().getValue());
    if (failed(fieldInfo)) return failure();
    auto fieldTy = dyn_cast<P4HIR::BitsType>(fieldInfo->type);
    if (!fieldTy) return failure();
    return fieldTy.getWidth();
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
