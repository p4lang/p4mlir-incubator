#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-switch-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SWITCHELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}

using namespace P4::P4MLIR;

namespace {

struct SwitchEliminationPass
    : public P4::P4MLIR::impl::SwitchEliminationBase<SwitchEliminationPass> {
    SwitchEliminationPass() = default;
    void runOnOperation() override;

private:
    static LogicalResult eliminateSwitch(P4HIR::ControlOp control, P4HIR::SwitchOp switchOp);
    static bool isActionRunSwitch(P4HIR::SwitchOp switchOp);
};

bool SwitchEliminationPass::isActionRunSwitch(P4HIR::SwitchOp switchOp) {
    auto extractOp = switchOp.getCondition().getDefiningOp<P4HIR::StructExtractOp>();
    if (!extractOp || extractOp.getFieldName() != "action_run")
        return false;
    return extractOp.getInput().getDefiningOp<P4HIR::TableApplyOp>() != nullptr;
}

LogicalResult SwitchEliminationPass::eliminateSwitch(P4HIR::ControlOp control,
                                                     P4HIR::SwitchOp switchOp) {
    if (isActionRunSwitch(switchOp))
        return success();

    auto loc = switchOp.getLoc();
    auto condType = switchOp.getCondition().getType();
    IRRewriter rewriter(switchOp->getContext());
    rewriter.setInsertionPoint(switchOp);
    auto ctx = rewriter.getContext();
    SymbolTable symbolTable(control->getParentOp());

    // Generate unique prefix for symbols
    unsigned counter = 0;
    auto prefix = SymbolTable::generateSymbolName<256>(
        "_switch",
        [&](llvm::StringRef candidate) {
            return symbolTable.lookup(candidate) != nullptr;
        },
        counter);

    auto hiddenAttr = rewriter.getDictionaryAttr(
        {rewriter.getNamedAttr("hidden", rewriter.getUnitAttr())});

    // Count non-default cases and find default case
    size_t numCases = 0;
    P4HIR::CaseOp defaultCase;
    for (auto caseOp : switchOp.cases()) {
        if (caseOp.getKind() == P4HIR::CaseOpKind::Default)
            defaultCase = caseOp;
        else
            ++numCases;
    }

    // Build action names for enum type
    llvm::SmallVector<Attribute> actionNameAttrs;
    for (size_t i = 0; i < numCases; ++i)
        actionNameAttrs.push_back(
            rewriter.getStringAttr(llvm::formatv("{0}_case_{1}", prefix, i)));
    auto defaultActionName = rewriter.getStringAttr(
        llvm::Twine(prefix, "_default").str());
    actionNameAttrs.push_back(defaultActionName);

    auto actionEnumType = P4HIR::EnumType::get(
        ctx, llvm::Twine(prefix, "_action_enum").str(),
        actionNameAttrs, DictionaryAttr());

    auto applyResultType = P4HIR::StructType::get(
        ctx, llvm::Twine(prefix, "_result").str(),
        {P4HIR::FieldInfo{rewriter.getStringAttr("hit"), rewriter.getType<P4HIR::BoolType>()},
         P4HIR::FieldInfo{rewriter.getStringAttr("miss"), rewriter.getType<P4HIR::BoolType>()},
         P4HIR::FieldInfo{rewriter.getStringAttr("action_run"), actionEnumType}});

    // Create action functions
    rewriter.setInsertionPoint(control.getBody().front().getTerminator());
    auto funcType = P4HIR::FuncType::get(ctx, {});

    for (auto actionNameAttr : actionNameAttrs) {
        auto actionName = mlir::cast<StringAttr>(actionNameAttr);
        auto funcOp = P4HIR::FuncOp::buildAction(rewriter, loc, actionName.getValue(), funcType,
                                                 {}, hiddenAttr);
        IRRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&funcOp.getBody().front());
        rewriter.create<P4HIR::ReturnOp>(loc);
    }

    // Build the table
    auto tableName = llvm::Twine(prefix, "_table").str();
    (void)rewriter.create<P4HIR::TableOp>(
        loc, tableName, hiddenAttr,
        [&](OpBuilder &b, Location tableLoc) {
            // Table key
            auto keyFuncType = P4HIR::FuncType::get(ctx, {condType});
            auto tableKeyOp = b.create<P4HIR::TableKeyOp>(
                tableLoc, keyFuncType, ArrayRef<DictionaryAttr>{}, DictionaryAttr());
            tableKeyOp.createEntryBlock();
            {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(&tableKeyOp.getBody().front());
                b.create<P4HIR::TableKeyEntryOp>(
                    tableLoc, "exact", tableKeyOp.getBody().front().getArgument(0),
                    mlir::DictionaryAttr::get(ctx, {}));
            }

            // Table actions
            b.create<P4HIR::TableActionsOp>(
                tableLoc, DictionaryAttr(),
                [&](OpBuilder &b, Location actionsLoc) {
                    for (auto actionNameAttr : actionNameAttrs) {
                        auto actionName = mlir::cast<StringAttr>(actionNameAttr);
                        auto actionFuncType = P4HIR::FuncType::get(ctx, {});
                        auto actionRef = FlatSymbolRefAttr::get(actionName);
                        b.create<P4HIR::TableActionOp>(
                            actionsLoc, actionRef, actionFuncType,
                            ArrayRef<DictionaryAttr>{}, DictionaryAttr(),
                            [&](OpBuilder &b, Block::BlockArgListType,
                                Location actionBodyLoc) {
                                auto controlRef = SymbolRefAttr::get(
                                    ctx, control.getName(),
                                    {FlatSymbolRefAttr::get(ctx, actionName)});
                                b.create<P4HIR::CallOp>(
                                    actionBodyLoc, controlRef, ValueRange{});
                            });
                    }
                });

            // Table default action
            b.create<P4HIR::TableDefaultActionOp>(
                tableLoc, false, DictionaryAttr(),
                [&](OpBuilder &b, Location defaultLoc) {
                    auto controlRef = SymbolRefAttr::get(
                        ctx, control.getName(),
                        {FlatSymbolRefAttr::get(ctx, defaultActionName)});
                    b.create<P4HIR::CallOp>(defaultLoc, controlRef, ValueRange{});
                });

            // Table entries
            b.create<P4HIR::TableEntriesOp>(
                tableLoc, true, DictionaryAttr(),
                [&](OpBuilder &b, Location entriesLoc) {
                    size_t caseIndex = 0;
                    for (auto caseOp : switchOp.cases()) {
                        if (caseOp.getKind() == P4HIR::CaseOpKind::Default)
                            continue;
                        auto actionName = mlir::cast<StringAttr>(actionNameAttrs[caseIndex]);
                        for (auto valueAttr : caseOp.getValue()) {
                            auto tupleType = TupleType::get(ctx, {condType});
                            auto keyAttr = P4HIR::AggAttr::get(
                                tupleType, b.getArrayAttr({valueAttr}));
                            b.create<P4HIR::TableEntryOp>(
                                entriesLoc, keyAttr, false, TypedAttr(), DictionaryAttr(),
                                [&](OpBuilder &b, Location entryLoc) {
                                    auto controlRef = SymbolRefAttr::get(
                                        ctx, control.getName(),
                                        {FlatSymbolRefAttr::get(ctx, actionName)});
                                    b.create<P4HIR::CallOp>(
                                        entryLoc, controlRef, ValueRange{});
                                });
                        }
                        ++caseIndex;
                    }
                });
        });

    // Apply table directly with switch condition
    rewriter.setInsertionPoint(switchOp);
    auto tableRef = SymbolRefAttr::get(ctx, control.getName(),
                                       {FlatSymbolRefAttr::get(ctx, tableName)});
    auto tableApply = rewriter.create<P4HIR::TableApplyOp>(
        loc, applyResultType, tableRef, ValueRange{switchOp.getCondition()});

    auto actionRunField = rewriter.create<P4HIR::StructExtractOp>(
        loc, tableApply.getResult(), "action_run");

    // Build the new switch on action_run
    (void)rewriter.create<P4HIR::SwitchOp>(
        loc, actionRunField.getResult(),
        [&](OpBuilder &b, Location switchLoc) {
            auto cloneCase = [&](P4HIR::CaseOp origCase, StringAttr actionName,
                                 P4HIR::CaseOpKind kind) {
                auto enumMemberAttr = P4HIR::EnumFieldAttr::get(
                    actionEnumType, actionName.getValue());
                b.create<P4HIR::CaseOp>(
                    switchLoc, b.getArrayAttr({enumMemberAttr}), kind,
                    [&](OpBuilder &b, Location caseLoc) {
                        IRMapping mapper;
                        for (auto &op : origCase.getCaseRegion().front())
                            b.clone(op, mapper);
                    });
            };

            size_t caseIndex = 0;
            for (auto caseOp : switchOp.cases()) {
                bool isDefault = caseOp.getKind() == P4HIR::CaseOpKind::Default;
                auto actionName = isDefault
                    ? defaultActionName
                    : mlir::cast<StringAttr>(actionNameAttrs[caseIndex]);
                cloneCase(caseOp, actionName,
                          isDefault ? P4HIR::CaseOpKind::Default
                                    : P4HIR::CaseOpKind::Equal);
                if (!isDefault) ++caseIndex;
            }

            b.create<P4HIR::YieldOp>(switchLoc);
        });

    rewriter.eraseOp(switchOp);
    return success();
}

void SwitchEliminationPass::runOnOperation() {
    auto module = getOperation();

    auto result = module->walk([&](P4HIR::ControlOp control) {
        return control->walk([&](P4HIR::SwitchOp switchOp) {
            if (failed(eliminateSwitch(control, switchOp)))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
    });

    if (result.wasInterrupted())
        signalPassFailure();
}

}

std::unique_ptr<Pass> P4::P4MLIR::createSwitchEliminationPass() {
    return std::make_unique<SwitchEliminationPass>();
}
