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
    Value cond = switchOp.getCondition();
    auto extractOp = cond.getDefiningOp<P4HIR::StructExtractOp>();
    if (!extractOp)
        return false;
    if (extractOp.getFieldName() != "action_run")
        return false;

    auto tableApply =
        extractOp.getInput().getDefiningOp<P4HIR::TableApplyOp>();
    if (!tableApply)
        return false;
    return true;
}

LogicalResult SwitchEliminationPass::eliminateSwitch(P4HIR::ControlOp control,
                                                     P4HIR::SwitchOp switchOp) {
    if (isActionRunSwitch(switchOp))
        return success();

    auto loc = switchOp.getLoc();
    auto condType = switchOp.getCondition().getType();
    OpBuilder builder(switchOp);
    auto ctx = builder.getContext();
    SymbolTable symbolTable(control->getParentOp());
    unsigned counter = 0;
    std::string prefix = SymbolTable::generateSymbolName<256>(
        "_switch",
        [&](llvm::StringRef candidate) {
            return symbolTable.lookup(candidate) != nullptr;
        },
        counter).str().str();

    llvm::SmallVector<P4HIR::CaseOp> cases;
    P4HIR::CaseOp defaultCase;
    for (auto caseOp : switchOp.cases()) {
        if (caseOp.getKind() == P4HIR::CaseOpKind::Default)
            defaultCase = caseOp;
        else
            cases.push_back(caseOp);
    }

    llvm::SmallVector<mlir::StringAttr> actionNames;
    mlir::StringAttr defaultActionName;
    builder.setInsertionPoint(control.getBody().front().getTerminator());

    auto hiddenAttr = builder.getDictionaryAttr(
        {builder.getNamedAttr("hidden", builder.getUnitAttr())});

    auto createAction = [&](mlir::StringAttr actionName) {
        auto funcType = P4HIR::FuncType::get(ctx, {});
        auto funcOp = P4HIR::FuncOp::buildAction(builder, loc, actionName, funcType,
                                                 {}, hiddenAttr);
        OpBuilder funcBuilder(funcOp.getBody());
        funcBuilder.setInsertionPointToEnd(&funcOp.getBody().front());
        funcBuilder.create<P4HIR::ReturnOp>(loc);
    };

    for (auto [i, _] : llvm::enumerate(cases)) {
        auto actionName =
            builder.getStringAttr(llvm::formatv("{0}_case_{1}", prefix, i).str());
        actionNames.push_back(actionName);
        createAction(actionName);
    }

    defaultActionName = builder.getStringAttr(prefix + "_default");
    actionNames.push_back(defaultActionName);
    createAction(defaultActionName);

    auto actionEnumType = P4HIR::EnumType::get(
        ctx, prefix + "_action_enum",
        llvm::SmallVector<Attribute>(actionNames.begin(), actionNames.end()),
        DictionaryAttr());
    

    auto applyResultType = P4HIR::StructType::get(
        ctx, prefix + "_result",
        {P4HIR::FieldInfo{builder.getStringAttr("hit"), builder.getType<P4HIR::BoolType>()},
         P4HIR::FieldInfo{builder.getStringAttr("miss"), builder.getType<P4HIR::BoolType>()},
         P4HIR::FieldInfo{builder.getStringAttr("action_run"), actionEnumType}});

    std::string tableName = prefix + "_table";

    (void)builder.create<P4HIR::TableOp>(
        loc, tableName, hiddenAttr,
        [&](OpBuilder &tableBuilder, Location tableLoc) {
            auto keyFuncType = P4HIR::FuncType::get(ctx, {condType});
            auto tableKeyOp = tableBuilder.create<P4HIR::TableKeyOp>(
                tableLoc, keyFuncType, ArrayRef<DictionaryAttr>{}, DictionaryAttr());
            tableKeyOp.createEntryBlock();
            {
                OpBuilder::InsertionGuard guard(tableBuilder);
                tableBuilder.setInsertionPointToStart(&tableKeyOp.getBody().front());
                auto keyArg = tableKeyOp.getBody().front().getArgument(0);
                tableBuilder.create<P4HIR::TableKeyEntryOp>(
                    tableLoc, "exact", keyArg, mlir::DictionaryAttr::get(ctx, {}));
            }

            tableBuilder.create<P4HIR::TableActionsOp>(
                tableLoc, DictionaryAttr(),
                [&](OpBuilder &actionsBuilder, Location actionsLoc) {
                    for (auto actionName : actionNames) {
                        auto actionFuncType = P4HIR::FuncType::get(ctx, {});
                        auto actionRef = FlatSymbolRefAttr::get(actionName);
                        actionsBuilder.create<P4HIR::TableActionOp>(
                            actionsLoc, actionRef, actionFuncType,
                            ArrayRef<DictionaryAttr>{}, DictionaryAttr(),
                            [&](OpBuilder &actionBodyBuilder, Block::BlockArgListType,
                                Location actionBodyLoc) {
                                auto controlRef = SymbolRefAttr::get(
                                    ctx, control.getName(),
                                    {FlatSymbolRefAttr::get(ctx, actionName)});
                                actionBodyBuilder.create<P4HIR::CallOp>(
                                    actionBodyLoc, controlRef, ValueRange{});
                            });
                    }
                });

            tableBuilder.create<P4HIR::TableDefaultActionOp>(
                tableLoc, false, DictionaryAttr(),
                [&](OpBuilder &defaultBuilder, Location defaultLoc) {
                    auto controlRef = SymbolRefAttr::get(
                        ctx, control.getName(),
                        {FlatSymbolRefAttr::get(ctx, defaultActionName)});
                    defaultBuilder.create<P4HIR::CallOp>(defaultLoc, controlRef, ValueRange{});
                });
            tableBuilder.create<P4HIR::TableEntriesOp>(
                tableLoc, true, DictionaryAttr(),
                [&](OpBuilder &b, Location entriesLoc) {
                    for (auto [caseOp, actionName] : llvm::zip(cases, llvm::ArrayRef(actionNames).drop_back())) {
                        for (auto valueAttr : caseOp.getValue()) {
                            auto tupleType = TupleType::get(ctx, {condType});
                            auto keyAttr = P4HIR::AggAttr::get(
                                tupleType, b.getArrayAttr({valueAttr}));            
                            b.create<P4HIR::TableEntryOp>(
                                entriesLoc, keyAttr, false, TypedAttr(), DictionaryAttr(),
                                [&](OpBuilder &entryBuilder, Location entryLoc) {
                                    auto controlRef = SymbolRefAttr::get(
                                        ctx, control.getName(),
                                        {FlatSymbolRefAttr::get(ctx, actionName)});
                                    entryBuilder.create<P4HIR::CallOp>(
                                        entryLoc, controlRef, ValueRange{});
                                });
                        }
                    }
                });
        });

    builder.setInsertionPoint(switchOp);

    // Create a temporary variable for the switch key (following p4c's pattern)
    std::string keyName = prefix + "_key";
    auto keyVar = builder.create<P4HIR::VariableOp>(
        loc, P4HIR::ReferenceType::get(condType), keyName, hiddenAttr);

    // Assign the condition expression to the key variable
    builder.create<P4HIR::AssignOp>(loc, switchOp.getCondition(), keyVar);

    // Read from the key variable for the table apply
    auto keyValue = builder.create<P4HIR::ReadOp>(loc, keyVar);

    auto tableRef = SymbolRefAttr::get(ctx, control.getName(),
                                       {FlatSymbolRefAttr::get(ctx, tableName)});
    auto tableApply = builder.create<P4HIR::TableApplyOp>(
        loc, applyResultType, tableRef, ValueRange{keyValue.getResult()});

    auto actionRunField = builder.create<P4HIR::StructExtractOp>(
        loc, tableApply.getResult(), "action_run");

    (void)builder.create<P4HIR::SwitchOp>(
        loc, actionRunField.getResult(),
        [&](OpBuilder &switchBuilder, Location switchLoc) {
            for (auto [caseOp, actionName] : llvm::zip(cases, llvm::ArrayRef(actionNames).drop_back())) {
                auto enumMemberAttr = P4HIR::EnumFieldAttr::get(actionEnumType, actionName);
                switchBuilder.create<P4HIR::CaseOp>(
                    switchLoc, switchBuilder.getArrayAttr({enumMemberAttr}),
                    P4HIR::CaseOpKind::Equal,
                    [&](OpBuilder &caseBuilder, Location caseLoc) {
                        IRMapping mapper;
                        for (auto &op : caseOp.getCaseRegion().front().without_terminator())
                            caseBuilder.clone(op, mapper);
                        caseBuilder.create<P4HIR::YieldOp>(caseLoc);
                    });
            }

            if (defaultCase) {
                auto enumMemberAttr = P4HIR::EnumFieldAttr::get(actionEnumType, defaultActionName);
                switchBuilder.create<P4HIR::CaseOp>(
                    switchLoc, switchBuilder.getArrayAttr({enumMemberAttr}),
                    P4HIR::CaseOpKind::Default,
                    [&](OpBuilder &caseBuilder, Location caseLoc) {
                        IRMapping mapper;
                        for (auto &op : defaultCase.getCaseRegion().front().without_terminator())
                            caseBuilder.clone(op, mapper);
                        caseBuilder.create<P4HIR::YieldOp>(caseLoc);
                    });
            }

            switchBuilder.create<P4HIR::YieldOp>(switchLoc);
        });

    switchOp.erase();
    return success();
}

void SwitchEliminationPass::runOnOperation() {
    auto module = getOperation();

    module->walk([&](P4HIR::ControlOp control) {
        return control->walk([&](P4HIR::SwitchOp switchOp) {
            if (failed(eliminateSwitch(control, switchOp)))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
    });
}

}

std::unique_ptr<Pass> P4::P4MLIR::createSwitchEliminationPass() {
    return std::make_unique<SwitchEliminationPass>();
}