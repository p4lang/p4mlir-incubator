#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
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

class NameGenerator {
public:
    std::string newName(llvm::StringRef prefix) {
        return llvm::formatv("{0}_{1}", prefix, counter++).str();
    }

private:
    unsigned counter = 0;
};

struct SwitchEliminationPass
    : public P4::P4MLIR::impl::SwitchEliminationBase<SwitchEliminationPass> {
    SwitchEliminationPass() = default;
    void runOnOperation() override;

private:
    LogicalResult eliminateSwitch(P4HIR::ControlOp control, P4HIR::SwitchOp switchOp,
                                  NameGenerator &nameGen);
    bool isActionRunSwitch(P4HIR::SwitchOp switchOp);
};

bool SwitchEliminationPass::isActionRunSwitch(P4HIR::SwitchOp switchOp) {
    auto condType = switchOp.getCondition().getType();
    if (auto structType = mlir::dyn_cast<P4HIR::StructType>(condType)) {
        for (auto field : structType.getFields()) {
            if (field.name == "action_run")
                return true;
        }
    }
    return mlir::isa<P4HIR::EnumType>(condType);
}

LogicalResult SwitchEliminationPass::eliminateSwitch(P4HIR::ControlOp control,
                                                     P4HIR::SwitchOp switchOp,
                                                     NameGenerator &nameGen) {
    if (isActionRunSwitch(switchOp))
        return success();

    auto loc = switchOp.getLoc();
    auto condType = switchOp.getCondition().getType();
    OpBuilder builder(switchOp);

    std::string prefix = nameGen.newName("_switch");

    llvm::SmallVector<P4HIR::CaseOp> cases;
    P4HIR::CaseOp defaultCase;
    for (auto caseOp : switchOp.cases()) {
        if (caseOp.getKind() == P4HIR::CaseOpKind::Default)
            defaultCase = caseOp;
        else
            cases.push_back(caseOp);
    }

    llvm::SmallVector<std::string> actionNames;
    llvm::SmallVector<mlir::Attribute> enumFields;
    auto ctx = builder.getContext();

    builder.setInsertionPoint(control.getBody().front().getTerminator());

    auto hiddenAttr = builder.getDictionaryAttr(
        {builder.getNamedAttr("hidden", builder.getUnitAttr())});

    for (size_t i = 0; i < cases.size(); ++i) {
        std::string actionName = llvm::formatv("{0}_case_{1}", prefix, i).str();
        actionNames.push_back(actionName);
        enumFields.push_back(builder.getStringAttr(actionName));

        auto funcType = P4HIR::FuncType::get(ctx, {});
        auto funcOp = P4HIR::FuncOp::buildAction(builder, loc, actionName, funcType,
                                                  {}, hiddenAttr);
        OpBuilder funcBuilder(funcOp.getBody());
        funcBuilder.setInsertionPointToEnd(&funcOp.getBody().front());
        funcBuilder.create<P4HIR::ReturnOp>(loc);
    }

    std::string defaultActionName = prefix + "_default";
    actionNames.push_back(defaultActionName);
    enumFields.push_back(builder.getStringAttr(defaultActionName));

    auto defaultFuncType = P4HIR::FuncType::get(ctx, {});
    auto defaultFuncOp = P4HIR::FuncOp::buildAction(builder, loc, defaultActionName,
                                                     defaultFuncType, {}, hiddenAttr);
    OpBuilder defaultFuncBuilder(defaultFuncOp.getBody());
    defaultFuncBuilder.setInsertionPointToEnd(&defaultFuncOp.getBody().front());
    defaultFuncBuilder.create<P4HIR::ReturnOp>(loc);

    auto actionEnumType = P4HIR::EnumType::get(
        ctx, prefix + "_action_enum", enumFields, DictionaryAttr());

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
                auto keyArg = tableKeyOp.getBody().front().addArgument(condType, tableLoc);
                tableBuilder.create<P4HIR::TableKeyEntryOp>(
                    tableLoc, "exact", keyArg, mlir::DictionaryAttr::get(ctx, {}));
            }

            tableBuilder.create<P4HIR::TableActionsOp>(
                tableLoc, DictionaryAttr(),
                [&](OpBuilder &actionsBuilder, Location actionsLoc) {
                    for (size_t i = 0; i < actionNames.size(); ++i) {
                        auto actionFuncType = P4HIR::FuncType::get(ctx, {});
                        auto actionRef = FlatSymbolRefAttr::get(ctx, actionNames[i]);
                        actionsBuilder.create<P4HIR::TableActionOp>(
                            actionsLoc, actionRef, actionFuncType,
                            ArrayRef<DictionaryAttr>{}, DictionaryAttr(),
                            [&](OpBuilder &actionBodyBuilder, Block::BlockArgListType,
                                Location actionBodyLoc) {
                                auto controlRef = SymbolRefAttr::get(
                                    ctx, control.getName(),
                                    {FlatSymbolRefAttr::get(ctx, actionNames[i])});
                                actionBodyBuilder.create<P4HIR::CallOp>(
                                    actionBodyLoc, controlRef, ValueRange{});
                            });
                    }
                });

            tableBuilder.create<P4HIR::TableDefaultActionOp>(
                tableLoc, DictionaryAttr(),
                [&](OpBuilder &defaultBuilder, Location defaultLoc) {
                    auto controlRef = SymbolRefAttr::get(
                        ctx, control.getName(),
                        {FlatSymbolRefAttr::get(ctx, defaultActionName)});
                    defaultBuilder.create<P4HIR::CallOp>(defaultLoc, controlRef, ValueRange{});
                });

            tableBuilder.create<P4HIR::TableEntriesOp>(
                tableLoc, true, DictionaryAttr(),
                [&](OpBuilder &entriesBuilder, Location entriesLoc) {
                    for (size_t i = 0; i < cases.size(); ++i) {
                        auto caseOp = cases[i];
                        for (auto valueAttr : caseOp.getValue()) {
                            auto tupleType = TupleType::get(ctx, {condType});
                            auto keyAttr = P4HIR::AggAttr::get(
                                tupleType, builder.getArrayAttr({valueAttr}));
                            entriesBuilder.create<P4HIR::TableEntryOp>(
                                entriesLoc, keyAttr, false, TypedAttr(), DictionaryAttr(),
                                [&](OpBuilder &entryBuilder, Location entryLoc) {
                                    auto controlRef = SymbolRefAttr::get(
                                        ctx, control.getName(),
                                        {FlatSymbolRefAttr::get(ctx, actionNames[i])});
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
        loc, P4HIR::ReferenceType::get(ctx, condType), keyName, hiddenAttr);

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
            for (size_t i = 0; i < cases.size(); ++i) {
                auto caseOp = cases[i];
                auto enumMemberAttr = P4HIR::EnumFieldAttr::get(
                    actionEnumType, builder.getStringAttr(actionNames[i]));

                switchBuilder.create<P4HIR::CaseOp>(
                    switchLoc, builder.getArrayAttr({enumMemberAttr}),
                    P4HIR::CaseOpKind::Equal,
                    [&](OpBuilder &caseBuilder, Location caseLoc) {
                        IRMapping mapper;
                        for (auto &op : caseOp.getCaseRegion().front()) {
                            if (!mlir::isa<P4HIR::YieldOp>(&op))
                                caseBuilder.clone(op, mapper);
                        }
                        caseBuilder.create<P4HIR::YieldOp>(caseLoc);
                    });
            }

            if (defaultCase) {
                auto enumMemberAttr = P4HIR::EnumFieldAttr::get(
                    actionEnumType, builder.getStringAttr(defaultActionName));

                switchBuilder.create<P4HIR::CaseOp>(
                    switchLoc, builder.getArrayAttr({enumMemberAttr}),
                    P4HIR::CaseOpKind::Default,
                    [&](OpBuilder &caseBuilder, Location caseLoc) {
                        IRMapping mapper;
                        for (auto &op : defaultCase.getCaseRegion().front()) {
                            if (!mlir::isa<P4HIR::YieldOp>(&op))
                                caseBuilder.clone(op, mapper);
                        }
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
    NameGenerator nameGen;

    llvm::SmallVector<std::pair<P4HIR::ControlOp, P4HIR::SwitchOp>> toProcess;

    module->walk([&](P4HIR::ControlOp control) {
        control->walk([&](P4HIR::SwitchOp switchOp) {
            if (!isActionRunSwitch(switchOp))
                toProcess.emplace_back(control, switchOp);
        });
    });

    for (auto [control, switchOp] : toProcess) {
        if (failed(eliminateSwitch(control, switchOp, nameGen)))
            return signalPassFailure();
    }
}

}

std::unique_ptr<Pass> P4::P4MLIR::createSwitchEliminationPass() {
    return std::make_unique<SwitchEliminationPass>();
}
