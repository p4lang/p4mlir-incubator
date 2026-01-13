#include "p4mlir/Targets/BMv2/Target.h"

#include <string>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "p4mlir/Common/Registration.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

using namespace llvm;
using namespace mlir;
using namespace P4::P4MLIR;

namespace {
constexpr StringRef id_attrname = "bmv2_id";

template <typename OpTy>
void setID(Operation *rootOp, std::function<bool(OpTy)> conditionF = [](OpTy) { return true; }) {
    int64_t id = 0;
    rootOp->walk([&](OpTy op) {
        if (conditionF(op)) {
            op->setAttr(id_attrname, IntegerAttr::get(op.getContext(), APSInt::get(id)));
            id++;
        }
    });
}

int64_t getId(Operation *op) {
    auto attr = dyn_cast_or_null<IntegerAttr>(op->getAttr(id_attrname));
    assert(attr && "getId called on op without id");
    return attr.getSInt();
}

void setUniqueIDS(ModuleOp moduleOp) {
    setID<BMv2IR::HeaderInstanceOp>(moduleOp);
    setID<BMv2IR::ParserOp>(moduleOp);
    setID<BMv2IR::ParserStateOp>(moduleOp);
    setID<P4HIR::FuncOp>(moduleOp, [](P4HIR::FuncOp funcOp) { return funcOp.getAction(); });
    setID<BMv2IR::ConditionalOp>(moduleOp);
    setID<BMv2IR::TableOp>(moduleOp);
    setID<BMv2IR::PipelineOp>(moduleOp);
    setID<BMv2IR::DeparserOp>(moduleOp);
    setID<BMv2IR::CalculationOp>(moduleOp);
    setID<BMv2IR::ChecksumOp>(moduleOp);
}

std::string asHexstr(P4HIR::IntAttr intAttr) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    auto bitTy = cast<P4HIR::BitsType>(intAttr.getType());
    auto bytes = llvm::divideCeil(bitTy.getWidth(), 8);
    auto width = bytes * 2;
    ss << llvm::format_hex_no_prefix(intAttr.getUInt(), width);
    return "0x" + ss.str();
}

json::Value to_JSON(Value val);
json::Value to_JSON(Operation *op);

json::Value to_JSON(BMv2IR::HeaderType headerTy, int64_t id) {
    json::Object res;

    json::Array fields;
    bool hasVarLenField = false;
    for (auto &field : headerTy.getFields()) {
        json::Array fieldDesc =
            llvm::TypeSwitch<Type, json::Array>(field.type)
                .Case([&](P4HIR::BitsType bitTy) -> json::Array {
                    return json::Array{field.name.str(), bitTy.getWidth(), bitTy.isSigned()};
                })
                .Case([&](P4HIR::VarBitsType varBitTy) -> json::Array {
                    hasVarLenField = true;
                    return json::Array{field.name.str(), "*", varBitTy.isSignedInteger()};
                });

        fields.push_back(std::move(fieldDesc));
    }

    res["fields"] = std::move(fields);
    res["name"] = headerTy.getName();
    res["id"] = id;
    if (hasVarLenField) res["max_length"] = headerTy.getMaxLength();

    return res;
}

json::Value to_JSON(BMv2IR::HeaderInstanceOp headerInstance) {
    json::Object res;
    auto name = cast<BMv2IR::HeaderType>(headerInstance.getHeaderType()).getName();
    res["name"] = headerInstance.getSymName().str();
    res["id"] = getId(headerInstance);
    res["header_type"] = name.str();
    res["metadata"] = headerInstance.getMetadata();

    return res;
}

json::Value to_JSON(BMv2IR::TransitionOp transitionOp) {
    json::Object res;
    auto type = transitionOp.getType();
    auto typeStr = BMv2IR::stringifyTransitionKind(type);
    res["type"] = typeStr;
    auto maybeNextState = transitionOp.getNextState();
    if (maybeNextState.has_value()) {
        auto nextStateName = maybeNextState.value().getLeafReference().getValue();
        res["next_state"] = nextStateName;
    } else {
        res["next_state"] = json::Value(nullptr);
    }

    switch (type) {
        case BMv2IR::TransitionKind::Hexstr: {
            auto valueAttr = cast<P4HIR::IntAttr>(transitionOp.getValueAttr());
            // FIXME: we need to ensure that hexstrings are correctly padded according to the bmv2 spec
            res["value"] = asHexstr(valueAttr);
            auto mask = transitionOp.getMask();
            if (mask.has_value()) {
                auto maskAttr = cast<P4HIR::IntAttr>(mask.value());
                res["mask"] = asHexstr(maskAttr);
            } else {
                res["mask"] = json::Value(nullptr);
            }
            break;
        }
        case BMv2IR::TransitionKind::Default: {
            res["value"] = json::Value(nullptr);
            res["mask"] = json::Value(nullptr);
            break;
        }
        case BMv2IR::TransitionKind::Parse_vset: {
            llvm_unreachable("JSON translation for parse_vsets NYI");
        }
    }
    return res;
}

json::Value to_JSON(BMv2IR::LookaheadOp lookAheadOp) {
    json::Object res;
    res["type"] = "lookahead";
    json::Array val{lookAheadOp.getBitOffset(), lookAheadOp.getBitwidth()};
    res["value"] = std::move(val);
    return res;
}

json::Value to_JSON(BMv2IR::FieldOp fieldOp) {
    json::Object res;
    res["type"] = "field";
    res["value"] = json::Array{fieldOp.getHeaderInstance().getLeafReference().getValue(),
                               fieldOp.getFieldMember().str()};

    return res;
}

json::Value to_JSON(BMv2IR::AssignHeaderOp assignOp) {
    // TODO: wrap this in a primitive node
    json::Object res;
    res["op"] = "assign_header";
    json::Object srcNode;
    srcNode["type"] = "header";
    srcNode["value"] = assignOp.getSrc().getLeafReference().getValue();
    json::Object dstNode;
    dstNode["type"] = "header";
    dstNode["value"] = assignOp.getDst().getLeafReference().getValue();

    json::Array parameters;
    parameters.push_back(std::move(dstNode));
    parameters.push_back(std::move(srcNode));
    res["parameters"] = std::move(parameters);
    return res;
}

json::Value to_JSON(BMv2IR::AssignOp assignOp) {
    json::Object res;
    res["op"] = "assign";
    json::Array params;
    params.push_back(to_JSON(assignOp.getDst()));
    params.push_back(to_JSON(assignOp.getSrc()));
    res["parameters"] = std::move(params);
    return res;
}

json::Value to_JSON(BMv2IR::ExtractOp extractOp) {
    json::Object res;
    res["op"] = "extract";
    // TODO: add support for non-regular extracts
    json::Array parameters;
    auto type = extractOp.getExtractType();
    if (type == BMv2IR::ExtractKind::Regular) {
        json::Object desc;
        desc["type"] = "regular";
        desc["value"] = extractOp.getHeaderInstance().getLeafReference().getValue();
        parameters.push_back(std::move(desc));
    } else {
        llvm_unreachable("Non-regular extracts not yet supported");
    }
    res["parameters"] = std::move(parameters);
    return res;
}

// Returns true for Operations that we want to emit directly (basically "root" operations for
// expression trees etc)
bool isPrimitive(Operation *op) {
    return isa<BMv2IR::AssignOp, BMv2IR::AssignHeaderOp, BMv2IR::ExtractOp, BMv2IR::LookaheadOp>(
        op);
}

json::Value to_JSON(BMv2IR::ParserStateOp stateOp) {
    json::Object res;
    res["name"] = stateOp.getSymName();
    res["id"] = getId(stateOp);

    json::Array transitions;
    if (!stateOp.getTransitions().empty()) {
        for (auto &op : stateOp.getTransitions().front()) {
            transitions.push_back(to_JSON(cast<BMv2IR::TransitionOp>(&op)));
        }
    }
    res["transitions"] = std::move(transitions);

    json::Array keys;
    if (!stateOp.getTransitionKeys().empty()) {
        for (auto &op : stateOp.getTransitionKeys().front()) {
            keys.push_back(to_JSON(&op));
        }
    }
    res["transition_key"] = std::move(keys);

    json::Array ops;
    if (!stateOp.getParserOps().empty()) {
        for (auto &op : stateOp.getParserOps().front()) {
            if (isPrimitive(&op)) ops.push_back(to_JSON(&op));
        }
    }
    res["parser_ops"] = std::move(ops);

    return res;
}

json::Value to_JSON(BMv2IR::ParserOp parserOp) {
    json::Object res;
    res["name"] = parserOp.getSymName();
    res["init_state"] = parserOp.getInitState().getLeafReference().getValue();
    res["id"] = getId(parserOp);

    json::Array states;
    parserOp.walk([&states](BMv2IR::ParserStateOp stateOp) { states.push_back(to_JSON(stateOp)); });

    res["parse_states"] = std::move(states);

    return res;
}

json::Value actionToJSON(P4HIR::FuncOp actionOp) {
    json::Object res;
    res["name"] = actionOp.getSymName();
    res["id"] = getId(actionOp);

    json::Array params;
    for (auto arg : actionOp.getArguments()) {
        json::Object paramDesc;
        paramDesc["name"] = actionOp.getArgumentName(arg.getArgNumber()).getValue();
        paramDesc["bitwidth"] = cast<P4HIR::BitsType>(arg.getType()).getWidth();
        params.push_back(std::move(paramDesc));
    }
    res["runtime_data"] = std::move(params);

    json::Array ops;
    for (auto &op : actionOp.getOps()) {
        if (isPrimitive(&op)) ops.push_back(to_JSON(&op));
    }
    res["primitives"] = std::move(ops);
    return res;
}

json::Value to_JSON(P4HIR::FuncOp funcOp) {
    if (funcOp.getAction()) return actionToJSON(funcOp);
    llvm_unreachable("Only Actions JSON conversion supported");
}

json::Value to_JSON(BMv2IR::ConditionalOp conditional) {
    json::Object res;
    res["name"] = conditional.getSymName();
    res["id"] = getId(conditional);
    res["true_next"] = conditional.getThenRef().getLeafReference().getValue();
    auto elseVal = conditional.getElseRef();
    if (elseVal.has_value())
        res["false_next"] = elseVal->getLeafReference().getValue();
    else
        res["false_next"] = json::Value(nullptr);
    res["expression"] = to_JSON(conditional.getConditionRegion().front().getTerminator());
    return res;
}

json::Value to_JSON(BMv2IR::TableOp tableOp) {
    json::Object res;
    res["name"] = tableOp.getSymName();
    res["id"] = getId(tableOp);
    res["max_size"] = tableOp.getSize();
    res["match_type"] = BMv2IR::stringifyTableMatchKind(tableOp.getMatchType());
    res["type"] = BMv2IR::stringifyTableType(tableOp.getTableType());
    res["support_timeout"] = tableOp.getSupportTimeout();

    json::Array key;
    for (auto attr : tableOp.getKeys()) {
        auto keyAttr = cast<BMv2IR::TableKeyAttr>(attr);
        json::Object keyEntry;
        auto matchTy = keyAttr.getMatchType();
        keyEntry["match_type"] = BMv2IR::stringifyTableMatchKind(matchTy);
        auto fieldNameAttr = keyAttr.getFieldName();
        auto headerName = keyAttr.getHeader().getLeafReference().getValue();
        auto fieldAttr = keyAttr.getFieldName();
        if (matchTy == BMv2IR::TableMatchKind::Valid) {
            assert(fieldAttr == nullptr && "Valid match kind expects only header name");
            keyEntry["target"] = json::Array{headerName};
        } else {
            keyEntry["target"] = json::Array{headerName, fieldNameAttr.getValue()};
        }
        auto mask = keyAttr.getMask();
        if (mask) keyEntry["mask"] = mask.getInt();
        key.push_back(std::move(keyEntry));
    }
    res["key"] = std::move(key);

    json::Array actions;
    json::Array actionIds;
    auto moduleOp = tableOp->getParentOfType<ModuleOp>();
    assert(moduleOp && "Expected parent module");
    auto getIdForAction = [&](SymbolRefAttr ref) {
        auto defOp = SymbolTable::lookupSymbolIn(moduleOp, ref);
        assert(defOp && "Can't find symbol def");
        auto funcOp = cast<P4HIR::FuncOp>(defOp);
        return getId(funcOp);
    };
    for (auto attr : tableOp.getActions()) {
        auto actionRef = cast<SymbolRefAttr>(attr);
        actions.emplace_back(actionRef.getLeafReference().getValue());
        actionIds.push_back(getIdForAction(actionRef));
    }
    res["actions"] = std::move(actions);
    res["action_ids"] = std::move(actionIds);

    json::Object nextTables;
    if (auto nextTablesArray = dyn_cast_or_null<ArrayAttr>(tableOp.getNextTables())) {
        for (auto attr : nextTablesArray) {
            auto actionTable = cast<BMv2IR::ActionTableAttr>(attr);
            auto action = actionTable.getAction().getLeafReference().getValue();
            auto nextTableAttr = actionTable.getTable();
            auto table =
                nextTableAttr ? nextTableAttr.getLeafReference().getValue() : json::Value(nullptr);
            nextTables[action] = table;
        }
    } else if (auto hitMissAttr =
                   dyn_cast_or_null<BMv2IR::HitOrMissAttr>(tableOp.getNextTables())) {
        nextTables["__HIT__"] = hitMissAttr.getHitNode().getLeafReference().getValue();
        nextTables["__MISS__"] = hitMissAttr.getMissNode().getLeafReference().getValue();
    } else {
        llvm_unreachable("Unsupported next_tables attribute");
    }
    res["next_tables"] = std::move(nextTables);

    auto defaultEntryAttr = tableOp.getDefaultEntry();
    if (defaultEntryAttr.has_value()) {
        json::Object defaultEntry;
        defaultEntry["action_id"] = getIdForAction(defaultEntryAttr->getAction());
        defaultEntry["action_const"] = defaultEntryAttr->getActionConst();
        defaultEntry["action_entry_const"] = defaultEntryAttr->getActionEntryConst();
        json::Array actionData;
        for (auto &data : defaultEntryAttr->getActionData()) {
            actionData.push_back(data);
        }
        defaultEntry["action_data"] = std::move(actionData);
        res["default_entry"] = std::move(defaultEntry);
    }

    return res;
}

json::Value to_JSON(BMv2IR::PipelineOp pipeline) {
    json::Object res;
    res["name"] = pipeline.getSymName();
    res["id"] = getId(pipeline);
    auto maybeInitTable = pipeline.getInitTable();
    if (maybeInitTable.has_value())
        res["init_table"] = maybeInitTable->getLeafReference().getValue();
    else
        res["init_table"] = json::Value(nullptr);

    json::Array tables;
    pipeline.walk([&](BMv2IR::TableOp table) { tables.push_back(to_JSON(table)); });
    res["tables"] = std::move(tables);

    json::Array conditionals;
    pipeline.walk(
        [&](BMv2IR::ConditionalOp conditional) { conditionals.push_back(to_JSON(conditional)); });
    res["conditionals"] = std::move(conditionals);

    return res;
}

json::Value to_JSON(P4HIR::ConstOp constOp) {
    return llvm::TypeSwitch<TypedAttr, json::Value>(constOp.getValue())
        .Case([](P4HIR::IntAttr intAttr) {
            json::Object res;
            res["type"] = "hexstr";
            res["value"] = asHexstr(intAttr);
            return res;
        })
        .Case([](P4HIR::BoolAttr boolAttr) { return json::Value(boolAttr.getValue()); });
}

json::Value asExpressionNode(json::Value val) {
    json::Object res;
    res["type"] = "expression";
    res["value"] = std::move(val);
    return res;
}

json::Value to_JSON(P4HIR::BinOp binOp) {
    json::Object value;
    auto kindToString = [](P4HIR::BinOpKind kind) {
        switch (kind) {
            case P4HIR::BinOpKind::Add:
                return "+";
            case P4HIR::BinOpKind::Sub:
                return "-";
            case P4HIR::BinOpKind::Mul:
                return "*";
            case P4HIR::BinOpKind::Div:
                return "/";
            case P4HIR::BinOpKind::And:
                return "and";
            case P4HIR::BinOpKind::Or:
                return "or";
            default:
                llvm_unreachable("Unhandled opkind");
        }
    };
    StringRef op = kindToString(binOp.getKind());
    value["op"] = op;
    value["left"] = to_JSON(binOp.getLhs());
    value["right"] = to_JSON(binOp.getRhs());
    return asExpressionNode(std::move(value));
}

json::Value to_JSON(P4HIR::CmpOp cmpOp) {
    json::Object value;
    auto kindToString = [](P4HIR::CmpOpKind kind) {
        switch (kind) {
            case P4HIR::CmpOpKind::Eq:
                return "==";
            case P4HIR::CmpOpKind::Ne:
                return "!=";
            case P4HIR::CmpOpKind::Ge:
                return ">=";
            case P4HIR::CmpOpKind::Gt:
                return ">";
            case P4HIR::CmpOpKind::Le:
                return "<=";
            case P4HIR::CmpOpKind::Lt:
                return "<";
        }
    };
    StringRef op = kindToString(cmpOp.getKind());
    value["op"] = op;
    value["left"] = to_JSON(cmpOp.getLhs());
    value["right"] = to_JSON(cmpOp.getRhs());
    return asExpressionNode(std::move(value));
}

json::Value to_JSON(BMv2IR::YieldOp yieldOp) {
    auto args = yieldOp.getArgs();
    assert(args.size() == 1 && "Unhandled number of yield args");
    return to_JSON(args[0]);
}

json::Value to_JSON(BMv2IR::DataToBoolOp d2b) {
    json::Object res;
    res["op"] = "d2b";
    res["left"] = json::Value(nullptr);
    res["right"] = to_JSON(d2b.getInput());
    return asExpressionNode(std::move(res));
}

json::Value to_JSON(BMv2IR::DeparserOp deparserOp) {
  json::Object res;
  res["name"] = deparserOp.getSymName();
  res["id"] = getId(deparserOp);
  json::Array order;
  for (auto a : deparserOp.getOrder()) {
    auto ref = cast<SymbolRefAttr>(a);
    order.push_back(ref.getLeafReference().getValue());
  }
  res["order"] = std::move(order);
  return res;
}

json::Value to_JSON(BMv2IR::CalculationOp calcOp) {
    json::Object res;
    res["name"] = calcOp.getSymName();
    res["id"] = getId(calcOp);

    json::Array input;
    auto yieldTerminator = cast<BMv2IR::YieldOp>(calcOp.getInputsRegion().front().getTerminator());
    for (auto yieldedVal : yieldTerminator.getArgs()) {
        Operation *op = yieldedVal.getDefiningOp();
        assert(op && "Expected yielded value to come from operation");
        input.push_back(to_JSON(op));
    }
    res["input"] = std::move(input);
    return res;
}

json::Value to_JSON(BMv2IR::ChecksumOp checksumOp) {
    json::Object res;
    res["name"] = checksumOp.getSymName();
    res["id"] = getId(checksumOp);
    res["type"] = checksumOp.getType();
    res["target"] = json::Array{checksumOp.getTargetHeader().getLeafReference().getValue(),
                                checksumOp.getTargetField()};
    res["calculation"] = checksumOp.getCalculation().getLeafReference().getValue();
    res["update"] = checksumOp.getUpdate();
    res["verify"] = !checksumOp.getUpdate();
    res["if_cond"] = to_JSON(checksumOp.getIfCondRegion().front().getTerminator());

    return res;
}

json::Value to_JSON(Operation *op) {
    return llvm::TypeSwitch<Operation *, json::Value>(op)
        .Case([](BMv2IR::AssignHeaderOp assignOp) { return to_JSON(assignOp); })
        .Case([](BMv2IR::LookaheadOp lookAheadOp) { return to_JSON(lookAheadOp); })
        .Case([](BMv2IR::AssignOp assignOp) { return to_JSON(assignOp); })
        .Case([](BMv2IR::ExtractOp extractOp) { return to_JSON(extractOp); })
        .Case([](BMv2IR::FieldOp fieldOp) { return to_JSON(fieldOp); })
        .Case([](P4HIR::ConstOp constOp) { return to_JSON(constOp); })
        .Case([](P4HIR::BinOp binOp) { return to_JSON(binOp); })
        .Case([](BMv2IR::YieldOp yieldOp) { return to_JSON(yieldOp); })
        .Case([](BMv2IR::DataToBoolOp d2bOp) { return to_JSON(d2bOp); })
        .Case([](P4HIR::CmpOp cmpOp) { return to_JSON(cmpOp); })
        .Default([](Operation *op) -> json::Value {
            llvm::errs() << "Unsupported op: " << op->getName().getIdentifier() << "\n";
            llvm_unreachable("Unsupported op");
        });
}

json::Value to_JSON(BlockArgument arg) {
    auto parent = arg.getParentBlock()->getParentOp();
    assert(parent && "Expected blockarg parentOp");
    auto funcOp = cast<P4HIR::FuncOp>(parent);
    assert(funcOp.getAction() && "Expected action");
    json::Object res;

    res["type"] = "runtime_data";
    res["value"] = arg.getArgNumber();
    return res;
}

json::Value to_JSON(Value val) {
    if (auto op = val.getDefiningOp()) return to_JSON(op);
    return to_JSON(cast<BlockArgument>(val));
}
}  // anonymous namespace

mlir::FailureOr<json::Value> P4::P4MLIR::bmv2irToJson(ModuleOp moduleOp) {
    // P4HIR/BMv2IR use Symbols/Values to reference other IR constructs, while some
    // BMv2 nodes require unique IDs. We add IDs as attributes right before converting
    // to ensure that they are unique and to avoid tracking them while manipulating IR earlier
    // in the pipeline.
    setUniqueIDS(moduleOp);
    json::Object root;

    // Emit header types and header instances
    SmallVector<BMv2IR::HeaderInstanceOp> headerInstances;
    moduleOp.walk([&](BMv2IR::HeaderInstanceOp instance) { headerInstances.push_back(instance); });
    llvm::SetVector<Type> headersTy;
    json::Array headerTyNodes;
    json::Array headerInstanceNodes;
    int64_t headerTypeID = 0;
    for (auto instance : headerInstances) {
        auto headerTy = cast<BMv2IR::HeaderType>(instance.getHeaderType());
        if (!headerTy) return instance.emitError("Unexpected type");
        bool inserted = headersTy.insert(headerTy);
        if (inserted) {
            headerTyNodes.push_back(to_JSON(headerTy, headerTypeID));
            headerTypeID++;
        }
        headerInstanceNodes.push_back(to_JSON(instance));
    }

    root["header_types"] = std::move(headerTyNodes);
    root["headers"] = std::move(headerInstanceNodes);

    // Emit parsers
    json::Array parsers;
    moduleOp.walk([&](BMv2IR::ParserOp parserOp) { parsers.push_back(to_JSON(parserOp)); });
    root["parsers"] = std::move(parsers);

    // Emit actions
    json::Array actions;
    moduleOp.walk([&](P4HIR::FuncOp funcOp) {
        if (funcOp.getAction()) actions.push_back(to_JSON(funcOp));
    });
    root["actions"] = std::move(actions);

    // Emit pipelines
    json::Array pipelines;
    moduleOp.walk([&](BMv2IR::PipelineOp pipeline) { pipelines.push_back(to_JSON(pipeline)); });
    root["pipelines"] = std::move(pipelines);

    // Emit deparsers
    json::Array deparsers;
    moduleOp.walk([&](BMv2IR::DeparserOp deparserOp) { deparsers.push_back(to_JSON(deparserOp)); });
    root["deparsers"] = std::move(deparsers);

    // Emit calculations
    json::Array calculations;
    moduleOp.walk([&](BMv2IR::CalculationOp calcOp) { calculations.push_back(to_JSON(calcOp)); });
    root["calculations"] = std::move(calculations);

    // Emit checksums
    json::Array checksums;
    moduleOp.walk([&](BMv2IR::ChecksumOp checksumOp) { checksums.push_back(to_JSON(checksumOp)); });
    root["checksums"] = std::move(checksums);

    json::Value res(std::move(root));

    return res;
}

void P4::P4MLIR::registerToBMv2JSONTranslation() {
    TranslateFromMLIRRegistration registration(
        "p4hir-to-bmv2-json", "Translate MLIR to BMv2 JSON",
        [](Operation *op, raw_ostream &output) {
            auto moduleOp = dyn_cast<ModuleOp>(op);
            if (!moduleOp) return failure();
            if (failed(bmv2irToJson(moduleOp, output))) return failure();
            return success();
        },
        [](DialectRegistry &registry) { P4::P4MLIR::registerAllDialects(registry); });
}

LogicalResult P4::P4MLIR::bmv2irToJson(ModuleOp moduleOp, raw_ostream &output) {
    auto maybeJsonModule = bmv2irToJson(moduleOp);
    if (failed(maybeJsonModule)) return failure();

    output << llvm::formatv("{0:2}", *maybeJsonModule) << "\n";
    return success();
}
