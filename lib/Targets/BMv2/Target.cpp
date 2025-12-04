#include "p4mlir/Targets/BMv2/Target.h"

#include <string>
#include <vector>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "p4mlir/Common/Registration.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
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
            // FIXME: print value as hexadecimal
            auto value = std::to_string(valueAttr.getValue().getSExtValue());
            res["value"] = value;
            auto mask = transitionOp.getMask();
            if (mask.has_value()) {
                auto maskAttr = cast<P4HIR::IntAttr>(mask.value());
                auto mask = std::to_string(maskAttr.getValue().getSExtValue());
                res["mask"] = mask;
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
            llvm_unreachable("JSON translation for parse_vsets NYE");
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

// Returns true for Operations that we don't want to emit directly
// when emitting lists of primitives
static bool skipOpEmission(Operation *op) { return isa<BMv2IR::FieldOp, P4HIR::ReturnOp>(op); }

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
            if (!skipOpEmission(&op)) ops.push_back(to_JSON(&op));
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
        if (!skipOpEmission(&op)) ops.push_back(to_JSON(&op));
    }
    res["primitives"] = std::move(ops);
    return res;
}

json::Value to_JSON(P4HIR::FuncOp funcOp) {
    if (funcOp.getAction()) return actionToJSON(funcOp);
    llvm_unreachable("Only Actions JSON conversion supported");
}

json::Value to_JSON(Operation *op) {
    return llvm::TypeSwitch<Operation *, json::Value>(op)
        .Case([](BMv2IR::AssignHeaderOp assignOp) { return to_JSON(assignOp); })
        .Case([](BMv2IR::LookaheadOp lookAheadOp) { return to_JSON(lookAheadOp); })
        .Case([](BMv2IR::AssignOp assignOp) { return to_JSON(assignOp); })
        .Case([](BMv2IR::ExtractOp extractOp) { return to_JSON(extractOp); })
        .Case([](BMv2IR::FieldOp fieldOp) { return to_JSON(fieldOp); })
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

static json::Value to_JSON(Value val) {
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
