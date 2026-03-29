#include "p4mlir/Targets/BMv2/Target.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "p4mlir/Common/Registration.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Attrs.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"

using namespace mlir;
using namespace P4::P4MLIR;

namespace {

/// Convert an MLIR attribute to its JSON string representation
static std::string attrToJsonString(mlir::Attribute attr) {
    std::string result;
    llvm::raw_string_ostream os(result);
    attr.print(os);
    return os.str();
}

/// Serialize parser operation to BMv2 JSON (extract: regular, stack, union_stack).
static llvm::json::Object serializeParserOp(Operation *op) {
    llvm::json::Object opJson;

    if (auto extractOp = dyn_cast<BMv2IR::ExtractOp>(op)) {
        opJson["op"] = "extract";
        llvm::json::Object param;

        auto extractKind = extractOp.getExtractType();
        if (extractKind == BMv2IR::ExtractKind::Regular) {
            param["type"] = "regular";
            param["value"] = extractOp.getValue().str();
        } else if (extractKind == BMv2IR::ExtractKind::Stack) {
            param["type"] = "stack";
            param["value"] = extractOp.getValue().str();
        } else if (extractKind == BMv2IR::ExtractKind::UnionStack) {
            param["type"] = "union_stack";
            llvm::json::Array unionValue;
            unionValue.push_back(extractOp.getValue().str());
            if (extractOp.getUnionMember())
                unionValue.push_back(extractOp.getUnionMember()->str());
            param["value"] = std::move(unionValue);
        }

        llvm::json::Array parameters;
        parameters.push_back(std::move(param));
        opJson["parameters"] = std::move(parameters);
    }

    return opJson;
}

/// Serialize transition key to BMv2 JSON (field, lookahead, stack_field, union_stack_field).
static llvm::json::Object serializeTransitionKey(Operation *op) {
    llvm::json::Object keyJson;

    if (auto fieldOp = dyn_cast<BMv2IR::FieldOp>(op)) {
        keyJson["type"] = "field";
        llvm::json::Array value;
        value.push_back(fieldOp.getHeaderInstance().str());
        value.push_back(fieldOp.getFieldMember().str());
        keyJson["value"] = std::move(value);
    } else if (auto lookaheadOp = dyn_cast<BMv2IR::LookaheadOp>(op)) {
        keyJson["type"] = "lookahead";
        llvm::json::Array value;
        value.push_back(lookaheadOp.getBitOffset());
        value.push_back(lookaheadOp.getBitwidth());
        keyJson["value"] = std::move(value);
    } else if (auto stackFieldOp = dyn_cast<BMv2IR::StackFieldOp>(op)) {
        keyJson["type"] = "stack_field";
        llvm::json::Array value;
        value.push_back(stackFieldOp.getHeaderStack().str());
        value.push_back(stackFieldOp.getFieldMember().str());
        keyJson["value"] = std::move(value);
    } else if (auto unionStackFieldOp = dyn_cast<BMv2IR::UnionStackFieldOp>(op)) {
        keyJson["type"] = "union_stack_field";
        llvm::json::Array value;
        value.push_back(unionStackFieldOp.getHeaderUnionStack().str());
        value.push_back(unionStackFieldOp.getUnionMember().str());
        value.push_back(unionStackFieldOp.getFieldMember().str());
        keyJson["value"] = std::move(value);
    }

    return keyJson;
}

/// Serialize parser state transition to BMv2 JSON (default, hexstr, parse_vset).
static llvm::json::Object serializeTransition(BMv2IR::TransitionOp transitionOp) {
    llvm::json::Object transJson;

    auto kind = transitionOp.getType();
    if (kind == BMv2IR::TransitionKind::Default) {
        transJson["type"] = "default";
        transJson["value"] = nullptr;
        transJson["mask"] = nullptr;
    } else if (kind == BMv2IR::TransitionKind::Hexstr) {
        transJson["type"] = "hexstr";
        transJson["value"] = transitionOp.getValue()
            ? llvm::json::Value(attrToJsonString(*transitionOp.getValue()))
            : llvm::json::Value(nullptr);
        transJson["mask"] = transitionOp.getMask()
            ? llvm::json::Value(attrToJsonString(*transitionOp.getMask()))
            : llvm::json::Value(nullptr);
    } else if (kind == BMv2IR::TransitionKind::Parse_vset) {
        transJson["type"] = "parse_vset";
        transJson["value"] = transitionOp.getValue()
            ? llvm::json::Value(attrToJsonString(*transitionOp.getValue()))
            : llvm::json::Value(nullptr);
        transJson["mask"] = nullptr;
    }

    // Set next_state: use state name or null for end of parsing
    transJson["next_state"] = transitionOp.getNextState()
        ? llvm::json::Value(transitionOp.getNextState()->getLeafReference().str())
        : llvm::json::Value(nullptr);

    return transJson;
}

/// Serialize parser state to BMv2 JSON with operations, transition keys, and transitions.
static llvm::json::Object serializeParserState(BMv2IR::ParserStateOp stateOp, int stateId) {
    llvm::json::Object stateJson;
    stateJson["name"] = stateOp.getSymName().str();
    stateJson["id"] = stateId;

    // Serialize parser operations (e.g., extract header fields)
    llvm::json::Array parserOps;
    for (auto &op : stateOp.getParserOps().front()) {
        if (isa<BMv2IR::ExtractOp>(&op))
            parserOps.push_back(serializeParserOp(&op));
    }
    stateJson["parser_ops"] = std::move(parserOps);

    // Serialize transition keys (fields used to determine next state)
    llvm::json::Array transitionKeys;
    for (auto &op : stateOp.getTransitionKeys().front()) {
        if (isa<BMv2IR::FieldOp, BMv2IR::LookaheadOp,
                BMv2IR::StackFieldOp, BMv2IR::UnionStackFieldOp>(&op))
            transitionKeys.push_back(serializeTransitionKey(&op));
    }
    stateJson["transition_key"] = std::move(transitionKeys);

    // Serialize state transitions
    llvm::json::Array transitions;
    for (auto &op : stateOp.getTransitions().front()) {
        if (auto transOp = dyn_cast<BMv2IR::TransitionOp>(&op))
            transitions.push_back(serializeTransition(transOp));
    }
    stateJson["transitions"] = std::move(transitions);

    return stateJson;
}

/// Serialize complete parser to BMv2 JSON with initial state and parse states.
static llvm::json::Object serializeParser(BMv2IR::ParserOp parserOp, int parserId) {
    // Collect all parse states
    llvm::json::Array parseStates;
    int stateId = 0;
    for (auto &op : parserOp.getBody().front()) {
        if (auto stateOp = dyn_cast<BMv2IR::ParserStateOp>(&op))
            parseStates.push_back(serializeParserState(stateOp, stateId++));
    }

    // Build parser JSON object
    llvm::json::Object parserJson;
    parserJson["name"] = parserOp.getSymName().str();
    parserJson["id"] = parserId;
    parserJson["init_state"] = parserOp.getInitState().getLeafReference().str();
    parserJson["parse_states"] = std::move(parseStates);

    return parserJson;
}

}  // namespace

/// Convert BMv2IR module to BMv2 JSON. Currently supports parsers only.
mlir::FailureOr<llvm::json::Value> P4::P4MLIR::bmv2irToJson(ModuleOp moduleOp) {
    llvm::json::Array parsers;

    // Collect all parsers from the module
    int parserId = 0;
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if (auto parserOp = dyn_cast<BMv2IR::ParserOp>(&op))
            parsers.push_back(serializeParser(parserOp, parserId++));
    }

    llvm::json::Object rootJson;
    rootJson["parsers"] = std::move(parsers);

    return llvm::json::Value(std::move(rootJson));
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

    output << *maybeJsonModule;
    return success();
}
