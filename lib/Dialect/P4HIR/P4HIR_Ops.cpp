#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
    if (mlir::isa<P4HIR::BoolAttr>(attrType)) {
        if (!mlir::isa<P4HIR::BoolType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.bool' for '" << attrType << "'";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr>(attrType)) {
        if (!mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") does not match value type (" << attrType << ")";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr, P4HIR::BoolAttr>(attrType)) return success();

    if (mlir::isa<P4HIR::AggAttr>(attrType)) {
        if (!mlir::isa<P4HIR::StructType>(opType))
            return op->emitOpError("result type (") << opType << ") is not an aggregate type";

        return success();
    }

    assert(isa<TypedAttr>(attrType) && "expected typed attribute");
    return op->emitOpError("constant with type ")
           << cast<TypedAttr>(attrType).getType() << " not supported";
}

LogicalResult P4HIR::ConstOp::verify() {
    // ODS already generates checks to make sure the result type is valid. We just
    // need to additionally check that the value's attribute type is consistent
    // with the result type.
    return checkConstantTypes(getOperation(), getType(), getValue());
}

void P4HIR::ConstOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    if (getName() && !getName()->empty()) {
        setNameFn(getResult(), *getName());
        return;
    }

    auto type = getType();
    if (auto intCst = mlir::dyn_cast<P4HIR::IntAttr>(getValue())) {
        auto intType = mlir::dyn_cast<P4HIR::BitsType>(type);

        // Build a complex name with the value and type.
        llvm::SmallString<32> specialNameBuffer;
        llvm::raw_svector_ostream specialName(specialNameBuffer);
        specialName << 'c' << intCst.getValue();
        if (intType) specialName << '_' << intType.getAlias();
        setNameFn(getResult(), specialName.str());
    } else if (auto boolCst = mlir::dyn_cast<P4HIR::BoolAttr>(getValue())) {
        setNameFn(getResult(), boolCst.getValue() ? "true" : "false");
    } else {
        setNameFn(getResult(), "cst");
    }
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void P4HIR::CastOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "cast");
}

//===----------------------------------------------------------------------===//
// ReadOp
//===----------------------------------------------------------------------===//

void P4HIR::ReadOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "val");
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

LogicalResult P4HIR::UnaryOp::verify() {
    switch (getKind()) {
        case P4HIR::UnaryOpKind::Neg:
        case P4HIR::UnaryOpKind::UPlus:
        case P4HIR::UnaryOpKind::Cmpl:
        case P4HIR::UnaryOpKind::LNot:
            // Nothing to verify.
            return success();
    }

    llvm_unreachable("Unknown UnaryOp kind?");
}

void P4HIR::UnaryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

void P4HIR::BinOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult P4HIR::ConcatOp::verify() {
    auto lhsType = cast<BitsType>(getOperand(0).getType());
    auto rhsType = cast<BitsType>(getOperand(1).getType());
    auto resultType = cast<BitsType>(getResult().getType());

    auto expectedWidth = lhsType.getWidth() + rhsType.getWidth();
    if (resultType.getWidth() != expectedWidth)
        return emitOpError() << "the resulting width of a concatenation operation must equal the "
                                "sum of the operand widths";

    if (resultType.isSigned() != lhsType.isSigned())
        return emitOpError() << "the signedness of the concatenation result must match the "
                                "signedness of the left-hand side operand";

    return success();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

void P4HIR::CmpOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void P4HIR::VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    if (getName() && !getName()->empty()) setNameFn(getResult(), *getName());
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

void P4HIR::ScopeOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                         SmallVectorImpl<RegionSuccessor> &regions) {
    // The only region always branch back to the parent operation.
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor(getODSResults(0)));
        return;
    }

    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getScopeRegion()));
}

void P4HIR::ScopeOp::build(OpBuilder &builder, OperationState &result,
                           function_ref<void(OpBuilder &, Type &, Location)> scopeBuilder) {
    assert(scopeBuilder && "the builder callback for 'then' must be present");

    OpBuilder::InsertionGuard guard(builder);
    Region *scopeRegion = result.addRegion();
    builder.createBlock(scopeRegion);

    mlir::Type yieldTy;
    scopeBuilder(builder, yieldTy, result.location);

    if (yieldTy) result.addTypes(TypeRange{yieldTy});
}

void P4HIR::ScopeOp::build(OpBuilder &builder, OperationState &result,
                           function_ref<void(OpBuilder &, Location)> scopeBuilder) {
    assert(scopeBuilder && "the builder callback for 'then' must be present");
    OpBuilder::InsertionGuard guard(builder);
    Region *scopeRegion = result.addRegion();
    builder.createBlock(scopeRegion);
    scopeBuilder(builder, result.location);
}

LogicalResult P4HIR::ScopeOp::verify() {
    if (getScopeRegion().empty()) {
        return emitOpError() << "p4hir.scope must not be empty since it should "
                                "include at least an implicit p4hir.yield ";
    }

    if (getScopeRegion().back().empty() || !getScopeRegion().back().mightHaveTerminator() ||
        !getScopeRegion().back().getTerminator()->hasTrait<OpTrait::IsTerminator>())
        return emitOpError() << "last block of p4hir.scope must be terminated";
    return success();
}
//===----------------------------------------------------------------------===//
// Custom Parsers & Printers
//===----------------------------------------------------------------------===//

// Check if a region's termination omission is valid and, if so, creates and
// inserts the omitted terminator into the region.
static LogicalResult ensureRegionTerm(OpAsmParser &parser, Region &region, SMLoc errLoc) {
    Location eLoc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    OpBuilder builder(parser.getBuilder().getContext());

    // Insert empty block in case the region is empty to ensure the terminator
    // will be inserted
    if (region.empty()) builder.createBlock(&region);

    Block &block = region.back();
    // Region is properly terminated: nothing to do.
    if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) return success();

    // Check for invalid terminator omissions.
    if (!region.hasOneBlock())
        return parser.emitError(errLoc, "multi-block region must not omit terminator");

    // Terminator was omitted correctly: recreate it.
    builder.setInsertionPointToEnd(&block);
    builder.create<P4HIR::YieldOp>(eLoc);
    return success();
}

static mlir::ParseResult parseOmittedTerminatorRegion(mlir::OpAsmParser &parser,
                                                      mlir::Region &scopeRegion) {
    auto regionLoc = parser.getCurrentLocation();
    if (parser.parseRegion(scopeRegion)) return failure();
    if (ensureRegionTerm(parser, scopeRegion, regionLoc).failed()) return failure();

    return success();
}

// True if the region's terminator should be omitted.
bool omitRegionTerm(mlir::Region &r) {
    const auto singleNonEmptyBlock = r.hasOneBlock() && !r.back().empty();
    const auto yieldsNothing = [&r]() {
        auto y = dyn_cast<P4HIR::YieldOp>(r.back().getTerminator());
        return y && y.getArgs().empty();
    };
    return singleNonEmptyBlock && yieldsNothing();
}

static void printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer, P4HIR::ScopeOp &,
                                         mlir::Region &scopeRegion) {
    printer.printRegion(scopeRegion,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/!omitRegionTerm(scopeRegion));
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

void P4HIR::TernaryOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                           SmallVectorImpl<RegionSuccessor> &regions) {
    // The `true` and the `false` region branch back to the parent operation.
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor(this->getODSResults(0)));
        return;
    }

    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getTrueRegion()));
    regions.push_back(RegionSuccessor(&getFalseRegion()));
}

void P4HIR::TernaryOp::build(OpBuilder &builder, OperationState &result, Value cond,
                             function_ref<void(OpBuilder &, Location)> trueBuilder,
                             function_ref<void(OpBuilder &, Location)> falseBuilder) {
    result.addOperands(cond);
    OpBuilder::InsertionGuard guard(builder);
    Region *trueRegion = result.addRegion();
    auto *block = builder.createBlock(trueRegion);
    trueBuilder(builder, result.location);
    Region *falseRegion = result.addRegion();
    builder.createBlock(falseRegion);
    falseBuilder(builder, result.location);

    auto yield = dyn_cast<YieldOp>(block->getTerminator());
    assert((yield && yield.getNumOperands() <= 1) && "expected zero or one result type");
    if (yield.getNumOperands() == 1) result.addTypes(TypeRange{yield.getOperandTypes().front()});
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::IfOp::parse(OpAsmParser &parser, OperationState &result) {
    // Create the regions for 'then'.
    result.regions.reserve(2);
    Region *thenRegion = result.addRegion();
    Region *elseRegion = result.addRegion();

    auto &builder = parser.getBuilder();
    OpAsmParser::UnresolvedOperand cond;
    Type boolType = P4HIR::BoolType::get(builder.getContext());

    if (parser.parseOperand(cond) || parser.resolveOperand(cond, boolType, result.operands))
        return failure();

    // Parse the 'then' region.
    auto parseThenLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                           /*argTypes=*/{}))
        return failure();
    if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed()) return failure();

    // If we find an 'else' keyword, parse the 'else' region.
    if (!parser.parseOptionalKeyword("else")) {
        auto parseElseLoc = parser.getCurrentLocation();
        if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{})) return failure();
        if (ensureRegionTerm(parser, *elseRegion, parseElseLoc).failed()) return failure();
    }

    // Parse the optional attribute list.
    return parser.parseOptionalAttrDict(result.attributes) ? failure() : success();
}

void P4HIR::IfOp::print(OpAsmPrinter &p) {
    p << " " << getCondition() << " ";
    auto &thenRegion = this->getThenRegion();
    p.printRegion(thenRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

    // Print the 'else' regions if it exists and has a block.
    auto &elseRegion = this->getElseRegion();
    if (!elseRegion.empty()) {
        p << " else ";
        p.printRegion(elseRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(elseRegion));
    }

    p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders.
void P4HIR::buildTerminatedBody(OpBuilder &builder, Location loc) {
    Block *block = builder.getBlock();

    // Region is properly terminated: nothing to do.
    if (block->mightHaveTerminator()) return;

    // add p4hir.yield to the end of the block
    builder.create<P4HIR::YieldOp>(loc);
}

void P4HIR::IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                      SmallVectorImpl<RegionSuccessor> &regions) {
    // The `then` and the `else` region branch back to the parent operation.
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor());
        return;
    }

    // Don't consider the else region if it is empty.
    Region *elseRegion = &this->getElseRegion();
    if (elseRegion->empty()) elseRegion = nullptr;

    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getThenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion) regions.push_back(RegionSuccessor(elseRegion));
}

void P4HIR::IfOp::build(OpBuilder &builder, OperationState &result, Value cond, bool withElseRegion,
                        function_ref<void(OpBuilder &, Location)> thenBuilder,
                        function_ref<void(OpBuilder &, Location)> elseBuilder) {
    assert(thenBuilder && "the builder callback for 'then' must be present");

    result.addOperands(cond);

    OpBuilder::InsertionGuard guard(builder);
    Region *thenRegion = result.addRegion();
    builder.createBlock(thenRegion);
    thenBuilder(builder, result.location);

    Region *elseRegion = result.addRegion();
    if (!withElseRegion) return;

    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
}

mlir::LogicalResult P4HIR::ReturnOp::verify() {
    // Returns can be present in multiple different scopes, get the
    // wrapping function and start from there.
    auto *fnOp = getOperation()->getParentOp();
    while (!isa<P4HIR::FuncOp>(fnOp)) fnOp = fnOp->getParentOp();

    // ReturnOps currently only have a single optional operand.
    if (getNumOperands() > 1) return emitOpError() << "expects at most 1 return operand";

    // Ensure returned type matches the function signature.
    auto expectedTy = cast<P4HIR::FuncOp>(fnOp).getFunctionType().getReturnType();
    auto actualTy =
        (getNumOperands() == 0 ? P4HIR::VoidType::get(getContext()) : getOperand(0).getType());
    if (actualTy != expectedTy)
        return emitOpError() << "returns " << actualTy << " but enclosing function returns "
                             << expectedTy;

    return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present.  This can check for preconditions of the
// getNumArguments hook not failing.
LogicalResult P4HIR::FuncOp::verifyType() {
    auto type = getFunctionType();
    if (!isa<P4HIR::FuncType>(type))
        return emitOpError("requires '" + getFunctionTypeAttrName().str() +
                           "' attribute of function type");
    if (auto rt = type.getReturnTypes(); !rt.empty() && mlir::isa<P4HIR::VoidType>(rt.front()))
        return emitOpError(
            "The return type for a function returning void should "
            "be empty instead of an explicit !p4hir.void");

    return success();
}

LogicalResult P4HIR::FuncOp::verify() {
    // TODO: Check that all reference-typed arguments have direction indicated
    // TODO: Check that actions do have body
    return success();
}

void P4HIR::FuncOp::build(OpBuilder &builder, OperationState &result, llvm::StringRef name,
                          P4HIR::FuncType type, ArrayRef<NamedAttribute> attrs,
                          ArrayRef<DictionaryAttr> argAttrs) {
    result.addRegion();

    result.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getFunctionTypeAttrName(result.name), TypeAttr::get(type));
    result.attributes.append(attrs.begin(), attrs.end());
    // We default to private visibility
    result.addAttribute(SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));

    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name),
                                                  getResAttrsAttrName(result.name));
}

void P4HIR::FuncOp::createEntryBlock() {
    assert(empty() && "can only create entry block for empty function");
    Block &first = getFunctionBody().emplaceBlock();
    auto loc = getFunctionBody().getLoc();
    for (auto argType : getFunctionType().getInputs()) first.addArgument(argType, loc);
}

void P4HIR::FuncOp::print(OpAsmPrinter &p) {
    if (getAction()) p << " action";

    // Print function name, signature, and control.
    p << ' ';
    p.printSymbolName(getSymName());
    auto fnType = getFunctionType();
    function_interface_impl::printFunctionSignature(p, *this, fnType.getInputs(), false,
                                                    fnType.getReturnTypes());

    if (mlir::ArrayAttr annotations = getAnnotationsAttr()) {
        p << ' ';
        p.printAttribute(annotations);
    }

    function_interface_impl::printFunctionAttributes(
        p, *this,
        // These are all omitted since they are custom printed already.
        {getFunctionTypeAttrName(), SymbolTable::getVisibilityAttrName(), getArgAttrsAttrName(),
         getActionAttrName(), getResAttrsAttrName()});

    // Print the body if this is not an external function.
    Region &body = getOperation()->getRegion(0);
    if (!body.empty()) {
        p << ' ';
        p.printRegion(body, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
    }
}

ParseResult P4HIR::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
    llvm::SMLoc loc = parser.getCurrentLocation();
    auto &builder = parser.getBuilder();

    // Parse action marker
    auto actionNameAttr = getActionAttrName(state.name);
    bool isAction = false;
    if (::mlir::succeeded(parser.parseOptionalKeyword(actionNameAttr.strref()))) {
        isAction = true;
        state.addAttribute(actionNameAttr, parser.getBuilder().getUnitAttr());
    }

    // Parse the name as a symbol.
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
        return failure();

    // We default to private visibility
    state.addAttribute(SymbolTable::getVisibilityAttrName(), builder.getStringAttr("private"));

    llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
    llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignature(parser, /*allowVariadic=*/false, arguments,
                                                        isVariadic, resultTypes, resultAttrs))
        return failure();

    // Actions have no results
    if (isAction && !resultTypes.empty())
        return parser.emitError(loc, "actions should not produce any results");
    else if (resultTypes.size() > 1)
        return parser.emitError(loc, "functions only supports zero or one results");

    // Build the function type.
    for (auto &arg : arguments) argTypes.push_back(arg.type);

    // Fetch return type or set it to void if empty/ommited.
    mlir::Type returnType =
        (resultTypes.empty() ? P4HIR::VoidType::get(builder.getContext()) : resultTypes.front());

    if (auto fnType = P4HIR::FuncType::get(argTypes, returnType)) {
        state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(fnType));
    } else
        return failure();

    // Parse an OptionalAttr<ArrayAttr>:$annotations
    mlir::ArrayAttr annotations;
    if (auto oa = parser.parseOptionalAttribute(annotations); oa.has_value())
        state.addAttribute(getAnnotationsAttrName(state.name), annotations);

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(state.attributes)) return failure();

    // Add the attributes to the function arguments.
    assert(resultAttrs.size() == resultTypes.size());
    function_interface_impl::addArgAndResultAttrs(builder, state, arguments, resultAttrs,
                                                  getArgAttrsAttrName(state.name),
                                                  getResAttrsAttrName(state.name));

    // Parse the action body.
    auto *body = state.addRegion();
    if (OptionalParseResult parseResult =
            parser.parseOptionalRegion(*body, arguments, /*enableNameShadowing=*/false);
        parseResult.has_value()) {
        if (failed(*parseResult)) return failure();
        // Function body was parsed, make sure its not empty.
        if (body->empty()) return parser.emitError(loc, "expected non-empty function body");
    } else if (isAction) {
        parser.emitError(loc, "action shall have a body");
    }

    return success();
}

void P4HIR::CallOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    if (getResult()) setNameFn(getResult(), "call");
}

LogicalResult P4HIR::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr) return emitOpError("requires a 'callee' symbol reference attribute");
    FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue() << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getOperand(i).getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType()
                   << " for operand number " << i;

    // Actions must not return any results
    if (fn.getAction() && getNumResults() != 0)
        return emitOpError("incorrect number of results for action call");

    // Void function must not return any results.
    if (fnType.isVoid() && getNumResults() != 0)
        return emitOpError("callee returns void but call has results");

    // Non-void function calls must return exactly one result.
    if (!fnType.isVoid() && getNumResults() != 1)
        return emitOpError("incorrect number of results for callee");

    // Parent function and return value types must match.
    if (!fnType.isVoid() && getResultTypes().front() != fnType.getReturnType())
        return emitOpError("result type mismatch: expected ")
               << fnType.getReturnType() << ", but provided " << getResult().getType();

    return success();
}

//===----------------------------------------------------------------------===//
// StructOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::StructOp::parse(OpAsmParser &parser, OperationState &result) {
    llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    Type declType;

    if (parser.parseLParen() || parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(declType))
        return failure();

    auto structType = mlir::dyn_cast<StructType>(declType);
    if (!structType) return parser.emitError(parser.getNameLoc(), "expected !p4hir.struct type");

    llvm::SmallVector<Type, 4> structInnerTypes;
    structType.getInnerTypes(structInnerTypes);
    result.addTypes(structType);

    if (parser.resolveOperands(operands, structInnerTypes, inputOperandsLoc, result.operands))
        return failure();
    return success();
}

void P4HIR::StructOp::print(OpAsmPrinter &printer) {
    printer << " (";
    printer.printOperands(getInput());
    printer << ")";
    printer.printOptionalAttrDict((*this)->getAttrs());
    printer << " : " << getType();
}

LogicalResult P4HIR::StructOp::verify() {
    auto elements = mlir::cast<StructType>(getType()).getElements();

    if (elements.size() != getInput().size()) return emitOpError("struct field count mismatch");

    for (const auto &[field, value] : llvm::zip(elements, getInput()))
        if (field.type != value.getType())
            return emitOpError("struct field `") << field.name << "` type does not match";

    return success();
}

void P4HIR::StructOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<32> name("struct_");
    name += getType().getName();
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Ensure an aggregate op's field index is within the bounds of
/// the aggregate type and the accessed field is of 'elementType'.
template <typename AggregateOp, typename AggregateType>
static LogicalResult verifyAggregateFieldIndexAndType(AggregateOp &op, AggregateType aggType,
                                                      Type elementType) {
    auto index = op.getFieldIndex();
    if (index >= aggType.getElements().size())
        return op.emitOpError() << "field index " << index
                                << " exceeds element count of aggregate type";

    if (elementType != aggType.getElements()[index].type)
        return op.emitOpError() << "type " << aggType.getElements()[index].type
                                << " of accessed field in aggregate at index " << index
                                << " does not match expected type " << elementType;

    return success();
}

LogicalResult P4HIR::StructExtractOp::verify() {
    return verifyAggregateFieldIndexAndType<StructExtractOp, StructType>(
        *this, getInput().getType(), getType());
}

template <typename AggregateType>
static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    StringAttr fieldName;
    AggregateType declType;

    if (parser.parseOperand(operand) || parser.parseLSquare() || parser.parseAttribute(fieldName) ||
        parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseCustomTypeWithFallback<AggregateType>(declType))
        return failure();

    auto fieldIndex = declType.getFieldIndex(fieldName);
    if (!fieldIndex) {
        parser.emitError(parser.getNameLoc(),
                         "field name '" + fieldName.getValue() + "' not found in aggregate type");
        return failure();
    }

    auto indexAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), *fieldIndex);
    result.addAttribute("fieldIndex", indexAttr);
    Type resultType = declType.getElements()[*fieldIndex].type;
    result.addTypes(resultType);

    if (parser.resolveOperand(operand, declType, result.operands)) return failure();
    return success();
}

template <typename AggregateType>
static ParseResult parseExtractRefOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    StringAttr fieldName;
    P4HIR::ReferenceType declType;

    if (parser.parseOperand(operand) || parser.parseLSquare() || parser.parseAttribute(fieldName) ||
        parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseCustomTypeWithFallback<P4HIR::ReferenceType>(declType))
        return failure();

    auto aggType = mlir::dyn_cast<AggregateType>(declType.getObjectType());
    if (!aggType) {
        parser.emitError(parser.getNameLoc(), "expected reference to aggregate type");
        return failure();
    }
    auto fieldIndex = aggType.getFieldIndex(fieldName);
    if (!fieldIndex) {
        parser.emitError(parser.getNameLoc(),
                         "field name '" + fieldName.getValue() + "' not found in aggregate type");
        return failure();
    }

    auto indexAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), *fieldIndex);
    result.addAttribute("fieldIndex", indexAttr);
    Type resultType = P4HIR::ReferenceType::get(aggType.getElements()[*fieldIndex].type);
    result.addTypes(resultType);

    if (parser.resolveOperand(operand, declType, result.operands)) return failure();
    return success();
}

/// Use the same printer for both struct_extract and struct_extract_ref since the
/// syntax is identical.
template <typename AggType>
static void printExtractOp(OpAsmPrinter &printer, AggType op) {
    printer << " ";
    printer.printOperand(op.getInput());
    printer << "[\"" << op.getFieldName() << "\"]";
    printer.printOptionalAttrDict(op->getAttrs(), {"fieldIndex"});
    printer << " : ";
    auto type = op.getInput().getType();
    if (auto validType = mlir::dyn_cast<P4HIR::ReferenceType>(type))
        printer.printStrippedAttrOrType(validType);
    else
        printer << type;
}

ParseResult P4HIR::StructExtractOp::parse(OpAsmParser &parser, OperationState &result) {
    return parseExtractOp<StructType>(parser, result);
}

void P4HIR::StructExtractOp::print(OpAsmPrinter &printer) { printExtractOp(printer, *this); }

void P4HIR::StructExtractOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                   StructType::FieldInfo field) {
    auto fieldIndex = mlir::cast<StructType>(input.getType()).getFieldIndex(field.name);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, field.type, input, *fieldIndex);
}

void P4HIR::StructExtractOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                   StringAttr fieldName) {
    auto structType = mlir::cast<StructType>(input.getType());
    auto fieldIndex = structType.getFieldIndex(fieldName);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    auto resultType = structType.getElements()[*fieldIndex].type;
    build(builder, odsState, resultType, input, *fieldIndex);
}

void P4HIR::StructExtractOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), getFieldName());
}

void P4HIR::StructExtractRefOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), getFieldName());
}

ParseResult P4HIR::StructExtractRefOp::parse(OpAsmParser &parser, OperationState &result) {
    return parseExtractRefOp<StructType>(parser, result);
}

void P4HIR::StructExtractRefOp::print(OpAsmPrinter &printer) { printExtractOp(printer, *this); }

LogicalResult P4HIR::StructExtractRefOp::verify() {
    auto type =
        mlir::cast<StructType>(mlir::cast<ReferenceType>(getInput().getType()).getObjectType());
    return verifyAggregateFieldIndexAndType<StructExtractRefOp, StructType>(
        *this, type, getType().getObjectType());
}

void P4HIR::StructExtractRefOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                      StructType::FieldInfo field) {
    auto structType =
        mlir::cast<StructType>(mlir::cast<ReferenceType>(input.getType()).getObjectType());
    auto fieldIndex = structType.getFieldIndex(field.name);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, ReferenceType::get(field.type), input, *fieldIndex);
}

void P4HIR::StructExtractRefOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                      StringAttr fieldName) {
    auto structType =
        mlir::cast<StructType>(mlir::cast<ReferenceType>(input.getType()).getObjectType());
    auto fieldIndex = structType.getFieldIndex(fieldName);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    auto resultType = ReferenceType::get(structType.getElements()[*fieldIndex].type);
    build(builder, odsState, resultType, input, *fieldIndex);
}

namespace {
struct P4HIROpAsmDialectInterface : public OpAsmDialectInterface {
    using OpAsmDialectInterface::OpAsmDialectInterface;

    AliasResult getAlias(Type type, raw_ostream &os) const final {
        if (auto infintType = mlir::dyn_cast<P4HIR::InfIntType>(type)) {
            os << infintType.getAlias();
            return AliasResult::OverridableAlias;
        }

        if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(type)) {
            os << bitsType.getAlias();
            return AliasResult::OverridableAlias;
        }

        if (auto voidType = mlir::dyn_cast<P4HIR::VoidType>(type)) {
            os << voidType.getAlias();
            return AliasResult::OverridableAlias;
        }

        if (auto structType = mlir::dyn_cast<P4HIR::StructType>(type)) {
            os << structType.getName();
            return AliasResult::OverridableAlias;
        }

        return AliasResult::NoAlias;
    }

    AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
        if (auto boolAttr = mlir::dyn_cast<P4HIR::BoolAttr>(attr)) {
            os << (boolAttr.getValue() ? "true" : "false");
            return AliasResult::FinalAlias;
        }

        if (auto intAttr = mlir::dyn_cast<P4HIR::IntAttr>(attr)) {
            os << "int" << intAttr.getValue();
            if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(intAttr.getType()))
                os << "_" << bitsType.getAlias();
            else if (auto infintType = mlir::dyn_cast<P4HIR::InfIntType>(intAttr.getType()))
                os << "_" << infintType.getAlias();

            return AliasResult::FinalAlias;
        }

        if (auto dirAttr = mlir::dyn_cast<P4HIR::ParamDirectionAttr>(attr)) {
            os << stringifyEnum(dirAttr.getValue());
            return AliasResult::FinalAlias;
        }

        return AliasResult::NoAlias;
    }
};
}  // namespace

void P4HIR::P4HIRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
        >();
    addInterfaces<P4HIROpAsmDialectInterface>();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.cpp.inc"
