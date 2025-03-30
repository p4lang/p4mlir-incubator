#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
    if (mlir::isa<P4HIR::BoolAttr>(attrType)) {
        if (auto aliasedType = mlir::dyn_cast<P4HIR::AliasType>(opType))
            opType = aliasedType.getAliasedType();
        if (!mlir::isa<P4HIR::BoolType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.bool' for '" << attrType << "'";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr>(attrType)) {
        if (auto aliasedType = mlir::dyn_cast<P4HIR::AliasType>(opType))
            opType = aliasedType.getAliasedType();
        if (!mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") does not match value type (" << attrType << ")";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr, P4HIR::BoolAttr>(attrType)) return success();

    if (mlir::isa<P4HIR::AggAttr>(attrType)) {
        if (!mlir::isa<P4HIR::StructType, P4HIR::HeaderType, P4HIR::HeaderUnionType,
                       mlir::TupleType>(opType))
            return op->emitOpError("result type (") << opType << ") is not an aggregate type";

        return success();
    }

    if (mlir::isa<P4HIR::EnumFieldAttr>(attrType)) {
        if (!mlir::isa<P4HIR::EnumType, P4HIR::SerEnumType>(opType))
            return op->emitOpError("result type (") << opType << ") is not an enum type";

        return success();
    }

    if (mlir::isa<P4HIR::ErrorCodeAttr>(attrType)) {
        if (!mlir::isa<P4HIR::ErrorType>(opType))
            return op->emitOpError("result type (") << opType << ") is not an error type";

        return success();
    }

    if (mlir::isa<P4HIR::ValidityBitAttr>(attrType)) {
        if (!mlir::isa<P4HIR::ValidBitType>(opType))
            return op->emitOpError("result type (") << opType << ") is not a validity bit type";

        return success();
    }

    if (mlir::isa<P4HIR::CtorParamAttr>(attrType)) {
        // We should be fine here
        return success();
    }

    if (mlir::isa<mlir::StringAttr>(attrType)) {
        if (!mlir::isa<P4HIR::StringType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.string' for '" << attrType << "'";
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
    } else if (auto validityCst = mlir::dyn_cast<P4HIR::ValidityBitAttr>(getValue())) {
        setNameFn(getResult(), stringifyEnum(validityCst.getValue()));
    } else if (auto errorCst = mlir::dyn_cast<P4HIR::ErrorCodeAttr>(getValue())) {
        llvm::SmallString<32> error("error_");
        error += errorCst.getField().getValue();
        setNameFn(getResult(), error);
    } else if (auto enumCst = mlir::dyn_cast<P4HIR::EnumFieldAttr>(getValue())) {
        llvm::SmallString<32> specialNameBuffer;
        llvm::raw_svector_ostream specialName(specialNameBuffer);
        if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(enumCst.getType()))
            specialName << enumType.getName() << '_' << enumCst.getField().getValue();
        else {
            specialName << mlir::cast<P4HIR::SerEnumType>(enumCst.getType()).getName() << '_'
                        << enumCst.getField().getValue();
        }

        setNameFn(getResult(), specialName.str());
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
// ShrOp, ShlOp
//===----------------------------------------------------------------------===//

void P4HIR::ShlOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "shl");
}

void P4HIR::ShrOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "shr");
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
// ShlOp & ShrOp
//===----------------------------------------------------------------------===//

LogicalResult verifyArithmeticShiftOperation(Operation *op, Type rhsType) {
    if (auto rhsBitsType = dyn_cast<P4HIR::BitsType>(rhsType)) {
        if (rhsBitsType.isSigned()) {
            return op->emitOpError()
                   << "the right-hand side operand of an arithmetic shift must be unsigned";
        }
    }
    return success();
}

LogicalResult P4HIR::ShlOp::verify() {
    auto rhsType = getOperand(1).getType();
    return verifyArithmeticShiftOperation(getOperation(), rhsType);
}

LogicalResult P4HIR::ShrOp::verify() {
    auto rhsType = getOperand(1).getType();
    return verifyArithmeticShiftOperation(getOperation(), rhsType);
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
                           mlir::DictionaryAttr annotations,
                           function_ref<void(OpBuilder &, Type &, Location)> scopeBuilder) {
    assert(scopeBuilder && "the builder callback for 'then' must be present");

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);
    Region *scopeRegion = result.addRegion();
    builder.createBlock(scopeRegion);

    mlir::Type yieldTy;
    scopeBuilder(builder, yieldTy, result.location);

    if (yieldTy) result.addTypes(TypeRange{yieldTy});
}

void P4HIR::ScopeOp::build(OpBuilder &builder, OperationState &result,
                           mlir::DictionaryAttr annotations,
                           function_ref<void(OpBuilder &, Location)> scopeBuilder) {
    assert(scopeBuilder && "the builder callback for 'then' must be present");

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

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

    // Parse annotations
    mlir::DictionaryAttr thenAnnotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(thenAnnotations)) return failure();
        result.addAttribute(getThenAnnotationsAttrName(result.name), thenAnnotations);
    }

    // Parse the 'then' region.
    auto parseThenLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                           /*argTypes=*/{}))
        return failure();
    if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed()) return failure();

    // If we find an 'else' keyword, parse the 'else' region.
    if (!parser.parseOptionalKeyword("else")) {
        auto parseElseLoc = parser.getCurrentLocation();

        // Parse annotations
        mlir::DictionaryAttr elseAnnotations;
        if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
            if (parser.parseAttribute<mlir::DictionaryAttr>(elseAnnotations)) return failure();
            result.addAttribute(getElseAnnotationsAttrName(result.name), elseAnnotations);
        }

        if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{})) return failure();
        if (ensureRegionTerm(parser, *elseRegion, parseElseLoc).failed()) return failure();
    }

    // Parse the optional attribute list.
    return parser.parseOptionalAttrDict(result.attributes) ? failure() : success();
}

void P4HIR::IfOp::print(OpAsmPrinter &p) {
    p << " " << getCondition();
    if (auto ann = getThenAnnotations(); ann && !ann->empty()) {
        p << " annotations ";
        p.printAttributeWithoutType(*ann);
    }
    p << ' ';
    auto &thenRegion = this->getThenRegion();
    p.printRegion(thenRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

    // Print the 'else' regions if it exists and has a block.
    auto &elseRegion = this->getElseRegion();
    if (!elseRegion.empty()) {
        p << " else";
        if (auto ann = getElseAnnotations(); ann && !ann->empty()) {
            p << " annotations ";
            p.printAttributeWithoutType(*ann);
        }
        p << ' ';
        p.printRegion(elseRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(elseRegion));
    }

    p.printOptionalAttrDict(getOperation()->getAttrs(),
                            {getThenAnnotationsAttrName(), getElseAnnotationsAttrName()});
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
                        mlir::DictionaryAttr thenAnnotations,
                        function_ref<void(OpBuilder &, Location)> elseBuilder,
                        mlir::DictionaryAttr elseAnnotations) {
    assert(thenBuilder && "the builder callback for 'then' must be present");

    result.addOperands(cond);
    if (thenAnnotations && !thenAnnotations.empty())
        result.addAttribute(getThenAnnotationsAttrName(result.name), thenAnnotations);

    OpBuilder::InsertionGuard guard(builder);
    Region *thenRegion = result.addRegion();
    builder.createBlock(thenRegion);
    thenBuilder(builder, result.location);

    Region *elseRegion = result.addRegion();
    if (!withElseRegion) return;

    if (elseAnnotations && !elseAnnotations.empty())
        result.addAttribute(getElseAnnotationsAttrName(result.name), elseAnnotations);

    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
}

mlir::LogicalResult P4HIR::ReturnOp::verify() {
    // Returns can be present in multiple different scopes, get the wrapping
    // function and start from there.
    auto fnOp = getOperation()->getParentOfType<FunctionOpInterface>();
    if (!fnOp || !mlir::isa<P4HIR::FuncOp, P4HIR::ControlOp>(fnOp)) {
        return emitOpError() << "returns are only possible from function-like objects: functions, "
                                "actions and control apply blocks";
    }

    // ReturnOps currently only have a single optional operand.
    if (getNumOperands() > 1) return emitOpError() << "expects at most 1 return operand";

    // Ensure returned type matches the function signature.
    auto expectedTy = mlir::cast<P4HIR::FuncType>(fnOp.getFunctionType()).getReturnType();
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
                          P4HIR::FuncType type, bool isExternal, ArrayRef<DictionaryAttr> argAttrs,
                          mlir::DictionaryAttr annotations) {
    result.addRegion();

    result.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getFunctionTypeAttrName(result.name), TypeAttr::get(type));
    // External functions are private, everything else is public
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        builder.getStringAttr(isExternal ? "private" : "public"));
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name), {});
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
    auto typeArguments = fnType.getTypeArguments();
    if (!typeArguments.empty()) {
        p << '<';
        llvm::interleaveComma(typeArguments, p, [&p](mlir::Type type) { p.printType(type); });
        p << '>';
    }

    function_interface_impl::printFunctionSignature(p, *this, fnType.getInputs(), false,
                                                    fnType.getReturnTypes());

    function_interface_impl::printFunctionAttributes(
        p, *this,
        // These are all omitted since they are custom printed already.
        {getFunctionTypeAttrName(), SymbolTable::getVisibilityAttrName(), getArgAttrsAttrName(),
         getActionAttrName(), getAnnotationsAttrName()});

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        p << " annotations ";
        p.printAttributeWithoutType(*ann);
    }

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

    // Try to parse type arguments if any
    llvm::SmallVector<mlir::Type, 1> typeArguments;
    if (succeeded(parser.parseOptionalLess())) {
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                mlir::Type type;
                if (parser.parseType(type)) return mlir::failure();
                typeArguments.push_back(type);
                return mlir::success();
            }) ||
            parser.parseGreater())
            return failure();
    }

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

    if (auto fnType = P4HIR::FuncType::get(argTypes, returnType, typeArguments)) {
        state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(fnType));
    } else
        return failure();

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(state.attributes)) return failure();

    // Add the attributes to the function arguments.
    assert(resultAttrs.size() == resultTypes.size());
    function_interface_impl::addArgAndResultAttrs(builder, state, arguments, resultAttrs,
                                                  getArgAttrsAttrName(state.name), {});

    // Parse annotations
    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        state.addAttribute(getAnnotationsAttrName(state.name), annotations);
    }

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

    // All functions are public except declarations
    state.addAttribute(SymbolTable::getVisibilityAttrName(),
                       builder.getStringAttr(body->empty() ? "private" : "public"));

    return success();
}

void P4HIR::CallOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    if (getResult()) setNameFn(getResult(), "call");
}

static mlir::ModuleOp getParentModule(Operation *from) {
    if (auto moduleOp = from->getParentOfType<mlir::ModuleOp>()) return moduleOp;

    from->emitOpError("could not find parent module op");
    return nullptr;
}

static mlir::Type substituteType(mlir::Type type, mlir::TypeRange calleeTypeArgs,
                                 std::optional<mlir::ArrayAttr> typeOperands) {
    if (auto typeVar = llvm::dyn_cast<P4HIR::TypeVarType>(type)) {
        size_t pos = llvm::find(calleeTypeArgs, typeVar) - calleeTypeArgs.begin();
        if (pos == calleeTypeArgs.size()) return {};
        return llvm::cast<mlir::TypeAttr>(typeOperands->getValue()[pos]).getValue();
    } else if (auto refType = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
        return P4HIR::ReferenceType::get(
            substituteType(refType.getObjectType(), calleeTypeArgs, typeOperands));
    } else if (auto tupleType = llvm::dyn_cast<mlir::TupleType>(type)) {
        llvm::SmallVector<mlir::Type> substituted;
        for (auto elTy : tupleType.getTypes())
            substituted.push_back(substituteType(elTy, calleeTypeArgs, typeOperands));
        return mlir::TupleType::get(type.getContext(), substituted);
    }

    return type;
};

LogicalResult P4HIR::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto sym = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!sym) return emitOpError("requires a 'callee' symbol reference attribute");

    // Callee might be:
    //  - Overload set, then we need to look for a particular overload
    //  - Normal functions. They are defined at top-level only. Top-level actions are also here.
    //  - Actions defined at control level. Check for them first.
    P4HIR::FuncOp fn;
    if (auto controlOp = (*this)->getParentOfType<P4HIR::ControlOp>()) {
        fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(controlOp, sym);
        if (fn && !fn.getAction())
            return emitOpError() << "'" << sym << "' does not reference a valid action";
    }

    if (!fn) {
        if (auto *decl = symbolTable.lookupNearestSymbolFrom(getParentModule(*this), sym)) {
            if ((fn = llvm::dyn_cast<P4HIR::FuncOp>(decl))) {
                // We good here
            } else if (auto ovl = llvm::dyn_cast<P4HIR::OverloadSetOp>(decl)) {
                // Find the FuncOp with the correct # of operands
                for (Operation &nestedOp : ovl.getBody().front()) {
                    auto f = llvm::cast<FuncOp>(nestedOp);
                    if (f.getNumArguments() == getNumOperands()) {
                        fn = f;
                        break;
                    }
                }
                if (!fn) return emitOpError() << "'" << sym << "' failed to resolve overload set";
            } else
                return emitOpError() << "'" << sym << "' does not reference a valid function";
        }
    }

    if (!fn) return emitOpError() << "'" << sym << "' does not reference a valid function";

    auto fnType = fn.getFunctionType();
    // Verify that the operand and result types match the callee.
    if (fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    auto calleeTypeArgs = fnType.getTypeArguments();
    auto typeOperands = getTypeOperands();
    if (!calleeTypeArgs.empty()) {
        if (!typeOperands)
            return emitOpError("expected type operands to be specifid for generic callee type");
        if (calleeTypeArgs.size() != getTypeOperands()->size())
            return emitOpError("incorrect number of type operands for callee");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
        mlir::Type expectedType = substituteType(fnType.getInput(i), calleeTypeArgs, typeOperands);
        if (!expectedType)
            return emitOpError("cannot resolve type operand for operand number ") << i;
        mlir::Type providedType = getOperand(i).getType();
        if (providedType != expectedType)
            return emitOpError("operand type mismatch: expected operand type ")
                   << expectedType << ", but provided " << providedType << " for operand number "
                   << i;
    }

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
    if (!fnType.isVoid() &&
        getResultTypes().front() !=
            substituteType(fnType.getReturnType(), calleeTypeArgs, typeOperands))
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

    auto structType = mlir::dyn_cast<StructLikeTypeInterface>(declType);
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
    auto elements = mlir::cast<StructLikeTypeInterface>(getType()).getFields();

    if (elements.size() != getInput().size()) return emitOpError("struct field count mismatch");

    for (const auto &[field, value] : llvm::zip(elements, getInput()))
        if (field.type != value.getType())
            return emitOpError("struct field `") << field.name << "` type does not match";

    return success();
}

void P4HIR::StructOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<32> name;
    if (auto structType = mlir::dyn_cast<StructType>(getType())) {
        name += "struct_";
        name += structType.getName();
    } else if (auto headerType = mlir::dyn_cast<HeaderType>(getType())) {
        name += "hdr_";
        name += headerType.getName();
    } else if (auto headerUnionType = mlir::dyn_cast<HeaderUnionType>(getType())) {
        name += "hdru_";
        name += headerUnionType.getName();
    }

    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Ensure an aggregate op's field index is within the bounds of
/// the aggregate type and the accessed field is of 'elementType'.
template <typename AggregateOp>
static LogicalResult verifyAggregateFieldIndexAndType(AggregateOp &op,
                                                      P4HIR::StructLikeTypeInterface aggType,
                                                      Type elementType) {
    auto index = op.getFieldIndex();
    auto fields = aggType.getFields();
    if (index >= fields.size())
        return op.emitOpError() << "field index " << index
                                << " exceeds element count of aggregate type";

    if (elementType != fields[index].type)
        return op.emitOpError() << "type " << fields[index].type
                                << " of accessed field in aggregate at index " << index
                                << " does not match expected type " << elementType;

    return success();
}

LogicalResult P4HIR::StructExtractOp::verify() {
    return verifyAggregateFieldIndexAndType(
        *this, mlir::cast<StructLikeTypeInterface>(getInput().getType()), getType());
}

static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    StringAttr fieldName;
    mlir::Type declType;

    if (parser.parseOperand(operand) || parser.parseLSquare() || parser.parseAttribute(fieldName) ||
        parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseCustomTypeWithFallback(declType))
        return failure();

    auto aggType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(declType);
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
    Type resultType = aggType.getFields()[*fieldIndex].type;
    result.addTypes(resultType);

    if (parser.resolveOperand(operand, declType, result.operands)) return failure();
    return success();
}

static ParseResult parseExtractRefOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    StringAttr fieldName;
    P4HIR::ReferenceType declType;

    if (parser.parseOperand(operand) || parser.parseLSquare() || parser.parseAttribute(fieldName) ||
        parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseCustomTypeWithFallback<P4HIR::ReferenceType>(declType))
        return failure();

    auto aggType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(declType.getObjectType());
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
    Type resultType = P4HIR::ReferenceType::get(aggType.getFields()[*fieldIndex].type);
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
    return parseExtractOp(parser, result);
}

void P4HIR::StructExtractOp::print(OpAsmPrinter &printer) { printExtractOp(printer, *this); }

void P4HIR::StructExtractOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                   P4HIR::FieldInfo field) {
    auto structType = mlir::cast<P4HIR::StructLikeTypeInterface>(input.getType());
    auto fieldIndex = structType.getFieldIndex(field.name);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, field.type, input, *fieldIndex);
}

void P4HIR::StructExtractOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                   StringAttr fieldName) {
    auto structType = mlir::cast<P4HIR::StructLikeTypeInterface>(input.getType());
    auto fieldIndex = structType.getFieldIndex(fieldName);
    auto fieldType = structType.getFieldType(fieldName);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, fieldType, input, *fieldIndex);
}

void P4HIR::StructExtractOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), getFieldName());
}

void P4HIR::StructExtractRefOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<16> name = getFieldName();
    name += "_field_ref";
    setNameFn(getResult(), name);
}

ParseResult P4HIR::StructExtractRefOp::parse(OpAsmParser &parser, OperationState &result) {
    return parseExtractRefOp(parser, result);
}

void P4HIR::StructExtractRefOp::print(OpAsmPrinter &printer) { printExtractOp(printer, *this); }

LogicalResult P4HIR::StructExtractRefOp::verify() {
    auto type = mlir::cast<StructLikeTypeInterface>(
        mlir::cast<ReferenceType>(getInput().getType()).getObjectType());
    return verifyAggregateFieldIndexAndType(*this, type, getType().getObjectType());
}

void P4HIR::StructExtractRefOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                      P4HIR::FieldInfo field) {
    auto structLikeType = mlir::cast<ReferenceType>(input.getType()).getObjectType();
    auto structType = mlir::cast<P4HIR::StructLikeTypeInterface>(structLikeType);
    auto fieldIndex = structType.getFieldIndex(field.name);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, ReferenceType::get(field.type), input, *fieldIndex);
}

void P4HIR::StructExtractRefOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                      StringAttr fieldName) {
    auto structLikeType = mlir::cast<ReferenceType>(input.getType()).getObjectType();
    auto structType = mlir::cast<P4HIR::StructLikeTypeInterface>(structLikeType);
    auto fieldIndex = structType.getFieldIndex(fieldName);
    auto fieldType = structType.getFieldType(fieldName);
    assert(fieldIndex.has_value() && "field name not found in aggregate type");
    build(builder, odsState, ReferenceType::get(fieldType), input, *fieldIndex);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::TupleOp::parse(OpAsmParser &parser, OperationState &result) {
    llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    Type declType;

    if (parser.parseLParen() || parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(declType))
        return failure();

    auto tupleType = mlir::dyn_cast<mlir::TupleType>(declType);
    if (!tupleType) return parser.emitError(parser.getNameLoc(), "expected !tuple type");

    result.addTypes(tupleType);
    if (parser.resolveOperands(operands, tupleType.getTypes(), inputOperandsLoc, result.operands))
        return failure();
    return success();
}

void P4HIR::TupleOp::print(OpAsmPrinter &printer) {
    printer << " (";
    printer.printOperands(getInput());
    printer << ")";
    printer.printOptionalAttrDict((*this)->getAttrs());
    printer << " : " << getType();
}

LogicalResult P4HIR::TupleOp::verify() {
    auto elementTypes = getType().getTypes();

    if (elementTypes.size() != getInput().size()) return emitOpError("tuple field count mismatch");

    for (const auto &[field, value] : llvm::zip(elementTypes, getInput()))
        if (field != value.getType()) return emitOpError("tuple field types do not match");

    return success();
}

void P4HIR::TupleOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "tuple");
}

// TODO: This duplicates lots of things above for structs. Find a way to generalize
LogicalResult P4HIR::TupleExtractOp::verify() {
    auto index = getFieldIndex();
    auto fields = getInput().getType();
    if (index >= fields.size())
        return emitOpError() << "field index " << index
                             << " exceeds element count of aggregate type";

    if (getType() != fields.getType(index))
        return emitOpError() << "type " << fields.getType(index)
                             << " of accessed field in aggregate at index " << index
                             << " does not match expected type " << getType();

    return success();
}

ParseResult P4HIR::TupleExtractOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    unsigned fieldIndex = -1U;
    mlir::TupleType declType;

    if (parser.parseOperand(operand) || parser.parseLSquare() || parser.parseInteger(fieldIndex) ||
        parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType<mlir::TupleType>(declType))
        return failure();

    auto indexAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), fieldIndex);
    result.addAttribute("fieldIndex", indexAttr);
    Type resultType = declType.getType(fieldIndex);
    result.addTypes(resultType);

    if (parser.resolveOperand(operand, declType, result.operands)) return failure();
    return success();
}

void P4HIR::TupleExtractOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printOperand(getInput());
    printer << "[" << getFieldIndex() << "]";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"fieldIndex"});
    printer << " : ";
    printer << getInput().getType();
}

void P4HIR::TupleExtractOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<16> name;
    llvm::raw_svector_ostream specialName(name);
    specialName << 't' << getFieldIndex();

    setNameFn(getResult(), name);
}

void P4HIR::TupleExtractOp::build(OpBuilder &builder, OperationState &odsState, Value input,
                                  unsigned fieldIndex) {
    auto tupleType = mlir::cast<mlir::TupleType>(input.getType());
    build(builder, odsState, tupleType.getType(fieldIndex), input, fieldIndex);
}

//===----------------------------------------------------------------------===//
// SliceOp, SliceRefOp
//===----------------------------------------------------------------------===//

void P4HIR::SliceOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<16> name;
    llvm::raw_svector_ostream specialName(name);
    specialName << 's' << getHighBit() << "_" << getLowBit();

    setNameFn(getResult(), name);
}

void P4HIR::SliceRefOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    llvm::SmallString<16> name;
    llvm::raw_svector_ostream specialName(name);
    specialName << 's' << getHighBit() << "_" << getLowBit();

    setNameFn(getResult(), name);
}

LogicalResult P4HIR::SliceOp::verify() {
    auto resultType = getResult().getType();
    auto sourceType = getInput().getType();
    if (resultType.isSigned()) return emitOpError() << "slice result type is always unsigned";

    if (getHighBit() < getLowBit()) return emitOpError() << "invalid slice indices";

    if (resultType.getWidth() != getHighBit() - getLowBit() + 1)
        return emitOpError() << "slice result type does not match extraction width";

    if (auto bitsType = llvm::dyn_cast<P4HIR::BitsType>(sourceType)) {
        if (bitsType.getWidth() <= getHighBit())
            return emitOpError() << "extraction indices out of bound";
    }

    return success();
}

LogicalResult P4HIR::SliceRefOp::verify() {
    auto resultType = getResult().getType();
    auto sourceType = llvm::cast<P4HIR::ReferenceType>(getInput().getType()).getObjectType();
    if (resultType.isSigned()) return emitOpError() << "slice result type is always unsigned";

    if (getHighBit() < getLowBit()) return emitOpError() << "invalid slice indices";

    if (resultType.getWidth() != getHighBit() - getLowBit() + 1)
        return emitOpError() << "slice result type does not match extraction width";

    if (auto bitsType = llvm::dyn_cast<P4HIR::BitsType>(sourceType)) {
        if (bitsType.getWidth() <= getHighBit())
            return emitOpError() << "extraction indices out of bound";
    }

    return success();
}

LogicalResult P4HIR::AssignSliceOp::verify() {
    auto sourceType = getValue().getType();
    auto resultType = llvm::cast<P4HIR::BitsType>(
        llvm::cast<P4HIR::ReferenceType>(getRef().getType()).getObjectType());
    if (sourceType.isSigned()) return emitOpError() << "slice result type is always unsigned";

    if (getHighBit() < getLowBit()) return emitOpError() << "invalid slice indices";

    if (sourceType.getWidth() != getHighBit() - getLowBit() + 1)
        return emitOpError() << "slice result type does not match slice width";

    if (resultType.getWidth() <= getHighBit())
        return emitOpError() << "slice insertion indices out of bound";

    return success();
}
//===----------------------------------------------------------------------===//
// ParserOp
//===----------------------------------------------------------------------===//

void P4HIR::ParserOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                            llvm::StringRef sym_name, P4HIR::FuncType applyType,
                            P4HIR::CtorType ctorType, ArrayRef<DictionaryAttr> argAttrs,
                            mlir::DictionaryAttr annotations) {
    result.addRegion();

    result.addAttribute(::SymbolTable::getSymbolAttrName(), builder.getStringAttr(sym_name));
    result.addAttribute(getApplyTypeAttrName(result.name), TypeAttr::get(applyType));
    result.addAttribute(getCtorTypeAttrName(result.name), TypeAttr::get(ctorType));

    // Parsers are top-level objects with public visibility
    result.addAttribute(::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("public"));

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name), {});
}

void P4HIR::ParserOp::createEntryBlock() {
    assert(empty() && "can only create entry block for empty parser");
    Block &first = getFunctionBody().emplaceBlock();
    auto loc = getFunctionBody().getLoc();
    for (auto argType : getFunctionType().getInputs()) first.addArgument(argType, loc);
}

void P4HIR::ParserOp::print(mlir::OpAsmPrinter &printer) {
    // This is essentially function_interface_impl::printFunctionOp, but we
    // always print body and we do not have result / argument attributes (for now)

    auto funcName = getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);

    function_interface_impl::printFunctionSignature(printer, *this, getApplyType().getInputs(),
                                                    false, {});

    printer << "(";
    llvm::interleaveComma(getCtorType().getInputs(), printer,
                          [&](std::pair<mlir::StringAttr, mlir::Type> namedType) {
                              printer << namedType.first.getValue() << ": ";
                              printer.printType(namedType.second);
                          });
    printer << ")";

    function_interface_impl::printFunctionAttributes(
        printer, *this,
        // These are all omitted since they are custom printed already.
        {getApplyTypeAttrName(), getCtorTypeAttrName(), ::SymbolTable::getVisibilityAttrName(),
         getAnnotationsAttrName(), getArgAttrsAttrName()});

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        printer << " annotations ";
        printer.printAttributeWithoutType(*ann);
    }

    printer << ' ';
    printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

mlir::ParseResult P4HIR::ParserOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // This is essentially function_interface_impl::parseFunctionOp, but we do not have
    // result / argument attributes (for now)
    llvm::SMLoc loc = parser.getCurrentLocation();
    auto &builder = parser.getBuilder();

    // Parse the name as a symbol.
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, ::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();

    // Parsers are visible from top-level
    result.addAttribute(::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("public"));

    llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
    llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignature(parser, /*allowVariadic=*/false, arguments,
                                                        isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Parsers have no results
    if (!resultTypes.empty())
        return parser.emitError(loc, "parsers should not produce any results");

    // Build the function type.
    for (auto &arg : arguments) argTypes.push_back(arg.type);

    if (auto fnType = P4HIR::FuncType::get(builder.getContext(), argTypes)) {
        result.addAttribute(getApplyTypeAttrName(result.name), TypeAttr::get(fnType));
    } else
        return mlir::failure();

    // Resonstruct the ctor type
    {
        llvm::SmallVector<std::pair<StringAttr, Type>> namedTypes;
        if (parser.parseLParen()) return mlir::failure();

        // `(` `)`
        if (failed(parser.parseOptionalRParen())) {
            if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                    std::string name;
                    mlir::Type type;
                    if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
                        parser.parseType(type))
                        return mlir::failure();
                    namedTypes.emplace_back(mlir::StringAttr::get(parser.getContext(), name), type);
                    return mlir::success();
                }))
                return mlir::failure();
            if (parser.parseRParen()) return mlir::failure();
        }

        auto ctorResultType = P4HIR::ParserType::get(parser.getContext(), nameAttr, argTypes);
        result.addAttribute(getCtorTypeAttrName(result.name),
                            TypeAttr::get(P4HIR::CtorType::get(namedTypes, ctorResultType)));
    }

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) return failure();

    // Parse annotations
    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    // Add the attributes to the function arguments.
    assert(resultAttrs.size() == resultTypes.size());
    function_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
                                                  getArgAttrsAttrName(result.name), {});

    // Parse the parser body.
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/false)) return mlir::failure();

    // Make sure its not empty.
    if (body->empty()) return parser.emitError(loc, "expected non-empty parser body");

    return mlir::success();
}

static mlir::LogicalResult verifyStateTarget(mlir::Operation *op, mlir::SymbolRefAttr stateName,
                                             mlir::SymbolTableCollection &symbolTable) {
    // We are using fully-qualified names to reference to parser states, this
    // allows not to rename states during inlining, so we need to lookup wrt top-level ModuleOp
    if (!symbolTable.lookupNearestSymbolFrom<P4HIR::ParserStateOp>(getParentModule(op), stateName))
        return op->emitOpError() << "'" << stateName << "' does not reference a valid state";

    return mlir::success();
}

mlir::LogicalResult P4HIR::ParserTransitionOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
    return verifyStateTarget(*this, getStateAttr(), symbolTable);
}

void P4HIR::ParserSelectCaseOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> keyBuilder,
    mlir::SymbolRefAttr nextState) {
    OpBuilder::InsertionGuard guard(builder);
    Region *keyRegion = result.addRegion();
    builder.createBlock(keyRegion);
    keyBuilder(builder, result.location);

    result.addAttribute("state", nextState);
}

mlir::LogicalResult P4HIR::ParserSelectCaseOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
    return verifyStateTarget(*this, getStateAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// SetOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::SetOp::parse(OpAsmParser &parser, OperationState &result) {
    llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    Type declType;

    if (parser.parseLParen() || parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(declType))
        return failure();

    auto setType = mlir::dyn_cast<P4HIR::SetType>(declType);
    if (!setType) return parser.emitError(parser.getNameLoc(), "expected !p4hir.set type");

    result.addTypes(setType);
    if (parser.resolveOperands(operands, setType.getElementType(), inputOperandsLoc,
                               result.operands))
        return failure();
    return success();
}

void P4HIR::SetOp::print(OpAsmPrinter &printer) {
    printer << " (";
    printer.printOperands(getInput());
    printer << ")";
    printer.printOptionalAttrDict((*this)->getAttrs());
    printer << " : " << getType();
}

LogicalResult P4HIR::SetOp::verify() {
    auto elementType = getType().getElementType();

    for (auto value : getInput())
        if (value.getType() != elementType) return emitOpError("set element types do not match");

    return success();
}

void P4HIR::SetOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "set");
}

void P4HIR::SetOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::ValueRange values) {
    result.addTypes(P4HIR::SetType::get(values.front().getType()));
    result.addOperands(values);
}

//===----------------------------------------------------------------------===//
// SetProductOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::SetProductOp::parse(OpAsmParser &parser, OperationState &result) {
    llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    Type declType;

    if (parser.parseLParen() || parser.parseOperandList(operands) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(declType))
        return failure();

    auto setType = mlir::dyn_cast<P4HIR::SetType>(declType);
    if (!setType) return parser.emitError(parser.getNameLoc(), "expected !p4hir.set type");
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(setType.getElementType());
    if (!tupleType) return parser.emitError(parser.getNameLoc(), "expected set of tuples");

    result.addTypes(setType);
    llvm::SmallVector<mlir::Type, 4> elements;
    for (auto elTy : tupleType.getTypes()) elements.push_back(P4HIR::SetType::get(elTy));

    if (parser.resolveOperands(operands, elements, inputOperandsLoc, result.operands))
        return failure();
    return success();
}

void P4HIR::SetProductOp::print(OpAsmPrinter &printer) {
    printer << " (";
    printer.printOperands(getInput());
    printer << ")";
    printer.printOptionalAttrDict((*this)->getAttrs());
    printer << " : " << getType();
}

LogicalResult P4HIR::SetProductOp::verify() {
    auto elementType = mlir::dyn_cast<mlir::TupleType>(getType().getElementType());
    if (!elementType) return emitError("expected set of tuple type result");

    if (elementType.size() != getInput().size())
        return emitError("type mismatch: result and operands have different lengths");

    for (const auto &[field, value] : llvm::zip(elementType.getTypes(), getInput()))
        if (field != mlir::cast<P4HIR::SetType>(value.getType()).getElementType())
            return emitOpError("set product operands do not match result type");

    return success();
}

void P4HIR::SetProductOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "setproduct");
}

void P4HIR::SetProductOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                                mlir::ValueRange values) {
    llvm::SmallVector<mlir::Type, 4> elements;
    for (auto elTy : values.getTypes())
        elements.push_back(mlir::cast<SetType>(elTy).getElementType());

    result.addTypes(P4HIR::SetType::get(builder.getTupleType(elements)));
    result.addOperands(values);
}

//===----------------------------------------------------------------------===//
// UniversalSetOp
//===----------------------------------------------------------------------===//

void P4HIR::UniversalSetOp::build(mlir::OpBuilder &builder, mlir::OperationState &result) {
    result.addTypes(P4HIR::SetType::get(P4HIR::DontcareType::get(builder.getContext())));
}

void P4HIR::UniversalSetOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "everything");
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

void P4HIR::RangeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "range");
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

void P4HIR::MaskOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "mask");
}

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//
ParseResult P4HIR::PackageOp::parse(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    // Parse the name as a symbol.
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name), result.attributes))
        return mlir::failure();

    llvm::SmallVector<Type, 0> typeArguments;
    if (succeeded(parser.parseOptionalLess())) {
        if (parser.parseCommaSeparatedList(
                OpAsmParser::Delimiter::Square,
                [&]() -> ParseResult {
                    P4HIR::TypeVarType type;

                    if (parser.parseCustomTypeWithFallback<P4HIR::TypeVarType>(type))
                        return mlir::failure();

                    typeArguments.push_back(type);
                    return mlir::success();
                }) ||
            parser.parseGreater())
            return mlir::failure();
        result.addAttribute(getTypeParametersAttrName(result.name),
                            builder.getTypeArrayAttr(typeArguments));
    }

    // Resonstruct the ctor type
    llvm::SmallVector<mlir::Attribute> argAttrs;
    bool noAttrs = true;
    {
        llvm::SmallVector<std::pair<StringAttr, Type>> namedTypes;
        if (parser.parseLParen()) return mlir::failure();

        // `(` `)`
        if (failed(parser.parseOptionalRParen())) {
            if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                    std::string name;
                    mlir::Type type;
                    mlir::NamedAttrList attrs;
                    if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
                        parser.parseType(type) || parser.parseOptionalAttrDict(attrs))
                        return mlir::failure();
                    namedTypes.emplace_back(mlir::StringAttr::get(parser.getContext(), name), type);
                    if (!attrs.empty()) noAttrs = false;
                    argAttrs.push_back(attrs.getDictionary(parser.getContext()));
                    return mlir::success();
                }) ||
                parser.parseRParen())
                return mlir::failure();
        }

        auto ctorResultType = P4HIR::PackageType::get(parser.getContext(), nameAttr, {});
        result.addAttribute(getCtorTypeAttrName(result.name),
                            TypeAttr::get(P4HIR::CtorType::get(namedTypes, ctorResultType)));
    }

    if (!noAttrs)
        result.addAttribute(getArgAttrsAttrName(result.name), builder.getArrayAttr(argAttrs));

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) return mlir::failure();

    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    return success();
}

void P4HIR::PackageOp::print(OpAsmPrinter &printer) {
    printer << ' ';
    printer.printSymbolName(getName());
    if (auto typeParams = getTypeParameters()) {
        printer << '<';
        printer << *typeParams;
        printer << '>';
    }
    printer << '(';

    auto argAttrs = getArgAttrsAttr();
    for (auto [i, namedType] : llvm::enumerate(getCtorType().getInputs())) {
        if (i > 0) printer << ", ";
        printer << namedType.first << " : ";
        printer.printType(namedType.second);
        if (argAttrs)
            printer.printOptionalAttrDict(llvm::cast<DictionaryAttr>(argAttrs[i]).getValue());
    }
    printer << ')';

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        printer << " annotations ";
        printer.printAttributeWithoutType(*ann);
    }
}

void P4HIR::PackageOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                             llvm::StringRef name, CtorType type,
                             llvm::ArrayRef<mlir::Type> type_parameters,
                             llvm::ArrayRef<mlir::DictionaryAttr> argAttrs,
                             mlir::DictionaryAttr annotations) {
    result.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getCtorTypeAttrName(result.name), TypeAttr::get(type));
    if (!type_parameters.empty())
        result.addAttribute(getTypeParametersAttrName(result.name),
                            builder.getTypeArrayAttr(type_parameters));
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name), {});
}

//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

void P4HIR::InstantiateOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getName());
}

LogicalResult P4HIR::InstantiateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto ctorAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!ctorAttr) return emitOpError("requires a 'callee' symbol reference attribute");

    auto getCtorType =
        [&](mlir::FlatSymbolRefAttr ctorAttr) -> std::pair<CtorType, mlir::Operation *> {
        if (ParserOp parser =
                symbolTable.lookupNearestSymbolFrom<ParserOp>(getParentModule(*this), ctorAttr)) {
            return {parser.getCtorType(), parser.getOperation()};
        } else if (ControlOp control = symbolTable.lookupNearestSymbolFrom<ControlOp>(
                       getParentModule(*this), ctorAttr)) {
            return {control.getCtorType(), control.getOperation()};
        } else if (ExternOp ext = symbolTable.lookupNearestSymbolFrom<ExternOp>(
                       getParentModule(*this), ctorAttr)) {
            // TBD
            return {};
        } else if (PackageOp pkg = symbolTable.lookupNearestSymbolFrom<PackageOp>(
                       getParentModule(*this), ctorAttr)) {
            return {pkg.getCtorType(), pkg.getOperation()};
        }

        return {};
    };

    // Verify that the operand and result types match the callee.
    auto [ctorType, definingOp] = getCtorType(ctorAttr);
    if (ctorType) {
        if (ctorType.getNumInputs() != getNumOperands())
            return emitOpError("incorrect number of operands for callee");

        for (unsigned i = 0, e = ctorType.getNumInputs(); i != e; ++i) {
            // Packages are a bit special and nasty: they could have mismatched
            // declaration and instantiation types as name of object is a part of type, e.g.:
            // control e();
            // package top(e _e);
            // top(c())
            // So we need to be a bit more relaxed here
            if (auto pkg = mlir::dyn_cast<PackageOp>(definingOp)) {
                // TBD: Check
            } else if (getOperand(i).getType() != ctorType.getInput(i))
                return emitOpError("operand type mismatch: expected operand type ")
                       << ctorType.getInput(i) << ", but provided " << getOperand(i).getType()
                       << " for operand number " << i;
        }

        // Object itself and return value types must match.
        if (auto pkg = mlir::dyn_cast<PackageOp>(definingOp)) {
            // TBD: Check
        } else if (getResult().getType() != ctorType.getReturnType())
            return emitOpError("result type mismatch: expected ")
                   << ctorType.getReturnType() << ", but provided " << getResult().getType();

        return mlir::success();
    }

    return mlir::success();

    // TBD: Handle extern ctors and turn empty ctors into error
    /* return emitOpError()
           << "'" << ctorAttr.getValue()
           << "' does not reference a valid P4 object (parser, extern, control or package)"; */
}

//===----------------------------------------------------------------------===//
// ExternOp
//===----------------------------------------------------------------------===//

mlir::Block &P4HIR::ExternOp::createEntryBlock() {
    assert(getBody().empty() && "can only create entry block for empty exern");
    return getBody().emplaceBlock();
}

//===----------------------------------------------------------------------===//
// CallMethodOp
//===----------------------------------------------------------------------===//
LogicalResult P4HIR::CallMethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // TBD
    return success();
}

//===----------------------------------------------------------------------===//
// OverloadSetOp
//===----------------------------------------------------------------------===//

mlir::Block &P4HIR::OverloadSetOp::createEntryBlock() {
    assert(getBody().empty() && "can only create entry block for empty overload block");
    return getBody().emplaceBlock();
}

//===----------------------------------------------------------------------===//
// ControlOp
//===----------------------------------------------------------------------===//

void P4HIR::ControlOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                             llvm::StringRef sym_name, P4HIR::FuncType applyType,
                             P4HIR::CtorType ctorType, ArrayRef<DictionaryAttr> argAttrs,
                             mlir::DictionaryAttr annotations) {
    result.addRegion();

    result.addAttribute(::SymbolTable::getSymbolAttrName(), builder.getStringAttr(sym_name));
    result.addAttribute(getApplyTypeAttrName(result.name), TypeAttr::get(applyType));
    result.addAttribute(getCtorTypeAttrName(result.name), TypeAttr::get(ctorType));

    // Controls are top-level objects with public visibility
    result.addAttribute(::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("public"));

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name), {});
}

void P4HIR::ControlOp::createEntryBlock() {
    assert(empty() && "can only create entry block for empty control");
    Block &first = getFunctionBody().emplaceBlock();
    auto loc = getFunctionBody().getLoc();
    for (auto argType : getFunctionType().getInputs()) first.addArgument(argType, loc);
}

void P4HIR::ControlOp::print(mlir::OpAsmPrinter &printer) {
    auto funcName = getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);

    // Print function signature
    function_interface_impl::printFunctionSignature(printer, *this, getApplyType().getInputs(),
                                                    false, {});

    // Print ctor parameters
    printer << "(";
    llvm::interleaveComma(getCtorType().getInputs(), printer,
                          [&](std::pair<mlir::StringAttr, mlir::Type> namedType) {
                              printer << namedType.first.getValue() << ": ";
                              printer.printType(namedType.second);
                          });
    printer << ")";

    function_interface_impl::printFunctionAttributes(
        printer, *this,
        // These are all omitted since they are custom printed already.
        {getApplyTypeAttrName(), getCtorTypeAttrName(), ::SymbolTable::getVisibilityAttrName(),
         getAnnotationsAttrName(), getArgAttrsAttrName()});

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        printer << " annotations ";
        printer.printAttributeWithoutType(*ann);
    }

    printer << ' ';
    printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

mlir::ParseResult P4HIR::ControlOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // This is essentially function_interface_impl::parseFunctionOp, but we do not have
    // result / argument attributes (for now)
    llvm::SMLoc loc = parser.getCurrentLocation();
    auto &builder = parser.getBuilder();

    // Parse the name as a symbol.
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, ::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();

    // Parsers are visible from top-level
    result.addAttribute(::SymbolTable::getVisibilityAttrName(), builder.getStringAttr("public"));

    llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
    llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignature(parser, /*allowVariadic=*/false, arguments,
                                                        isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Controls have no results
    if (!resultTypes.empty())
        return parser.emitError(loc, "controls should not produce any results");

    // Build the function type.
    for (auto &arg : arguments) argTypes.push_back(arg.type);

    if (auto fnType = P4HIR::FuncType::get(builder.getContext(), argTypes)) {
        result.addAttribute(getApplyTypeAttrName(result.name), TypeAttr::get(fnType));
    } else
        return mlir::failure();

    // Resonstruct the ctor type
    {
        llvm::SmallVector<std::pair<StringAttr, Type>> namedTypes;
        if (parser.parseLParen()) return mlir::failure();

        // `(` `)`
        if (failed(parser.parseOptionalRParen())) {
            if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                    std::string name;
                    mlir::Type type;
                    if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
                        parser.parseType(type))
                        return mlir::failure();
                    namedTypes.emplace_back(mlir::StringAttr::get(parser.getContext(), name), type);
                    return mlir::success();
                }) ||
                parser.parseRParen())
                return mlir::failure();
        }

        auto ctorResultType = P4HIR::ControlType::get(parser.getContext(), nameAttr, argTypes);
        result.addAttribute(getCtorTypeAttrName(result.name),
                            TypeAttr::get(P4HIR::CtorType::get(namedTypes, ctorResultType)));
    }

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) return failure();

    // Parse annotations
    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    // Add the attributes to the control arguments.
    assert(resultAttrs.size() == resultTypes.size());
    function_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
                                                  getArgAttrsAttrName(result.name), {});

    // Parse the control body.
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/false)) return mlir::failure();

    // Make sure its not empty.
    if (body->empty()) return parser.emitError(loc, "expected non-empty control body");

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableOp
//===----------------------------------------------------------------------===//

void P4HIR::TableOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, llvm::StringRef name,
    mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> entryBuilder) {
    result.addAttribute(getSymNameAttrName(result.name), builder.getStringAttr(name));

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);

    Region *entryRegion = result.addRegion();
    builder.createBlock(entryRegion);
    entryBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// TableApplyOp
//===----------------------------------------------------------------------===//
LogicalResult P4HIR::TableApplyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // TBD
    return success();
}

void P4HIR::TableApplyOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    llvm::SmallString<32> result(getCallee().getRootReference().getValue());
    result += "_apply_result";
    setNameFn(getResult(), result);
}

//===----------------------------------------------------------------------===//
// TableEntryOp
//===----------------------------------------------------------------------===//

void P4HIR::TableEntryOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::StringAttr name, bool isConst,
    mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Type &, mlir::Location)> entryBuilder) {
    OpBuilder::InsertionGuard guard(builder);

    Region *entryRegion = result.addRegion();
    builder.createBlock(entryRegion);
    mlir::Type yieldTy;
    entryBuilder(builder, yieldTy, result.location);

    if (isConst) result.addAttribute(getIsConstAttrName(result.name), builder.getUnitAttr());
    result.addAttribute(getNameAttrName(result.name), name);

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    if (yieldTy) result.addTypes(TypeRange{yieldTy});
}

void P4HIR::TableEntryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// TableActionsOp
//===----------------------------------------------------------------------===//

void P4HIR::TableActionsOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> entryBuilder) {
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);

    Region *entryRegion = result.addRegion();
    builder.createBlock(entryRegion);
    entryBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// TableDefaultActionOp
//===----------------------------------------------------------------------===//

void P4HIR::TableDefaultActionOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> entryBuilder) {
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);

    Region *entryRegion = result.addRegion();
    builder.createBlock(entryRegion);
    entryBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// TableSizeOp
//===----------------------------------------------------------------------===//
void P4HIR::TableSizeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "size");
}

//===----------------------------------------------------------------------===//
// TableKeyOp
//===----------------------------------------------------------------------===//

void P4HIR::TableKeyOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> keyBuilder) {
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);

    Region *entryRegion = result.addRegion();
    builder.createBlock(entryRegion);
    keyBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// TableActionOp
//===----------------------------------------------------------------------===//

void P4HIR::TableActionOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::SymbolRefAttr action,
    P4HIR::FuncType cplaneType, ArrayRef<mlir::DictionaryAttr> argAttrs,
    mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Block::BlockArgListType, mlir::Location)>
        entryBuilder) {
    result.addAttribute(getCplaneTypeAttrName(result.name), TypeAttr::get(cplaneType));
    result.addAttribute(getActionAttrName(result.name), action);

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                  /*resultAttrs=*/std::nullopt,
                                                  getArgAttrsAttrName(result.name), {});

    OpBuilder::InsertionGuard guard(builder);
    auto *body = result.addRegion();

    Block &first = body->emplaceBlock();
    for (auto argType : cplaneType.getInputs()) first.addArgument(argType, result.location);
    builder.setInsertionPointToStart(&first);
    entryBuilder(builder, first.getArguments(), result.location);
}

void P4HIR::TableActionOp::print(mlir::OpAsmPrinter &printer) {
    auto actName = getActionAttr();

    printer << " ";
    printer << actName;

    printer << '(';
    const auto argTypes = getCplaneType().getInputs();
    mlir::ArrayAttr argAttrs = getArgAttrsAttr();
    for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
        if (i > 0) printer << ", ";

        ArrayRef<NamedAttribute> attrs;
        if (argAttrs) attrs = llvm::cast<DictionaryAttr>(argAttrs[i]).getValue();
        printer.printRegionArgument(getBody().front().getArgument(i), attrs);
    }
    printer << ')';

    function_interface_impl::printFunctionAttributes(
        printer, *this,
        // These are all omitted since they are custom printed already.
        {getActionAttrName(), getCplaneTypeAttrName(), getAnnotationsAttrName(),
         getArgAttrsAttrName()});

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        printer << " annotations ";
        printer.printAttributeWithoutType(*ann);
    }

    printer << ' ';
    printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

mlir::ParseResult P4HIR::TableActionOp::parse(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
    // This is essentially function_interface_impl::parseFunctionOp, but we do not have
    // result / argument attributes (for now)
    llvm::SMLoc loc = parser.getCurrentLocation();
    auto &builder = parser.getBuilder();

    // Parse the name as a symbol.
    SymbolRefAttr actionAttr;
    if (parser.parseCustomAttributeWithFallback(actionAttr, builder.getType<::mlir::NoneType>(),
                                                getActionAttrName(result.name), result.attributes))
        return mlir::failure();

    llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
    llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignature(parser, /*allowVariadic=*/false, arguments,
                                                        isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Table actions have no results
    if (!resultTypes.empty())
        return parser.emitError(loc, "table actions should not produce any results");

    // Build the function type.
    for (auto &arg : arguments) argTypes.push_back(arg.type);

    if (auto fnType = P4HIR::FuncType::get(builder.getContext(), argTypes)) {
        result.addAttribute(getCplaneTypeAttrName(result.name), TypeAttr::get(fnType));
    } else
        return mlir::failure();

    // If additional attributes are present, parse them.
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) return failure();

    // Add the attributes to the function arguments.
    assert(resultAttrs.size() == resultTypes.size());
    function_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
                                                  getArgAttrsAttrName(result.name), {});

    // Parse annotations
    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    // Parse the body.
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/false)) return mlir::failure();

    // Make sure its not empty.
    if (body->empty()) return parser.emitError(loc, "expected non-empty table action body");

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// SwitchOp & CaseOp
//===----------------------------------------------------------------------===//
void P4HIR::CaseOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                        SmallVectorImpl<RegionSuccessor> &regions) {
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor());
        return;
    }

    regions.push_back(RegionSuccessor(&getCaseRegion()));
}

void P4HIR::CaseOp::build(OpBuilder &builder, OperationState &result, ArrayAttr value,
                          P4HIR::CaseOpKind kind,
                          llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> caseBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    result.addAttribute("value", value);
    result.getOrAddProperties<Properties>().kind =
        P4HIR::CaseOpKindAttr::get(builder.getContext(), kind);
    Region *caseRegion = result.addRegion();
    builder.createBlock(caseRegion);

    caseBuilder(builder, result.location);
}

LogicalResult P4HIR::CaseOp::verify() {
    // TODO: Check that case type corresponds to switch condition type
    return success();
}

ParseResult parseSwitchOp(OpAsmParser &parser, mlir::Region &bodyRegion,
                          mlir::OpAsmParser::UnresolvedOperand &cond, mlir::Type &condType) {
    if (parser.parseLParen() || parser.parseOperand(cond) || parser.parseColon() ||
        parser.parseType(condType) || parser.parseRParen() ||
        parser.parseRegion(bodyRegion, /*arguments=*/{},
                           /*argTypes=*/{}))
        return failure();

    return ::mlir::success();
}

void printSwitchOp(OpAsmPrinter &p, P4HIR::SwitchOp op, mlir::Region &bodyRegion,
                   mlir::Value condition, mlir::Type condType) {
    p << "(";
    p << condition;
    p << " : ";
    p.printStrippedAttrOrType(condType);
    p << ")";

    p << ' ';
    p.printRegion(bodyRegion, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

void P4HIR::SwitchOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                          SmallVectorImpl<RegionSuccessor> &regions) {
    // If any index all the underlying regions branch back to the parent
    // operation.
    if (!point.isParent()) {
        regions.push_back(RegionSuccessor());
        return;
    }

    regions.push_back(RegionSuccessor(&getBody()));
}

LogicalResult P4HIR::SwitchOp::verify() { return success(); }

void P4HIR::SwitchOp::build(OpBuilder &builder, OperationState &result, mlir::Value cond,
                            function_ref<void(OpBuilder &, Location)> switchBuilder) {
    assert(switchBuilder && "the builder callback for regions must be present");
    OpBuilder::InsertionGuard guardSwitch(builder);
    Region *switchRegion = result.addRegion();
    builder.createBlock(switchRegion);
    result.addOperands(cond);
    switchBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void P4HIR::ForOp::build(
    OpBuilder &builder, OperationState &result,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> updateBuilder) {
    OpBuilder::InsertionGuard guard(builder);

    Region *condRegion = result.addRegion();
    builder.createBlock(condRegion);
    condBuilder(builder, result.location);

    Region *bodyRegion = result.addRegion();
    builder.createBlock(bodyRegion);
    bodyBuilder(builder, result.location);

    Region *updateRegion = result.addRegion();
    builder.createBlock(updateRegion);
    updateBuilder(builder, result.location);
}

void P4HIR::ForOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                       SmallVectorImpl<mlir::RegionSuccessor> &regions) {
    // The entry into the operation is always the condition region
    if (point.isParent()) {
        regions.push_back(RegionSuccessor(&getCond()));
        return;
    }

    Region *from = point.getRegionOrNull();
    assert(from && "expected non-null origin region");

    // After evaluating the loop condition:
    // - Control may enter the body if the condition is true
    // - Or exit the loop if false
    if (from == &getCond()) {
        regions.push_back(RegionSuccessor(&getBody()));
        regions.push_back(RegionSuccessor());
        return;
    }

    // After executing the body, proceed to the update region
    if (from == &getBody()) {
        regions.push_back(RegionSuccessor(&getUpdates()));
        return;
    }

    // After updates, re-check the loop condition
    if (from == &getUpdates()) {
        regions.push_back(RegionSuccessor(&getCond()));
        return;
    }

    llvm_unreachable("Unknown branch origin");
}

llvm::SmallVector<Region *> P4HIR::ForOp::getLoopRegions() {
    // TODO: The LoopLikeInterface ODS says:
    // > Returns the regions that make up the body of the loop and should be
    // > inspected for loop-invariant operations.
    // When they say body does that include the condition and updates?
    // Should this be {&getCond(), &getBody(), &getUpdates} instead?
    // For ClangIR they only return {&getBody()}
    return {&getBody()};
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange P4HIR::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
    auto parent = dyn_cast<P4HIR::ForOp>(getOperation()->getParentOp());
    assert(parent && "p4hir.condition must be inside a p4hir.for");

    assert((point.isParent() || point.getRegionOrNull() == &parent.getBody()) &&
           "condition op can only exit the loop or branch to the body region");

    // No values are yielded to the successor region
    return MutableOperandRange(getOperation(), 0, 0);
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

        if (auto validBitType = mlir::dyn_cast<P4HIR::ValidBitType>(type)) {
            os << validBitType.getAlias();
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

        if (auto headerType = mlir::dyn_cast<P4HIR::HeaderType>(type)) {
            os << headerType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto headerUnionType = mlir::dyn_cast<P4HIR::HeaderUnionType>(type)) {
            os << headerUnionType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(type)) {
            os << enumType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto serEnumType = mlir::dyn_cast<P4HIR::SerEnumType>(type)) {
            os << serEnumType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto errorType = mlir::dyn_cast<P4HIR::ErrorType>(type)) {
            os << errorType.getAlias();
            return AliasResult::OverridableAlias;
        }

        if (auto stringType = mlir::dyn_cast<P4HIR::StringType>(type)) {
            os << stringType.getAlias();
            return AliasResult::OverridableAlias;
        }

        if (auto typevarType = mlir::dyn_cast<P4HIR::TypeVarType>(type)) {
            os << "type_" << typevarType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type)) {
            os << aliasType.getName();
            return AliasResult::OverridableAlias;
        }

        if (auto parserType = mlir::dyn_cast<P4HIR::ParserType>(type)) {
            os << parserType.getName();
            for (auto typeArg : parserType.getTypeArguments()) {
                os << "_";
                getAlias(typeArg, os);
            }
            return AliasResult::OverridableAlias;
        }

        if (auto controlType = mlir::dyn_cast<P4HIR::ControlType>(type)) {
            os << controlType.getName();
            for (auto typeArg : controlType.getTypeArguments()) {
                os << "_";
                getAlias(typeArg, os);
            }
            return AliasResult::OverridableAlias;
        }

        if (auto externType = mlir::dyn_cast<P4HIR::ExternType>(type)) {
            os << externType.getName();
            for (auto typeArg : externType.getTypeArguments()) {
                os << "_";
                getAlias(typeArg, os);
            }
            return AliasResult::OverridableAlias;
        }

        if (auto packageType = mlir::dyn_cast<P4HIR::PackageType>(type)) {
            os << packageType.getName();
            for (auto typeArg : packageType.getTypeArguments()) {
                os << "_";
                getAlias(typeArg, os);
            }
            return AliasResult::OverridableAlias;
        }

        if (auto ctorType = mlir::dyn_cast<P4HIR::CtorType>(type)) {
            os << "ctor_";
            getAlias(ctorType.getReturnType(), os);
            return AliasResult::OverridableAlias;
        }

        return AliasResult::NoAlias;
    }

    AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
        if (auto boolAttr = mlir::dyn_cast<P4HIR::BoolAttr>(attr)) {
            os << (boolAttr.getValue() ? "true" : "false");
            if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(boolAttr.getType()))
                os << "_" << aliasType.getName();

            return AliasResult::FinalAlias;
        }

        if (auto intAttr = mlir::dyn_cast<P4HIR::IntAttr>(attr)) {
            os << "int" << intAttr.getValue();
            if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(intAttr.getType()))
                os << "_" << bitsType.getAlias();
            else if (auto infintType = mlir::dyn_cast<P4HIR::InfIntType>(intAttr.getType()))
                os << "_" << infintType.getAlias();
            else if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(intAttr.getType()))
                os << "_" << aliasType.getName();

            return AliasResult::FinalAlias;
        }

        if (auto dirAttr = mlir::dyn_cast<P4HIR::ParamDirectionAttr>(attr)) {
            os << stringifyEnum(dirAttr.getValue());
            return AliasResult::FinalAlias;
        }

        if (auto validAttr = mlir::dyn_cast<P4HIR::ValidityBitAttr>(attr)) {
            os << stringifyEnum(validAttr.getValue());
            return AliasResult::FinalAlias;
        }

        if (auto errorAttr = mlir::dyn_cast<P4HIR::ErrorCodeAttr>(attr)) {
            os << "error_" << errorAttr.getField().getValue();
            return AliasResult::FinalAlias;
        }

        if (auto enumFieldAttr = mlir::dyn_cast<P4HIR::EnumFieldAttr>(attr)) {
            if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(enumFieldAttr.getType()))
                os << enumType.getName() << "_" << enumFieldAttr.getField().getValue();
            else
                os << mlir::cast<P4HIR::SerEnumType>(enumFieldAttr.getType()).getName() << "_"
                   << enumFieldAttr.getField().getValue();

            return AliasResult::FinalAlias;
        }

        if (auto ctorParamAttr = mlir::dyn_cast<P4HIR::CtorParamAttr>(attr)) {
            os << ctorParamAttr.getParent().getRootReference().getValue() << "_"
               << ctorParamAttr.getName().getValue();
            return AliasResult::FinalAlias;
        }

        if (auto matchKindAttr = mlir::dyn_cast<P4HIR::MatchKindAttr>(attr)) {
            os << matchKindAttr.getValue().getValue();
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
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.cpp.inc"
