#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void P4HIR::AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                            ::mlir::Type ref, ::mlir::Type objectType, ::llvm::StringRef name) {
    odsState.addAttribute(getObjectTypeAttrName(odsState.name), ::mlir::TypeAttr::get(objectType));
    odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
    odsState.addTypes(ref);
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

void P4HIR::P4HIRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
        >();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.cpp.inc"
