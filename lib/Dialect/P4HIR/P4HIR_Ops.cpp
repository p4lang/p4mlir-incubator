#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace mlir;
using namespace P4::P4MLIR;

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

static P4HIR::IntAttr applyToIntegerAttrs(
    mlir::PatternRewriter &builder, mlir::Value res, mlir::Attribute lhs, mlir::Attribute rhs,
    llvm::function_ref<llvm::APInt(const llvm::APInt &, const llvm::APInt &)> binFn) {
    llvm::APInt lhsVal = mlir::cast<P4HIR::IntAttr>(lhs).getValue();
    llvm::APInt rhsVal = mlir::cast<P4HIR::IntAttr>(rhs).getValue();
    llvm::APInt value = binFn(lhsVal, rhsVal);
    return P4HIR::IntAttr::get(res.getType(), value);
}

static P4HIR::IntAttr addIntegerAttrs(mlir::PatternRewriter &builder, mlir::Value res,
                                      mlir::Attribute lhs, mlir::Attribute rhs) {
    return applyToIntegerAttrs(builder, res, lhs, rhs, std::plus<APInt>());
}

static P4HIR::IntAttr subIntegerAttrs(mlir::PatternRewriter &builder, mlir::Value res,
                                      mlir::Attribute lhs, mlir::Attribute rhs) {
    return applyToIntegerAttrs(builder, res, lhs, rhs, std::minus<APInt>());
}

// Helper function to check if `attr` is an integer constant that equals the signed amount `val`.
static bool isIntegerValue(mlir::Attribute attr, int64_t val) {
    if (auto intAttr = mlir::dyn_cast_if_present<P4HIR::IntAttr>(attr)) {
        llvm::APInt intVal = intAttr.getValue();
        return intVal == llvm::APInt(intVal.getBitWidth(), val, true);
    }

    return false;
}

// Helper function to create a integer `attr` that represents `val` sign-extended to match `type`.
static mlir::Attribute getIntegerAttr(mlir::Type type, int64_t val) {
    if (auto intType = mlir::dyn_cast<P4HIR::BitsType>(type))
        return P4HIR::IntAttr::get(type, llvm::APInt(intType.getWidth(), val, true));
    if (auto intType = mlir::dyn_cast<P4HIR::InfIntType>(type))
        return P4HIR::IntAttr::get(type, llvm::APInt(64, val, true));

    return {};
}

static bool isSignedIntegerType(const mlir::Type type) {
    if (const auto bitsType = dyn_cast<P4HIR::BitsType>(type)) return bitsType.isSigned();

    // InfIntType is always considered a signed integer type
    if (mlir::isa<P4HIR::InfIntType>(type)) return true;

    return false;
}

namespace {
#include "p4mlir/Dialect/P4HIR/P4HIR_Patterns.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
    if (mlir::isa<P4HIR::BoolAttr>(attrType)) {
        if (auto aliasedType = mlir::dyn_cast<P4HIR::AliasType>(opType))
            opType = aliasedType.getCanonicalType();
        if (!mlir::isa<P4HIR::BoolType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.bool' for '" << attrType << "'";
        return success();
    }

    if (mlir::isa<P4HIR::IntAttr>(attrType)) {
        if (auto aliasedType = mlir::dyn_cast<P4HIR::AliasType>(opType))
            opType = aliasedType.getCanonicalType();
        if (!mlir::isa<P4HIR::BitsType, P4HIR::InfIntType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") does not match value type (" << attrType << ")";
        return success();
    }

    if (mlir::isa<P4HIR::AggAttr>(attrType)) {
        if (!mlir::isa<P4HIR::StructType, P4HIR::HeaderType, P4HIR::HeaderUnionType,
                       mlir::TupleType, P4HIR::ArrayType>(opType))
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

    if (mlir::isa<P4HIR::UniversalSetAttr, P4HIR::SetAttr>(attrType)) {
        if (!mlir::isa<P4HIR::SetType>(opType))
            return op->emitOpError("result type (")
                   << opType << ") must be '!p4hir.set' for '" << attrType << "'";
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
    } else if (mlir::isa<P4HIR::UniversalSetAttr>(getValue())) {
        setNameFn(getResult(), "everything");
    } else if (mlir::isa<P4HIR::SetAttr>(getValue())) {
        setNameFn(getResult(), "set");
    } else {
        setNameFn(getResult(), "cst");
    }
}

OpFoldResult P4HIR::ConstOp::fold(FoldAdaptor adaptor) {
    assert(adaptor.getOperands().empty() && "constant has no operands");
    return adaptor.getValueAttr();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void P4HIR::CastOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "cast");
}

OpFoldResult P4HIR::CastOp::fold(FoldAdaptor) {
    // Identity.
    // cast(%a) : A -> A ==> %a
    if (getOperand().getType() == getType()) return getOperand();

    // Casts of integer constants
    if (auto inputConst = mlir::dyn_cast_if_present<ConstOp>(getOperand().getDefiningOp()))
        if (auto castResult = P4HIR::foldConstantCast(getType(), inputConst.getValue()))
            return castResult;

    return {};
}

LogicalResult P4HIR::CastOp::canonicalize(P4HIR::CastOp op, PatternRewriter &rewriter) {
    // Composition.
    // %b = cast(%a) : A -> B
    //      cast(%b) : B -> C
    // ===> cast(%a) : A -> C
    if (auto inputCast = mlir::dyn_cast_if_present<CastOp>(op.getSrc().getDefiningOp())) {
        auto bitcast =
            rewriter.createOrFold<P4HIR::CastOp>(op.getLoc(), op.getType(), inputCast.getSrc());
        rewriter.replaceOp(op, bitcast);
        return success();
    }

    return failure();
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

static P4HIR::UnaryOp getDefiningUnop(P4HIR::UnaryOpKind kind, mlir::Value val) {
    if (auto unop = val.getDefiningOp<P4HIR::UnaryOp>())
        if (unop.getKind() == kind) return unop;
    return {};
}

static P4HIR::BinOp getDefiningBinop(P4HIR::BinOpKind kind, mlir::Value val) {
    if (auto binop = val.getDefiningOp<P4HIR::BinOp>())
        if (binop.getKind() == kind) return binop;
    return {};
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

void P4HIR::UnaryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

OpFoldResult P4HIR::UnaryOp::fold(FoldAdaptor adaptor) {
    P4HIR::UnaryOpKind kind = getKind();

    // Identity.
    // plus(x) -> x
    if (kind == P4HIR::UnaryOpKind::UPlus) return getInput();

    bool isIdempotent = (kind == P4HIR::UnaryOpKind::LNot) || (kind == P4HIR::UnaryOpKind::Cmpl) ||
                        (kind == P4HIR::UnaryOpKind::Neg);

    // OP(OP(x)) = x
    if (isIdempotent)
        if (auto inputOp = getDefiningUnop(kind, getInput())) return inputOp.getInput();

    // Constant folding
    if (auto opAttr = adaptor.getInput()) {
        if (kind == P4HIR::UnaryOpKind::LNot) {
            if (auto boolAttr = mlir::dyn_cast<P4HIR::BoolAttr>(opAttr)) {
                return P4HIR::BoolAttr::get(getContext(), !boolAttr.getValue());
            }
        } else if (kind == P4HIR::UnaryOpKind::Neg) {
            if (auto intAttr = mlir::dyn_cast<P4HIR::IntAttr>(opAttr)) {
                return P4HIR::IntAttr::get(intAttr.getType(), -intAttr.getValue());
            }
        } else if (kind == P4HIR::UnaryOpKind::Cmpl) {
            if (auto intAttr = mlir::dyn_cast<P4HIR::IntAttr>(opAttr)) {
                return P4HIR::IntAttr::get(intAttr.getType(), ~intAttr.getValue());
            }
        }
        // UPlus gets handled by the identity rule above
    }

    return {};
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

// Assuming that `op` is a commutative operation, canonicalize the position of constant argumets.
static void sortCommutativeArgs(Operation *op, ArrayRef<Attribute> operands) {
    assert(mlir::isa<P4HIR::BinOp>(op) && mlir::cast<P4HIR::BinOp>(op).isCommutative());

    OpOperand *operandsBegin = op->getOpOperands().begin();
    auto isNonConstant = [&](OpOperand &o) {
        return !static_cast<bool>(operands[std::distance(operandsBegin, &o)]);
    };
    auto *firstConstantIt = llvm::find_if_not(op->getOpOperands(), isNonConstant);
    std::stable_partition(firstConstantIt, op->getOpOperands().end(), isNonConstant);
}

// Describes how to extend InfInt arguments before constant folding:
// None: Leave argument bit widths unaffected.
// Max: Extend the argument with smaller bit width to match the one with the larger bit width.
// AddLike: Like MAX but with one bit more to potentially store a carry/borrow bit.
// MulLike: Extend arguments to the sum of their bit widths.
// ShlLike: Extend the LHS by the amount specified in RHS.
enum class InfIntExt { None, Max, AddLike, MulLike, ShlLike };

// Helper function to constant fold a binary operation.
// `operands` is an array of constant operands and `calculate` is a functor object describing a
// binary operation on those arguments. `calculate` will be called with two APSInt arguments and
// should produce a result suitable to `resultType` (APSInt or bool). If folding is successfull a
// constant of type `resultType` is returned. `extKind` describes how to extend InfInt arguments
// before calling `calculate`.
template <class F>
static Attribute constFoldBinOp(llvm::ArrayRef<Attribute> operands, mlir::Type resultType,
                                InfIntExt extKind, F &&calculate) {
    assert(operands.size() == 2 && "binary op takes two operands");

    if (!resultType || !operands[0] || !operands[1]) return {};

    auto lhs = P4HIR::getConstantInt(operands[0]);
    auto rhs = P4HIR::getConstantInt(operands[1]);

    if (!lhs || !rhs) return {};

    auto isInfInt = [](Attribute attr) {
        return mlir::isa<P4HIR::InfIntType>(mlir::cast<TypedAttr>(attr).getType());
    };

    if (isInfInt(operands[0]) || isInfInt(operands[1])) {
        unsigned lhsBits = lhs->getActiveBits() + 1;
        unsigned rhsBits = rhs->getActiveBits() + 1;

        switch (extKind) {
            case InfIntExt::None:
                break;
            case InfIntExt::Max:
                lhsBits = rhsBits = std::max(lhsBits, rhsBits);
                break;
            case InfIntExt::AddLike:
                lhsBits = rhsBits = std::max(lhsBits, rhsBits) + 1;
                break;
            case InfIntExt::MulLike:
                lhsBits = rhsBits = (lhsBits + rhsBits) + 1;
                break;
            case InfIntExt::ShlLike: {
                lhsBits = (lhsBits + rhs->getLimitedValue());
                break;
            }
        }

        lhs = lhs->extOrTrunc(lhsBits);
        rhs = rhs->extOrTrunc(rhsBits);
    }

    auto calRes = calculate(lhs.value(), rhs.value());

    if constexpr (std::is_same_v<decltype(calRes), bool>)
        return P4HIR::BoolAttr::get(resultType.getContext(), calRes);
    else
        return P4HIR::IntAttr::get(resultType.getContext(), resultType, calRes);
}

void P4HIR::BinOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

LogicalResult P4HIR::BinOp::verify() {
    if (mlir::isa<P4HIR::BitsType>(getType())) return success();

    if (mlir::isa<P4HIR::InfIntType>(getType())) {
        switch (getKind()) {
            case BinOpKind::Mul:
            case BinOpKind::Div:
            case BinOpKind::Mod:
            case BinOpKind::Add:
            case BinOpKind::Sub:
                return mlir::success();
            case BinOpKind::AddSat:
            case BinOpKind::SubSat:
                return emitOpError() << "Saturating arithmetic ('" << stringifyEnum(getKind())
                                     << "') is not valid for " << getType();
            case BinOpKind::Or:
            case BinOpKind::Xor:
            case BinOpKind::And:
                return emitOpError() << "Bitwise operations ('" << stringifyEnum(getKind())
                                     << "') are not valid for " << getType();
        }

        return emitOpError("Unknown BinOp kind");
    }

    return emitOpError("Unknown BinOp result type");
}

OpFoldResult P4HIR::BinOp::fold(FoldAdaptor adaptor) {
    P4HIR::BinOpKind kind = getKind();

    if (isCommutative()) sortCommutativeArgs(getOperation(), adaptor.getOperands());

    auto foldIntBinop = [&](auto binop, InfIntExt ext = InfIntExt::Max) {
        return constFoldBinOp(adaptor.getOperands(), getType(), ext, binop);
    };

    if (kind == P4HIR::BinOpKind::Add) {
        // addi(a, 0) -> a
        if (isIntegerValue(adaptor.getRhs(), 0)) return getLhs();

        // addi(subi(a, b), b) -> a
        if (auto sub = getDefiningBinop(P4HIR::BinOpKind::Sub, getLhs()))
            if (getRhs() == sub.getRhs()) return sub.getLhs();

        // addi(b, subi(a, b)) -> a
        if (auto sub = getDefiningBinop(P4HIR::BinOpKind::Sub, getRhs()))
            if (getLhs() == sub.getRhs()) return sub.getLhs();

        return foldIntBinop(std::plus<llvm::APSInt>{}, InfIntExt::AddLike);
    } else if (kind == P4HIR::BinOpKind::Sub) {
        // subi(a, 0) -> a
        if (isIntegerValue(adaptor.getRhs(), 0)) return getLhs();

        // subi(a, a) -> 0
        if (getRhs() == getLhs()) return getIntegerAttr(getType(), 0);

        if (auto add = getDefiningBinop(P4HIR::BinOpKind::Add, getLhs())) {
            // subi(addi(a, b), b) -> a
            if (getRhs() == add.getRhs()) return add.getLhs();

            // subi(addi(a, b), a) -> b
            if (getRhs() == add.getLhs()) return add.getRhs();
        }

        return foldIntBinop(std::minus<llvm::APSInt>{}, InfIntExt::AddLike);
    } else if (kind == P4HIR::BinOpKind::AddSat) {
        // add_sat(a, 0) -> 0
        if (isIntegerValue(adaptor.getRhs(), 0)) return getLhs();

        return foldIntBinop([&](const auto &a, const auto &b) {
            if (isSignedIntegerType(getType()))
                return a.sadd_sat(b);
            else
                return a.uadd_sat(b);
        });
    } else if (kind == P4HIR::BinOpKind::SubSat) {
        // sub_sat(a, 0) -> 0
        if (isIntegerValue(adaptor.getRhs(), 0)) return getLhs();

        // sub_sat(a, a) -> 0
        if (getRhs() == getLhs()) return getIntegerAttr(getType(), 0);

        return foldIntBinop([&](const auto &a, const auto &b) {
            if (isSignedIntegerType(getType()))
                return a.ssub_sat(b);
            else
                return a.usub_sat(b);
        });
    } else if (kind == P4HIR::BinOpKind::Mul) {
        // mul(a, 1) -> a
        if (isIntegerValue(adaptor.getRhs(), 1)) return getLhs();

        // mul(a, 0) -> 0
        if (isIntegerValue(adaptor.getRhs(), 0)) return getIntegerAttr(getType(), 0);

        return foldIntBinop(std::multiplies<llvm::APInt>{}, InfIntExt::MulLike);
    } else if (kind == P4HIR::BinOpKind::Div) {
        // div(a, 1) -> a
        if (isIntegerValue(adaptor.getRhs(), 1)) return getLhs();

        // div(0, a) -> 0
        if (isIntegerValue(adaptor.getLhs(), 0)) return getIntegerAttr(getType(), 0);

        return foldIntBinop(std::divides<llvm::APSInt>{});
    } else if (kind == P4HIR::BinOpKind::Mod) {
        // mod(a, 1) -> 0
        if (isIntegerValue(adaptor.getRhs(), 1)) return getIntegerAttr(getType(), 0);

        // mod(0, a) -> 0
        if (isIntegerValue(adaptor.getLhs(), 0)) return getIntegerAttr(getType(), 0);

        return foldIntBinop(std::modulus<llvm::APSInt>{});
    } else if (kind == P4HIR::BinOpKind::And || kind == P4HIR::BinOpKind::Or) {
        // 0 and -1 represent all-zeros or all-ones constants when sign-extended in
        // `isIntegerValue`.
        int64_t neutralVal = (kind == P4HIR::BinOpKind::And) ? int64_t(-1) : int64_t(0);
        int64_t absorbVal = (kind == P4HIR::BinOpKind::And) ? int64_t(0) : int64_t(-1);

        // OP(a, neutralVal) -> a
        if (isIntegerValue(adaptor.getRhs(), neutralVal)) return getLhs();

        /// OP(a, absorbVal) -> absorbVal
        if (isIntegerValue(adaptor.getRhs(), absorbVal))
            return getIntegerAttr(getType(), absorbVal);

        /// OP(a, a) -> a
        if (getLhs() == getRhs()) return getLhs();

        /// OP(OP(x, a), a) -> OP(x, a)
        /// OP(OP(a, x), a) -> OP(a, x)
        if (auto nested = getDefiningBinop(kind, getLhs()))
            if (nested.getLhs() == getRhs() || nested.getRhs() == getRhs()) return getLhs();

        /// OP(a, OP(x, a)) -> OP(x, a)
        /// OP(a, OP(a, x)) -> OP(a, x)
        if (auto nested = getDefiningBinop(kind, getRhs()))
            if (nested.getLhs() == getLhs() || nested.getRhs() == getLhs()) return getRhs();

        /// OP(not(x), x) -> absorbVal
        if (auto bitnot = getDefiningUnop(P4HIR::UnaryOpKind::Cmpl, getLhs()))
            if (bitnot.getInput() == getRhs()) return getIntegerAttr(getType(), absorbVal);

        /// OP(x, not(x)) -> absorbVal
        if (auto bitnot = getDefiningUnop(P4HIR::UnaryOpKind::Cmpl, getRhs()))
            if (bitnot.getInput() == getLhs()) return getIntegerAttr(getType(), absorbVal);

        if (kind == P4HIR::BinOpKind::And)
            return foldIntBinop(std::bit_and<llvm::APInt>{});
        else
            return foldIntBinop(std::bit_or<llvm::APInt>{});
    } else if (kind == P4HIR::BinOpKind::Xor) {
        // xor(a, 0) -> a
        if (isIntegerValue(adaptor.getRhs(), 0)) return getLhs();

        /// xor(x, x) -> 0
        if (getLhs() == getRhs()) return getIntegerAttr(getType(), 0);

        /// xor(xor(x, a), a) -> x
        /// xor(xor(a, x), a) -> x
        if (auto nested = getDefiningBinop(P4HIR::BinOpKind::Xor, getLhs())) {
            if (nested.getRhs() == getRhs()) return nested.getLhs();
            if (nested.getLhs() == getRhs()) return nested.getRhs();
        }

        /// xor(a, xor(x, a)) -> x
        /// xor(a, xor(a, x)) -> x
        if (auto nested = getDefiningBinop(P4HIR::BinOpKind::Xor, getRhs())) {
            if (nested.getRhs() == getLhs()) return nested.getLhs();
            if (nested.getLhs() == getLhs()) return nested.getRhs();
        }

        /// xor(not(x), x) -> 11...11
        if (auto bitnot = getDefiningUnop(P4HIR::UnaryOpKind::Cmpl, getLhs()))
            if (bitnot.getInput() == getRhs()) return getIntegerAttr(getType(), int64_t(-1));

        /// xor(x, not(x)) -> 11...11
        if (auto bitnot = getDefiningUnop(P4HIR::UnaryOpKind::Cmpl, getRhs()))
            if (bitnot.getInput() == getLhs()) return getIntegerAttr(getType(), int64_t(-1));

        return foldIntBinop(std::bit_xor<llvm::APInt>{});
    }

    return {};
}

void P4HIR::BinOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
    patterns.add<AddAddCst, SubAddCst, AddSubCst, SubSubCst, AddSubLhsCst, SubSubLhsCst,
                 SubRhsSubCst, SubRhsSubLhsCst, AddNeg, AddRhsNeg, SubRhsNeg, MulToNeg, SubToNeg,
                 SubSatToNeg, AddCmplToNeg, MulMulCst>(context);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult P4HIR::ConcatOp::verify() {
    auto lhsType = cast<BitsType>(getLhs().getType());
    auto rhsType = cast<BitsType>(getRhs().getType());
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

void P4HIR::ShlOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "shl");
}

void P4HIR::ShrOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "shr");
}

LogicalResult verifyShiftOperation(Operation *op, Type rhsType) {
    // FIXME: Relax this condition for compile-time known non-negative values.
    if (auto rhsBitsType = mlir::dyn_cast<P4HIR::BitsType>(rhsType)) {
        if (rhsBitsType.isSigned()) {
            return op->emitOpError("the right-hand side operand of a shift must be unsigned");
        }
    }
    return success();
}

LogicalResult P4HIR::ShlOp::verify() {
    auto rhsType = getRhs().getType();
    return verifyShiftOperation(getOperation(), rhsType);
}

LogicalResult P4HIR::ShrOp::verify() {
    auto rhsType = getRhs().getType();
    return verifyShiftOperation(getOperation(), rhsType);
}

template <typename ShiftOp>
OpFoldResult foldZeroConstants(ShiftOp op, typename ShiftOp::FoldAdaptor adaptor) {
    // shl/shr(x, 0) -> x
    if (isIntegerValue(adaptor.getRhs(), 0)) return op.getLhs();

    // shl/shr(0, c) -> 0
    if (isIntegerValue(adaptor.getLhs(), 0)) return getIntegerAttr(op.getType(), 0);

    return {};
}

OpFoldResult P4HIR::ShlOp::fold(FoldAdaptor adaptor) {
    if (auto fold = foldZeroConstants(*this, adaptor)) {
        return fold;
    }

    auto rhsAttr = mlir::dyn_cast_if_present<P4HIR::IntAttr>(adaptor.getRhs());
    if (!rhsAttr) return {};
    auto shift = rhsAttr.getValue();

    // Shift overflow.
    // shl(%x : bit/int<W>, c) -> 0 if c >= W
    if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(getType())) {
        unsigned width = bitsType.getWidth();
        if (shift.uge(width)) return getIntegerAttr(bitsType, 0);
    }

    return constFoldBinOp(adaptor.getOperands(), getType(), InfIntExt::ShlLike,
                          [&](const auto &a, const auto &b) { return a << b.getZExtValue(); });
}

OpFoldResult P4HIR::ShrOp::fold(FoldAdaptor adaptor) {
    if (auto fold = foldZeroConstants(*this, adaptor)) {
        return fold;
    }

    auto rhsAttr = mlir::dyn_cast_if_present<P4HIR::IntAttr>(adaptor.getRhs());
    if (!rhsAttr) return {};
    auto shift = rhsAttr.getValue();

    // Shift overflow on unsigned fixed-width integers.
    // shr(%x : bit<W>, c) -> 0 if c >= W
    if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(getType())) {
        unsigned width = bitsType.getWidth();
        if (bitsType.isUnsigned() && shift.uge(width)) return getIntegerAttr(bitsType, 0);
    }

    return constFoldBinOp(adaptor.getOperands(), getType(), InfIntExt::None,
                          [&](const auto &a, const auto &b) {
                              return a >> std::min(b.getLimitedValue(), (uint64_t)a.getBitWidth());
                          });
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

void P4HIR::CmpOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), stringifyEnum(getKind()));
}

OpFoldResult P4HIR::CmpOp::fold(FoldAdaptor adaptor) {
    P4HIR::CmpOpKind kind = getKind();

    // cmp(kind, x, x)
    if (getLhs() == getRhs()) {
        switch (kind) {
            case P4HIR::CmpOpKind::Lt:
            case P4HIR::CmpOpKind::Gt:
            case P4HIR::CmpOpKind::Ne:
                return P4HIR::BoolAttr::get(getContext(), false);
            case P4HIR::CmpOpKind::Le:
            case P4HIR::CmpOpKind::Ge:
            case P4HIR::CmpOpKind::Eq:
                return P4HIR::BoolAttr::get(getContext(), true);
        }
    }

    // Move constant to the right side.
    if (adaptor.getLhs() && !adaptor.getRhs()) {
        using KindPair = std::pair<P4HIR::CmpOpKind, P4HIR::CmpOpKind>;
        const KindPair swapKinds[] = {
            {P4HIR::CmpOpKind::Lt, P4HIR::CmpOpKind::Gt},
            {P4HIR::CmpOpKind::Gt, P4HIR::CmpOpKind::Lt},
            {P4HIR::CmpOpKind::Le, P4HIR::CmpOpKind::Ge},
            {P4HIR::CmpOpKind::Ge, P4HIR::CmpOpKind::Le},
            {P4HIR::CmpOpKind::Eq, P4HIR::CmpOpKind::Eq},
            {P4HIR::CmpOpKind::Ne, P4HIR::CmpOpKind::Ne},
        };

        for (auto [from, to] : swapKinds) {
            if (kind == from) {
                setKind(to);
                mlir::Value lhs = getLhs();
                mlir::Value rhs = getRhs();
                getLhsMutable().assign(rhs);
                getRhsMutable().assign(lhs);
                return getResult();
            }
        }

        llvm_unreachable("unknown cmp kind");
    }

    auto binop = [&](const auto &a, const auto &b) {
        switch (kind) {
            case P4HIR::CmpOpKind::Lt:
                return a < b;
            case P4HIR::CmpOpKind::Gt:
                return a > b;
            case P4HIR::CmpOpKind::Le:
                return a <= b;
            case P4HIR::CmpOpKind::Ge:
                return a >= b;
            case P4HIR::CmpOpKind::Eq:
                return a == b;
            case P4HIR::CmpOpKind::Ne:
                return a != b;
            default:
                break;
        }

        llvm_unreachable("Unknown cmp kind");
        return false;
    };

    return constFoldBinOp(adaptor.getOperands(), getType(), InfIntExt::Max, binop);
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void P4HIR::VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    if (getName() && !getName()->empty()) setNameFn(getResult(), *getName());
}

LogicalResult P4HIR::VariableOp::canonicalize(P4HIR::VariableOp op, PatternRewriter &rewriter) {
    // Check if the variable has one unique assignment to it, all other
    // uses are reads, and that all uses are in the same block as the variable
    // itself.
    P4HIR::AssignOp uniqueAssignOp;
    for (auto *user : op->getUsers()) {
        // Ensure there is at most one unique assignment to the variable.
        if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(user)) {
            if (uniqueAssignOp) return failure();
            uniqueAssignOp = assignOp;
        }
    }

    if (!uniqueAssignOp) return failure();

    for (auto *user : op->getUsers()) {
        if (user == uniqueAssignOp) continue;
        if (user->getBlock() != uniqueAssignOp->getBlock()) return failure();

        // Ensure all other users are reads and after the write.
        if (!mlir::isa<ReadOp>(user) || user->isBeforeInBlock(uniqueAssignOp)) return failure();
    }

    // Remove the assign op and replace all reads with the new assigned var op.
    mlir::Value assignedValue = uniqueAssignOp.getValue();
    rewriter.eraseOp(uniqueAssignOp);
    for (auto *user : llvm::make_early_inc_range(op->getUsers())) {
        auto readOp = mlir::cast<P4HIR::ReadOp>(user);
        rewriter.replaceOp(readOp, assignedValue);
    }

    // Remove the original variable.
    rewriter.eraseOp(op);
    return success();
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
                          mlir::DictionaryAttr annotations, ArrayRef<DictionaryAttr> resAttrs) {
    result.addRegion();

    result.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute(getFunctionTypeAttrName(result.name), TypeAttr::get(type));
    // External functions are private, everything else is public
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        builder.getStringAttr(isExternal ? "private" : "public"));
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    call_interface_impl::addArgAndResultAttrs(builder, result, argAttrs, resAttrs,
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
    llvm::SmallVector<DictionaryAttr, 0> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 1> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignatureWithArguments(
            parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes, resultAttrs))
        return failure();

    // Actions have no results
    if (isAction && !resultTypes.empty())
        return parser.emitError(loc, "actions should not produce any results");
    else if (resultTypes.size() > 1)
        return parser.emitError(loc, "functions only supports zero or one results");

    // Build the function type.
    for (auto &arg : arguments) argTypes.push_back(arg.type);

    // Fetch return type or set it to void if empty / not present.
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
    call_interface_impl::addArgAndResultAttrs(builder, state, arguments, resultAttrs,
                                              getArgAttrsAttrName(state.name),
                                              getResAttrsAttrName(state.name));

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
                                 mlir::TypeRange typeOperands) {
    if (auto typeVar = llvm::dyn_cast<P4HIR::TypeVarType>(type)) {
        size_t pos = llvm::find(calleeTypeArgs, typeVar) - calleeTypeArgs.begin();
        if (pos == calleeTypeArgs.size()) return {};
        return typeOperands[pos];
    } else if (auto refType = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
        return P4HIR::ReferenceType::get(
            substituteType(refType.getObjectType(), calleeTypeArgs, typeOperands));
    } else if (auto tupleType = llvm::dyn_cast<mlir::TupleType>(type)) {
        llvm::SmallVector<mlir::Type> substituted;
        for (auto elTy : tupleType.getTypes())
            substituted.push_back(substituteType(elTy, calleeTypeArgs, typeOperands));
        return mlir::TupleType::get(type.getContext(), substituted);
    } else if (auto extType = llvm::dyn_cast<P4HIR::ExternType>(type)) {
        llvm::SmallVector<mlir::Type> substituted;
        for (auto typeArg : extType.getTypeArguments())
            substituted.push_back(substituteType(typeArg, calleeTypeArgs, typeOperands));
        return P4HIR::ExternType::get(type.getContext(), extType.getName(), substituted,
                                      extType.getAnnotations());
    }

    return type;
}

// Callee might be:
//  - Overload set, then we need to look for a particular overload
//  - Normal functions. They are defined at top-level only. Top-level actions are also here.
//  - Actions defined at control level. Check for them first.
// This largely duplicates verifySymbolUses() below, though the latter emits diagnostics
mlir::Operation *P4HIR::CallOp::resolveCallableInTable(mlir::SymbolTableCollection *symbolTable) {
    auto sym = (*this)->getAttrOfType<SymbolRefAttr>("callee");
    if (!sym) return nullptr;

    if (auto *decl = symbolTable->lookupSymbolIn(getParentModule(*this), sym)) {
        if (auto fn = llvm::dyn_cast<P4HIR::FuncOp>(decl)) {
            return fn;
        } else if (auto ovl = llvm::dyn_cast<P4HIR::OverloadSetOp>(decl)) {
            // Find the FuncOp with the correct # of operands
            for (Operation &nestedOp : ovl.getBody().front()) {
                auto f = llvm::cast<FuncOp>(nestedOp);
                if (f.getNumArguments() == getNumOperands()) return f;
            }
        }
    }

    return nullptr;
}

LogicalResult P4HIR::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto sym = (*this)->getAttrOfType<SymbolRefAttr>("callee");
    if (!sym) return emitOpError("requires a 'callee' symbol reference attribute");

    // Callee might be:
    //  - Overload set, then we need to look for a particular overload
    //  - Normal functions. They are defined at top-level only. Top-level actions are also here.
    //  - Actions defined at control level. References to them should be fully qualified
    P4HIR::FuncOp fn;
    if (auto *decl = symbolTable.lookupSymbolIn(getParentModule(*this), sym)) {
        if ((fn = llvm::dyn_cast<P4HIR::FuncOp>(decl))) {
            // Action nested in control
            if (fn->getParentOfType<P4HIR::ControlOp>() && !fn.getAction())
                return emitOpError() << "'" << sym << "' does not reference a valid action";
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

    if (!fn) return emitOpError() << "'" << sym << "' does not reference a valid function";

    auto fnType = fn.getFunctionType();
    // Verify that the operand and result types match the callee.
    if (fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    auto calleeTypeArgs = fnType.getTypeArguments();
    SmallVector<mlir::Type, 1> typeOperands;
    if (getTypeOperands())
        llvm::append_range(typeOperands, getTypeOperands()->getAsValueRange<mlir::TypeAttr>());
    if (calleeTypeArgs.size() != typeOperands.size())
        return emitOpError("incorrect number of type operands for callee");

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

OpFoldResult P4HIR::StructExtractOp::fold(FoldAdaptor adaptor) {
    // Fold extract from aggregate constant
    if (auto aggAttr = adaptor.getInput()) {
        return mlir::cast<P4HIR::AggAttr>(aggAttr).getFields()[getFieldIndex()];
    }
    // Fold extract from struct
    if (auto structOp = mlir::dyn_cast_if_present<P4HIR::StructOp>(getInput().getDefiningOp())) {
        return structOp.getOperand(getFieldIndex());
    }

    return {};
}

LogicalResult P4HIR::StructExtractOp::canonicalize(P4HIR::StructExtractOp op,
                                                   PatternRewriter &rewriter) {
    // Simple SROA / load shrinking: turn (struct_extract (read ref), field)
    // into (read (struct_extract_ref ref, field)) if `read` operation has a
    // single use. Usually these come from struct field access and it is
    // beneficial to project from whole-width read to a single-field read. We do
    // not do complete SROA here as it would require tracking writes as well as
    // reads.
    if (auto readOp = op.getInput().getDefiningOp<P4HIR::ReadOp>(); readOp && readOp->hasOneUse()) {
        OpBuilder::InsertionGuard guard(rewriter);
        auto ref = readOp.getRef();
        rewriter.setInsertionPoint(readOp);
        auto fieldRef = rewriter.create<P4HIR::StructExtractRefOp>(
            op.getLoc(), P4HIR::ReferenceType::get(op.getType()), ref, op.getFieldIndexAttr());
        rewriter.replaceOpWithNewOp<P4HIR::ReadOp>(op, fieldRef);
        rewriter.eraseOp(readOp);
        return success();
    }

    return failure();
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

OpFoldResult P4HIR::SliceOp::fold(FoldAdaptor adaptor) {
    if (adaptor.getInput()) {
        auto input = P4HIR::getConstantInt(adaptor.getInput()).value();
        auto sliceVal = input.extractBits((getHighBit() - getLowBit() + 1), getLowBit());
        return P4HIR::IntAttr::get(getContext(), getType(), sliceVal);
    }

    return {};
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

// Parser states use fully-qualified names so we lookup from the top-level moduleOp
static P4HIR::ParserStateOp lookupParserState(Operation *op, mlir::SymbolRefAttr stateName) {
    auto res = getParentModule(op).lookupSymbol<P4HIR::ParserStateOp>(stateName);
    assert(res && "expected valid parser state lookup");
    return res;
}

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
    call_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                              /*resultAttrs=*/{}, getArgAttrsAttrName(result.name),
                                              {});
}

void P4HIR::ParserOp::createEntryBlock() {
    assert(empty() && "can only create entry block for empty parser");
    Block &first = getFunctionBody().emplaceBlock();
    auto loc = getFunctionBody().getLoc();
    for (auto argType : getFunctionType().getInputs()) first.addArgument(argType, loc);
}

P4HIR::ParserStateOp P4HIR::ParserOp::getStartState() {
    auto transition = llvm::cast<ParserTransitionOp>(getBody().back().getTerminator());
    return lookupParserState(getOperation(), transition.getStateAttr());
}

mlir::SymbolRefAttr P4HIR::ParserStateOp::getSymbolRef() {
    auto parser = getOperation()->getParentOfType<P4HIR::ParserOp>();
    auto leafSymbol = mlir::FlatSymbolRefAttr::get(getContext(), getSymName());
    auto symbol = mlir::SymbolRefAttr::get(getContext(), parser.getSymName(), {leafSymbol});
    return symbol;
}

P4HIR::ParserStateOp::StateRange P4HIR::ParserStateOp::getNextStates() {
    auto &block = getBody().back();

    if (block.begin() == block.end()) return {block.begin(), block.end()};

    return mlir::TypeSwitch<Operation *, StateRange>(this->getNextTransition())
        .Case<ParserTransitionOp>([&](auto) {
            // Wrap terminator by itself
            return StateRange{--block.end(), block.end()};
        })
        .Case<ParserTransitionSelectOp>([&](ParserTransitionSelectOp select) {
            auto selects = select.selects();
            // Wrap filtered iterator over select cases
            return StateRange(mlir::Block::iterator(selects.begin()),
                              mlir::Block::iterator(selects.end()));
        })
        .Case<ParserAcceptOp, ParserRejectOp>(
            [&](auto) { return StateRange{block.end(), block.end()}; })
        .Default([&](auto) {
            llvm_unreachable("Unknown parser terminator");
            return StateRange{block.end(), block.end()};
        });
}

P4HIR::ParserStateOp P4HIR::ParserStateOp::StateIterator::mapElement(mlir::Operation &op) const {
    return mlir::TypeSwitch<Operation *, ParserStateOp>(&op)
        .Case<ParserTransitionOp>([&](ParserTransitionOp transition) {
            return lookupParserState(&op, transition.getStateAttr());
        })
        .Case<ParserSelectCaseOp>([&](ParserSelectCaseOp select) {
            return lookupParserState(&op, select.getStateAttr());
        })
        .Default([&](auto) {
            llvm_unreachable("Unknown parser terminator");
            return nullptr;
        });
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
    llvm::SmallVector<DictionaryAttr, 8> argAttrs;
    llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic;
    if (function_interface_impl::parseFunctionSignatureWithArguments(
            parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Parsers have no results
    if (!resultTypes.empty() || !resultAttrs.empty())
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
    if (mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    // Add the attributes to the function arguments.
    assert(argAttrs.empty() || argAttrs.size() == argTypes.size());
    call_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
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
    if (!symbolTable.lookupSymbolIn<P4HIR::ParserStateOp>(getParentModule(op), stateName))
        return op->emitOpError() << "'" << stateName << "' does not reference a valid state";

    return mlir::success();
}

mlir::LogicalResult P4HIR::ParserTransitionOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
    return verifyStateTarget(*this, getStateAttr(), symbolTable);
}

P4HIR::ParserStateOp P4HIR::ParserTransitionOp::getNextState() {
    return lookupParserState(getOperation(), getStateAttr());
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

P4HIR::ParserStateOp P4HIR::ParserTransitionSelectOp::StateIterator::mapElement(
    P4HIR::ParserSelectCaseOp op) const {
    return lookupParserState(op.getOperation(), op.getStateAttr());
}

bool P4HIR::isUniversalSetValue(mlir::Value val) {
    auto cst = val.getDefiningOp<P4HIR::ConstOp>();
    return cst && mlir::isa<P4HIR::UniversalSetAttr>(cst.getValue());
}

bool P4HIR::ParserSelectCaseOp::isDefault() {
    // Return result not relying on folding, so default case might
    // be a single universal set constant or a tuple of them.
    return llvm::all_of(getSelectKeys(), isUniversalSetValue);
}

mlir::ValueRange P4HIR::ParserSelectCaseOp::getSelectKeys() {
    auto yield = mlir::cast<YieldOp>(getRegion().front().getTerminator());
    return yield.getArgs();
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

OpFoldResult P4HIR::SetOp::fold(FoldAdaptor adaptor) {
    // Fold constant inputs into set attribute
    if (llvm::any_of(adaptor.getInput(), [](Attribute attr) { return !attr; }))  // NOLINT
        return {};

    return P4HIR::SetAttr::get(getType(), SetKind::Constant,
                               mlir::ArrayAttr::get(getContext(), adaptor.getInput()));
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

void P4HIR::RangeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "range");
}

LogicalResult P4HIR::RangeOp::verify() {
    // Ranges are allowed if their direct parent is a ParserSelectCaseOp.
    // This covers the common use case in P4 select expressions.
    if (mlir::isa<P4HIR::ParserSelectCaseOp>(getOperation()->getParentOp())) {
        return mlir::success();
    }

    // However, ranges can also be used as collections in ForInOp, which means
    // their results can only be used once and their user must be a ForInOp.
    mlir::Value result = getResult();
    if (!result.hasOneUse()) {
        return emitOpError("when not nested in p4hir.select_case, ")
               << "expected single use by p4hir.foreach but found "
               << std::distance(result.user_begin(), result.user_end()) << " uses";
    }
    mlir::Operation *user = *result.user_begin();
    if (!mlir::isa<P4HIR::ForInOp>(user)) {
        return emitOpError("when not nested in p4hir.select_case, ")
               << "the user must be p4hir.foreach, but found " << user->getName();
    }

    return mlir::success();
}

OpFoldResult P4HIR::RangeOp::fold(FoldAdaptor adaptor) {
    // Fold constant inputs into set attribute
    if (adaptor.getLhs() && adaptor.getRhs())
        return P4HIR::SetAttr::get(
            getType(), SetKind::Range,
            mlir::ArrayAttr::get(getContext(), {adaptor.getLhs(), adaptor.getRhs()}));

    return {};
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

void P4HIR::MaskOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "mask");
}

OpFoldResult P4HIR::MaskOp::fold(FoldAdaptor adaptor) {
    // Fold constant inputs into set attribute
    if (adaptor.getLhs() && adaptor.getRhs())
        return P4HIR::SetAttr::get(
            getType(), SetKind::Mask,
            mlir::ArrayAttr::get(getContext(), {adaptor.getLhs(), adaptor.getRhs()}));

    return {};
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

    call_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                              /*resultAttrs=*/{}, getArgAttrsAttrName(result.name),
                                              {});
}

//===----------------------------------------------------------------------===//
// InstantiateOp, ConstructOp, SymToValueOp
//===----------------------------------------------------------------------===//

LogicalResult P4HIR::InstantiateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto ctorAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>(getCalleeAttrName());
    if (!ctorAttr) return emitOpError("requires a 'callee' symbol reference attribute");

    auto getCtorType =
        [&](mlir::FlatSymbolRefAttr ctorAttr) -> std::pair<CtorType, mlir::Operation *> {
        if (ParserOp parser =
                symbolTable.lookupSymbolIn<ParserOp>(getParentModule(*this), ctorAttr)) {
            return {parser.getCtorType(), parser.getOperation()};
        } else if (ControlOp control =
                       symbolTable.lookupSymbolIn<ControlOp>(getParentModule(*this), ctorAttr)) {
            return {control.getCtorType(), control.getOperation()};
        } else if (ExternOp ext =
                       symbolTable.lookupSymbolIn<ExternOp>(getParentModule(*this), ctorAttr)) {
            // TBD
            return {};
        } else if (PackageOp pkg =
                       symbolTable.lookupSymbolIn<PackageOp>(getParentModule(*this), ctorAttr)) {
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

        return mlir::success();
    }

    return mlir::success();

    // TBD: Handle extern ctors and turn empty ctors into error
    /* return emitOpError()
           << "'" << ctorAttr.getValue()
           << "' does not reference a valid P4 object (parser, extern, control or package)"; */
}

void P4HIR::ConstructOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getCallee());
}

LogicalResult P4HIR::ConstructOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto ctorAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>(getCalleeAttrName());
    if (!ctorAttr) return emitOpError("requires a 'callee' symbol reference attribute");

    auto getCtorType =
        [&](mlir::FlatSymbolRefAttr ctorAttr) -> std::pair<CtorType, mlir::Operation *> {
        if (ParserOp parser =
                symbolTable.lookupSymbolIn<ParserOp>(getParentModule(*this), ctorAttr)) {
            return {parser.getCtorType(), parser.getOperation()};
        } else if (ControlOp control =
                       symbolTable.lookupSymbolIn<ControlOp>(getParentModule(*this), ctorAttr)) {
            return {control.getCtorType(), control.getOperation()};
        } else if (ExternOp ext =
                       symbolTable.lookupSymbolIn<ExternOp>(getParentModule(*this), ctorAttr)) {
            // TBD
            return {};
        } else if (PackageOp pkg =
                       symbolTable.lookupSymbolIn<PackageOp>(getParentModule(*this), ctorAttr)) {
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

        return mlir::success();
    }

    return mlir::success();

    // TBD: Handle extern ctors and turn empty ctors into error
    /* return emitOpError()
           << "'" << ctorAttr.getValue()
           << "' does not reference a valid P4 object (parser, extern, control or package)"; */
}

LogicalResult P4HIR::SymToValueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the decl attribute was specified.
    auto declAttr = (*this)->getAttrOfType<SymbolRefAttr>(getDeclAttrName());
    if (!declAttr) return emitOpError("requires a 'decl' symbol reference attribute");

    auto decl = symbolTable.lookupSymbolIn(getParentModule(*this), declAttr);
    if (!decl) return emitOpError("cannot resolve symbol '") << declAttr << "' to declaration";

    return mlir::success();
}

void P4HIR::SymToValueOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), getDecl().getLeafReference());
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

LogicalResult P4HIR::ApplyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto calleeAttr = (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
    if (!calleeAttr) return emitOpError("requires a 'callee' symbol reference attribute");

    // Check that callee type corresponds to argument operands
    auto inst =
        symbolTable.lookupSymbolIn<P4HIR::InstantiateOp>(getParentModule(*this), calleeAttr);
    if (!inst)
        return emitOpError() << "'" << calleeAttr << "' does not reference a valid instantiation";

    // Lookup parser / control. Note that these never have type parameters per
    // P4 spec.
    if (inst.getTypeParameters())
        return emitOpError() << "parser or control instantiation never has type parameters";

    auto op = symbolTable.lookupSymbolIn(getParentModule(*this), inst.getCalleeAttr());
    if (!op)
        return emitOpError() << "'" << inst.getCalleeAttr()
                             << "' does not reference a valid parser or control";

    P4HIR::FuncType applyType;
    if (auto parser = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
        applyType = parser.getApplyType();
    } else if (auto control = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
        applyType = control.getApplyType();
    } else
        return emitOpError("invalid symbol definition, expected parser or control, but got ") << op;

    if (applyType.getNumInputs() != getArgOperands().size())
        return emitOpError("expected ")
               << getArgOperands().size() << " operands, but got " << applyType.getNumInputs();

    for (auto typeAndIdx : llvm::enumerate(applyType.getInputs())) {
        mlir::Type providedType = getArgOperands()[typeAndIdx.index()].getType();
        mlir::Type expectedType = typeAndIdx.value();

        if (providedType != expectedType)
            return emitOpError("operand type mismatch: expected operand type ")
                   << expectedType << ", but provided " << providedType << " for operand number "
                   << typeAndIdx.index();
    }

    return success();
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
bool P4HIR::CallMethodOp::isIndirect() { return (bool)getObj(); }

mlir::Value P4HIR::CallMethodOp::getIndirectCallee() { return getObj(); }

LogicalResult P4HIR::CallMethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto sym = (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
    if (!sym) return emitOpError("requires a 'callee' symbol reference attribute");

    // Symbol ref should be pair of extern + method name
    auto nestedRefs = sym.getNestedReferences();
    if (nestedRefs.empty()) return emitOpError("requires a nested method name in '") << sym << "'";
    auto methodSymAttr = nestedRefs.back();
    auto extSymAttr = mlir::SymbolRefAttr::get(sym.getRootReference(), nestedRefs.drop_back());

    // Extern symbol refers to instantiation here. We need to resolve it first.
    SmallVector<mlir::Type> baseTypeOperands;
    if (!isIndirect()) {
        auto inst =
            symbolTable.lookupSymbolIn<P4HIR::InstantiateOp>(getParentModule(*this), extSymAttr);
        if (!inst)
            return emitOpError() << "'" << extSymAttr
                                 << "' does not reference a valid instantiation";
        extSymAttr = mlir::SymbolRefAttr::get(inst.getCalleeAttr().getRootReference());
        if (auto typeOps = inst.getTypeParameters())
            llvm::append_range(baseTypeOperands, typeOps->getAsValueRange<mlir::TypeAttr>());
    } else {
        // First operand of indirect calls must be extern
        auto extType = mlir::dyn_cast<P4HIR::ExternType>(getObj().getType());
        if (!extType)
            return emitOpError(
                "invalid number of operands of indirect method call: should always have extern "
                "operand");
        llvm::append_range(baseTypeOperands, extType.getTypeArguments());
    }

    // Grab the extern
    P4HIR::FuncOp fn;
    P4HIR::ExternOp ext;
    if ((ext = symbolTable.lookupSymbolIn<P4HIR::ExternOp>(getParentModule(*this), extSymAttr))) {
        // Now, perform a method lookup
        auto *decl = ext.lookupSymbol(methodSymAttr);
        if ((fn = llvm::dyn_cast<P4HIR::FuncOp>(decl))) {
            // We good here
        } else if (auto ovl = llvm::dyn_cast<P4HIR::OverloadSetOp>(decl)) {
            // Find the FuncOp with the correct # of operands
            for (Operation &nestedOp : ovl.getBody().front()) {
                auto f = llvm::cast<FuncOp>(nestedOp);
                if (f.getNumArguments() == getArgOperands().size()) {
                    fn = f;
                    break;
                }
            }
            if (!fn)
                return emitOpError() << "'" << methodSymAttr << "' failed to resolve overload set";
        }
    }

    if (!ext) return emitOpError() << "'" << extSymAttr << "' does not reference a valid extern";
    if (!fn)
        return emitOpError() << "'" << methodSymAttr << "' does not reference a valid function";

    auto fnType = fn.getFunctionType();
    auto arguments = getArgOperands();

    // Verify that the operand and result types match the callee.
    if (fnType.getNumInputs() != arguments.size())
        return emitOpError("incorrect number of operands for callee");

    // Methods are never actions
    if (fn.getAction()) return emitOpError("methods cannot be actions");

    // Extern methods are always declarations
    if (!fn.isDeclaration()) return emitOpError("extern methods must be declarations");

    // Substitute type parameters:
    //   - From top-level extern
    //   - From function itself
    SmallVector<mlir::Type, 1> calleeTypeParams(fnType.getTypeArguments()), typeOperands;
    if (getTypeOperands())
        llvm::append_range(typeOperands, getTypeOperands()->getAsValueRange<mlir::TypeAttr>());
    if (calleeTypeParams.size() != typeOperands.size())
        return emitOpError() << "incorrect number of type operands for callee, expected "
                             << calleeTypeParams.size() << ", got" << typeOperands.size();

    auto extTypeParams = ext.getTypeParameters();
    if (extTypeParams) {
        if (baseTypeOperands.empty())
            return emitOpError("expected type operands to be specified for generic extern type");
        if (extTypeParams->size() != baseTypeOperands.size())
            return emitOpError() << "incorrect number of type operands for extern, expected "
                                 << extTypeParams->size() << ", got" << baseTypeOperands.size();
        llvm::append_range(calleeTypeParams, extTypeParams->getAsValueRange<mlir::TypeAttr>());
        llvm::append_range(typeOperands, baseTypeOperands);
    }

    for (auto idxAndArg : llvm::enumerate(arguments)) {
        size_t index = idxAndArg.index();
        mlir::Type expectedType =
            substituteType(fnType.getInput(index), calleeTypeParams, typeOperands);
        if (!expectedType)
            return emitOpError("cannot resolve type operand for argument number ") << index;
        mlir::Type providedType = idxAndArg.value().getType();
        if (providedType != expectedType)
            return emitOpError("operand type mismatch: expected argument type ")
                   << expectedType << ", but provided " << providedType << " for argument number "
                   << index;
    }

    // Void function must not return any results.
    if (fnType.isVoid() && getNumResults() != 0)
        return emitOpError("callee returns void but call has results");

    // Non-void function calls must return exactly one result.
    if (!fnType.isVoid()) {
        auto resultType = substituteType(fnType.getReturnType(), calleeTypeParams, typeOperands);
        if (!resultType) return emitOpError("cannot resolve type operand for result type");

        // Result type after substitution really could be void
        if (mlir::isa<P4HIR::VoidType>(resultType)) {
            if (getNumResults() != 0)
                return emitOpError("callee returns void but call has results");
        } else {
            if (getNumResults() != 1) return emitOpError("incorrect number of results for callee");

            // Parent function and return value types must match.
            if (getResultTypes().front() != resultType)
                return emitOpError("result type mismatch: expected ")
                       << resultType << ", but provided " << getResult().getType();
        }
    }

    return success();
}

mlir::StringAttr P4HIR::CallMethodOp::getMethodName() { return getCallee().getLeafReference(); }

// Callee might be:
//  - Overload set, then we need to look for a particular overload
//  - Normal methods
mlir::Operation *P4HIR::CallMethodOp::resolveCallableInTable(
    mlir::SymbolTableCollection *symbolTable) {
    auto sym = getCallee();
    if (!sym) return nullptr;

    // Grab the extern
    if (auto ext = getExtern()) {
        // Now perform a method lookup
        auto *decl = ext.lookupSymbol(getMethodName());
        if (auto fn = llvm::dyn_cast<P4HIR::FuncOp>(decl)) {
            return fn;
        } else if (auto ovl = llvm::dyn_cast<P4HIR::OverloadSetOp>(decl)) {
            // Find the FuncOp with the correct # of operands
            for (Operation &nestedOp : ovl.getBody().front()) {
                auto f = llvm::cast<FuncOp>(nestedOp);
                if (f.getNumArguments() == getArgOperands().size()) return f;
            }
        }
    }

    return nullptr;
}

P4HIR::ExternOp P4HIR::CallMethodOp::getExtern() {
    auto sym = getCallee();

    // Symbol ref should be pair of extern + method name
    auto nestedRefs = sym.getNestedReferences();
    assert(!nestedRefs.empty() && "requires a nested method name");
    auto extSymAttr = mlir::SymbolRefAttr::get(sym.getRootReference(), nestedRefs.drop_back());

    if (!isIndirect()) {
        auto inst = getParentModule(*this).lookupSymbol<P4HIR::InstantiateOp>(extSymAttr);
        assert(inst && "expected a valid instantiation");
        extSymAttr = mlir::SymbolRefAttr::get(inst.getCalleeAttr().getRootReference());
    }

    auto res = getParentModule(*this).lookupSymbol<P4HIR::ExternOp>(extSymAttr);
    assert(res && "expected valid extern reference");
    return res;
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
    call_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                              /*resultAttrs=*/{}, getArgAttrsAttrName(result.name),
                                              {});
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
    // This is essentially function_interface_impl::parseFunctionOp, but there is no control results
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
    if (function_interface_impl::parseFunctionSignatureWithArguments(
            parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Controls have no results
    if (!resultTypes.empty() || !resultAttrs.empty())
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
    call_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
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
    // Check that the callee attribute was specified.
    auto calleeAttr = (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
    if (!calleeAttr) return emitOpError("requires a 'decl' symbol reference attribute");

    auto table = symbolTable.lookupSymbolIn<P4HIR::TableOp>(getParentModule(*this), calleeAttr);
    if (!table) return emitOpError("cannot resolve symbol '") << calleeAttr << "' to a valid table";

    return mlir::success();
}

void P4HIR::TableApplyOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    llvm::SmallString<32> result(getCallee().getLeafReference());
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
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::FlatSymbolRefAttr action,
    P4HIR::FuncType cplaneType, ArrayRef<mlir::DictionaryAttr> argAttrs,
    mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Block::BlockArgListType, mlir::Location)>
        entryBuilder) {
    result.addAttribute(getCplaneTypeAttrName(result.name), TypeAttr::get(cplaneType));
    result.addAttribute(getActionAttrName(result.name), action);

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    call_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                              /*resultAttrs=*/{}, getArgAttrsAttrName(result.name),
                                              {});

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
    llvm::SmallVector<DictionaryAttr, 0> resultAttrs;
    llvm::SmallVector<Type, 8> argTypes;
    llvm::SmallVector<Type, 0> resultTypes;
    bool isVariadic = false;
    if (function_interface_impl::parseFunctionSignatureWithArguments(
            parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes, resultAttrs))
        return mlir::failure();

    // Table actions have no results
    if (!resultTypes.empty() || !resultAttrs.empty())
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
    call_interface_impl::addArgAndResultAttrs(builder, result, arguments, resultAttrs,
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
    OpBuilder &builder, OperationState &result, mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> updateBuilder) {
    OpBuilder::InsertionGuard guard(builder);

    Region *condRegion = result.addRegion();
    builder.createBlock(condRegion);
    condBuilder(builder, result.location);

    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    Region *bodyRegion = result.addRegion();
    builder.createBlock(bodyRegion);
    bodyBuilder(builder, result.location);

    Region *updateRegion = result.addRegion();
    builder.createBlock(updateRegion);
    updateBuilder(builder, result.location);
}

void P4HIR::ForOp::build(
    OpBuilder &builder, OperationState &result,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> updateBuilder) {
    build(builder, result, mlir::DictionaryAttr(), condBuilder, bodyBuilder, updateBuilder);
}

void P4HIR::ForOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                       SmallVectorImpl<mlir::RegionSuccessor> &regions) {
    // The entry into the operation is always the condition region
    if (point.isParent()) {
        regions.push_back(RegionSuccessor(&getCondRegion()));
        return;
    }

    Region *from = point.getRegionOrNull();
    assert(from && "expected non-null origin region");

    // After evaluating the loop condition:
    // - Control may enter the body if the condition is true
    // - Or exit the loop if false
    if (from == &getCondRegion()) {
        regions.push_back(RegionSuccessor(&getBodyRegion()));
        regions.push_back(RegionSuccessor());
        return;
    }

    // After executing the body, proceed to the update region
    if (from == &getBodyRegion()) {
        regions.push_back(RegionSuccessor(&getUpdatesRegion()));
        return;
    }

    // After updates, re-check the loop condition
    if (from == &getUpdatesRegion()) {
        regions.push_back(RegionSuccessor(&getCondRegion()));
        return;
    }

    llvm_unreachable("Unknown branch origin");
}

LogicalResult P4HIR::ForOp::verify() {
    Block &condBlock = getCondRegion().back();
    if (!mlir::isa<P4HIR::ConditionOp>(condBlock.back())) {
        return emitOpError("expected condition region to terminate with 'p4hir.condition'");
    }

    // TODO: What would we verify here? Simply that 'body' region has a terminator?

    Block &updatesBlock = getUpdatesRegion().back();
    if (!mlir::isa<P4HIR::YieldOp>(updatesBlock.back())) {
        return emitOpError("expected updates region to terminate with 'p4hir.yield'");
    }

    return success();
}

llvm::SmallVector<Region *> P4HIR::ForOp::getLoopRegions() { return {&getBodyRegion()}; }

//===----------------------------------------------------------------------===//
// ForInOp
//===----------------------------------------------------------------------===//

void P4HIR::ForInOp::build(
    OpBuilder &builder, OperationState &result, mlir::Value collection,
    mlir::DictionaryAttr annotations,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value, mlir::Location)> bodyBuilder) {
    result.addOperands(collection);
    if (annotations && !annotations.empty())
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);

    OpBuilder::InsertionGuard guard(builder);

    // TODO: Support different collection types
    auto collectionType = mlir::cast<P4HIR::SetType>(collection.getType());
    mlir::Type elementType = collectionType.getElementType();

    Region *region = result.addRegion();
    Block *block = builder.createBlock(region);

    mlir::BlockArgument iterationArg = block->addArgument(elementType, result.location);
    bodyBuilder(builder, iterationArg, result.location);
}

void P4HIR::ForInOp::build(
    OpBuilder &builder, OperationState &result, mlir::Value collection,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value, mlir::Location)> bodyBuilder) {
    build(builder, result, collection, DictionaryAttr(), bodyBuilder);
}

ParseResult P4HIR::ForInOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse loop iteration variable including its type
    mlir::OpAsmParser::Argument iterationArg;
    if (parser.parseArgument(iterationArg, /*allowType=*/true, /*allowAttrs=*/false))
        return parser.emitError(parser.getNameLoc(),
                                "expected iteration variable argument ('%var : type')");

    if (parser.parseKeyword("in")) return failure();

    // Parse collection operand and its type
    OpAsmParser::UnresolvedOperand collection;
    Type collectionType;
    if (parser.parseOperand(collection) || parser.parseColonType(collectionType))
        return parser.emitError(parser.getNameLoc(),
                                "expected collection operand ('%collection : type')");
    if (parser.resolveOperand(collection, collectionType, result.operands)) return failure();

    // Verify that the collection type is iterable and determine element type
    Type expectedElementType;
    if (auto setType = mlir::dyn_cast<P4HIR::SetType>(collectionType)) {
        expectedElementType = setType.getElementType();
        // TODO: Add support for other collection types like arrays
    } else {
        return parser.emitError(loc, "expected an iterable collection type, found")
               << collectionType;
    }
    if (iterationArg.type != expectedElementType) {
        return parser.emitError(loc, "loop variable type (")
               << iterationArg.type << ") does not match element type of collection ("
               << expectedElementType << ")";
    }

    // Parse optional annotations
    mlir::DictionaryAttr annotations;
    if (::mlir::succeeded(parser.parseOptionalKeyword("annotations"))) {
        if (parser.parseAttribute<mlir::DictionaryAttr>(annotations)) return failure();
        result.addAttribute(getAnnotationsAttrName(result.name), annotations);
    }

    // Parse the loop body region, passing the iteration variable as a block argument
    Region *bodyRegion = result.addRegion();
    SmallVector<OpAsmParser::Argument, 1> regionArgs = {iterationArg};
    if (parser.parseRegion(*bodyRegion, regionArgs, /*enableNameShadowing=*/false))
        return failure();

    return success();
}

void P4HIR::ForInOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOperand(getRegion().getArgument(0));
    printer << " : ";
    printer.printType(getRegion().getArgument(0).getType());

    printer << " in " << getCollection() << " : ";
    printer.printType(getCollection().getType());

    if (auto ann = getAnnotations(); ann && !ann->empty()) {
        printer << " annotations ";
        printer.printAttributeWithoutType(*ann);
    }
    printer << ' ';
    printer.printRegion(getBodyRegion(),
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
}

void P4HIR::ForInOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                         SmallVectorImpl<RegionSuccessor> &regions) {
    regions.push_back(RegionSuccessor(&getBodyRegion()));
    regions.push_back(RegionSuccessor());
}

llvm::SmallVector<Region *> P4HIR::ForInOp::getLoopRegions() { return {&getBodyRegion()}; }

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

mlir::MutableOperandRange P4HIR::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
    auto parent = mlir::cast<P4HIR::ForOp>(getOperation()->getParentOp());
    assert((point.isParent() || point.getRegionOrNull() == &parent.getBodyRegion()) &&
           "condition op can only exit the loop or branch to the body region");

    // No values are yielded to the successor region
    return mlir::MutableOperandRange(getOperation(), 0, 0);
}

//===----------------------------------------------------------------------===//
// UninitializedOp
//===----------------------------------------------------------------------===//

void P4HIR::UninitializedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    setNameFn(getResult(), "uninitialized");
}

//===----------------------------------------------------------------------===//
// ArrayOp
//===----------------------------------------------------------------------===//

ParseResult P4HIR::ArrayOp::parse(OpAsmParser &parser, OperationState &result) {
    llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    Type declType;

    if (parser.parseLSquare() || parser.parseOperandList(operands) || parser.parseRSquare() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(declType))
        return failure();

    auto arrayType = mlir::dyn_cast<ArrayType>(declType);
    if (!arrayType) return parser.emitError(parser.getNameLoc(), "expected !p4hir.array type");

    llvm::SmallVector<Type, 4> arrayInnerTypes(arrayType.getSize(), arrayType.getElementType());
    result.addTypes(arrayType);

    if (parser.resolveOperands(operands, arrayInnerTypes, inputOperandsLoc, result.operands))
        return failure();
    return success();
}

void P4HIR::ArrayOp::print(OpAsmPrinter &printer) {
    printer << " [";
    printer.printOperands(getInput());
    printer << "]";
    printer.printOptionalAttrDict((*this)->getAttrs());
    printer << " : " << getType();
}

LogicalResult P4HIR::ArrayOp::verify() {
    auto arrayType = mlir::cast<ArrayType>(getType());

    if (arrayType.getSize() != getInput().size())
        return emitOpError("array element count mismatch");

    for (auto value : getInput())
        if (arrayType.getElementType() != value.getType())
            return emitOpError("value `") << value << "` type does not match array element type";

    return success();
}

void P4HIR::ArrayOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "array");
}

OpFoldResult P4HIR::ArrayGetOp::fold(FoldAdaptor adaptor) {
    // We can only fold constant indices
    auto idxAttr = mlir::dyn_cast_if_present<P4HIR::IntAttr>(adaptor.getIndex());
    if (!idxAttr) return {};

    // Fold extract from aggregate constant
    if (auto aggAttr = adaptor.getInput())
        return mlir::cast<P4HIR::AggAttr>(aggAttr).getFields()[idxAttr.getUInt()];

    // Fold extract from array
    if (auto arrayOp = mlir::dyn_cast_if_present<P4HIR::ArrayOp>(getInput().getDefiningOp()))
        return arrayOp.getOperand(idxAttr.getUInt());

    return {};
}

LogicalResult P4HIR::ArrayGetOp::canonicalize(P4HIR::ArrayGetOp op, PatternRewriter &rewriter) {
    // Simple SROA / load shrinking: turn (array_get (read ref), idx)
    // into (read (array_element_ref ref, idx)) if `read` operation has a
    // single use. Usually these come from header stack field access and it is
    // beneficial to project from whole-width read to a single-field read. We do
    // not do complete SROA here as it would require tracking writes as well as
    // reads.
    if (auto readOp = op.getInput().getDefiningOp<P4HIR::ReadOp>(); readOp && readOp->hasOneUse()) {
        OpBuilder::InsertionGuard guard(rewriter);
        auto ref = readOp.getRef();
        rewriter.setInsertionPoint(readOp);
        auto eltRef = rewriter.create<P4HIR::ArrayElementRefOp>(
            op.getLoc(), P4HIR::ReferenceType::get(op.getType()), ref, op.getIndex());
        rewriter.replaceOpWithNewOp<P4HIR::ReadOp>(op, eltRef);
        rewriter.eraseOp(readOp);
        return success();
    }

    return failure();
}

void P4HIR::ArrayGetOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "array_elt");
}

void P4HIR::ArrayElementRefOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
    setNameFn(getResult(), "elt_ref");
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands P4HIR::BrOp::getSuccessorOperands(unsigned index) {
    assert(index == 0 && "invalid successor index");
    return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *P4HIR::BrOp::getSuccessorForOperands(ArrayRef<Attribute>) { return getDest(); }

//===----------------------------------------------------------------------===//
// CondBrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands P4HIR::CondBrOp::getSuccessorOperands(unsigned index) {
    assert(index < getNumSuccessors() && "invalid successor index");
    return SuccessorOperands(index == 0 ? getDestOperandsTrueMutable()
                                        : getDestOperandsFalseMutable());
}

Block *P4HIR::CondBrOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
    if (IntegerAttr condAttr = dyn_cast_if_present<IntegerAttr>(operands.front()))
        return condAttr.getValue().isOne() ? getDestTrue() : getDestFalse();
    return nullptr;
}

namespace {
struct P4HIROpAsmDialectInterface : public OpAsmDialectInterface {
    using OpAsmDialectInterface::OpAsmDialectInterface;

    AliasResult getAlias(Type type, raw_ostream &os) const final {
        return mlir::TypeSwitch<Type, AliasResult>(type)
            .Case<P4HIR::InfIntType, P4HIR::BitsType, P4HIR::ValidBitType, P4HIR::VoidType,
                  P4HIR::ErrorType, P4HIR::StringType>([&](auto type) {
                os << type.getAlias();
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::StructType, P4HIR::HeaderType, P4HIR::HeaderUnionType, P4HIR::SerEnumType,
                  P4HIR::AliasType>([&](auto type) {
                os << type.getName();
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::ParserType, P4HIR::ControlType, P4HIR::ExternType, P4HIR::PackageType>(
                [&](auto type) {
                    os << type.getName();
                    for (auto typeArg : type.getTypeArguments()) {
                        os << "_";
                        getAlias(typeArg, os);
                    }
                    return AliasResult::OverridableAlias;
                })
            .Case<P4HIR::EnumType>([&](auto type) {
                auto name = type.getName();
                os << (name.empty() ? "anon" : name);
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::TypeVarType>([&](auto type) {
                os << "type_" << type.getName();
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::CtorType>([&](auto type) {
                os << "ctor_";
                getAlias(type.getReturnType(), os);
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::ArrayType>([&](auto type) {
                os << "arr_" << type.getSize() << "x";
                getAlias(type.getElementType(), os);
                return AliasResult::OverridableAlias;
            })
            .Case<P4HIR::HeaderStackType>([&](auto type) {
                os << "hs_" << type.getArraySize() << "x";
                getAlias(type.getArrayElementType(), os);
                return AliasResult::OverridableAlias;
            })
            .Default([](Type) { return AliasResult::NoAlias; });
    }

    AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
        return mlir::TypeSwitch<Attribute, AliasResult>(attr)
            .Case<P4HIR::BoolAttr>([&](auto boolAttr) {
                os << (boolAttr.getValue() ? "true" : "false");
                if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(boolAttr.getType()))
                    os << "_" << aliasType.getName();

                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::IntAttr>([&](auto intAttr) {
                os << "int" << intAttr.getValue();
                if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(intAttr.getType()))
                    os << "_" << bitsType.getAlias();
                else if (auto infintType = mlir::dyn_cast<P4HIR::InfIntType>(intAttr.getType()))
                    os << "_" << infintType.getAlias();
                else if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(intAttr.getType()))
                    os << "_" << aliasType.getName();

                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::ParamDirectionAttr, P4HIR::ValidityBitAttr>([&](auto attr) {
                os << stringifyEnum(attr.getValue());
                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::ErrorCodeAttr>([&](auto errorAttr) {
                os << "error_" << errorAttr.getField().getValue();
                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::EnumFieldAttr>([&](auto enumFieldAttr) {
                if (auto enumType = mlir::dyn_cast<P4HIR::EnumType>(enumFieldAttr.getType()))
                    os << (enumType.getName().empty() ? "anon" : enumType.getName()) << "_"
                       << enumFieldAttr.getField().getValue();
                else
                    os << mlir::cast<P4HIR::SerEnumType>(enumFieldAttr.getType()).getName() << "_"
                       << enumFieldAttr.getField().getValue();

                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::CtorParamAttr>([&](auto ctorParamAttr) {
                os << ctorParamAttr.getParent().getRootReference().getValue() << "_"
                   << ctorParamAttr.getName().getValue();
                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::MatchKindAttr>([&](auto matchKindAttr) {
                os << matchKindAttr.getValue().getValue();
                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::UniversalSetAttr>([&](auto) {
                os << "everything";
                return AliasResult::FinalAlias;
            })
            .Case<P4HIR::SetAttr>([&](auto setAttr) {
                if (setAttr.getMembers().size() > 2)
                    return AliasResult::NoAlias;  // or it will be too long

                os << "set_" << stringifyEnum(setAttr.getKind()) << "_of";
                for (auto attr : setAttr.getMembers()) {
                    os << "_";
                    getAlias(attr, os);
                }

                return AliasResult::FinalAlias;
            })

            .Default([](Attribute) { return AliasResult::NoAlias; });
        return AliasResult::NoAlias;
    }
};

/// This class defines the interface for handling inlining with P4HIR operations.
struct P4HIRInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    // All call operations can be inlined.
    bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final {
        if (mlir::isa<P4HIR::CallOp>(call) &&
            mlir::isa<P4HIR::FuncOp, P4HIR::OverloadSetOp>(callable))
            return true;

        return false;
    }
    // All operations can be inlined.
    // TODO: Actually not, but we are protected by that isLegalToInline check above
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
    /// All regions can be inlined.
    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final { return true; }

    // Handle the given inlined terminator by replacing it with a new operation
    // as necessary.
    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        // Only return needs to be handled here.
        auto returnOp = mlir::cast<P4HIR::ReturnOp>(op);
        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
            from.replaceAllUsesWith(to);
    }
};
}  // namespace

Operation *P4HIR::P4HIRDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                                    Location loc) {
    auto typedAttr = mlir::cast<mlir::TypedAttr>(value);
    assert(typedAttr.getType() == type && "type mismatch");
    return builder.create<P4HIR::ConstOp>(loc, typedAttr);
}

void P4HIR::P4HIRDialect::initialize() {
    registerTypes();
    registerAttributes();
    addOperations<
#define GET_OP_LIST
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
        >();
    addInterfaces<P4HIROpAsmDialectInterface>();
    addInterfaces<P4HIRInlinerInterface>();
}

#define GET_OP_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.cpp.inc"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.cpp.inc"  // NOLINT
#include "p4mlir/Dialect/P4HIR/P4HIR_OpsEnums.cpp.inc"
