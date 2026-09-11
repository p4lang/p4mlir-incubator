// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_MATCHERS_H
#define P4MLIR_DIALECT_P4HIR_MATCHERS_H

#include <functional>

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace detail {

template <typename OpType, typename... OperandMatchers>
struct RecursivePredicatedPatternMatcher {
    RecursivePredicatedPatternMatcher(std::function<bool(OpType)> pred, OperandMatchers... matchers)
        : pred(pred), operandMatchers(matchers...) {}
    bool match(mlir::Operation *op) {
        if (!mlir::isa<OpType>(op) || op->getNumOperands() != sizeof...(OperandMatchers))
            return false;
        if (!pred(mlir::cast<OpType>(op))) return false;
        bool res = true;
        enumerate(operandMatchers, [&](size_t index, auto &matcher) {
            res &= mlir::detail::matchOperandOrValueAtIndex(op, index, matcher);
        });
        return res;
    }

    std::function<bool(OpType)> pred;
    std::tuple<OperandMatchers...> operandMatchers;
};

template <typename OpType, typename... Matchers>
inline auto m_OpWithPred(std::function<bool(OpType)> pred, Matchers... matchers) {
    return RecursivePredicatedPatternMatcher<OpType, Matchers...>(pred, matchers...);
}

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<
    llvm::is_detected<mlir::detail::has_compatible_matcher_t, MatcherClass, mlir::Value>::value,
    bool>
matchOpOrValue(mlir::Value val, MatcherClass &matcher) {
    return matcher.match(val);
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<mlir::detail::has_compatible_matcher_t, MatcherClass,
                                   mlir::Operation *>::value,
                 bool>
matchOpOrValue(mlir::Value val, MatcherClass &matcher) {
    if (auto *defOp = val.getDefiningOp()) return matcher.match(defOp);
    return false;
}

template <typename OperandMatcher>
struct RecursivePredicatedIntegerCastPatternMatcher {
    using BitsType = P4::P4MLIR::P4HIR::BitsType;
    RecursivePredicatedIntegerCastPatternMatcher(std::function<bool(BitsType, BitsType)> pred,
                                                 OperandMatcher matcher, bool optional)
        : pred(pred), operandMatcher(matcher), optional(optional) {}
    bool match(mlir::Value val) {
        auto castOp = val.getDefiningOp<P4::P4MLIR::P4HIR::CastOp>();
        if (!castOp) {
            if (optional)
                return matchOpOrValue(val, operandMatcher);
            else
                return false;
        }

        auto fromType = mlir::dyn_cast<BitsType>(castOp.getSrc().getType());
        auto toType = mlir::dyn_cast<BitsType>(castOp.getType());
        if (!fromType || !toType || !pred(fromType, toType)) return false;
        return matchOperandOrValueAtIndex(castOp, 0, operandMatcher);
    }

    std::function<bool(BitsType, BitsType)> pred;
    OperandMatcher operandMatcher;
    bool optional;
};

template <typename Matcher>
inline auto m_IntegerCastOpWithPred(
    std::function<bool(P4::P4MLIR::P4HIR::BitsType, P4::P4MLIR::P4HIR::BitsType)> pred,
    Matcher matcher, bool optional) {
    return RecursivePredicatedIntegerCastPatternMatcher<Matcher>(pred, matcher, optional);
}

template <typename Matcher>
inline auto m_IntegerExt(Matcher matcher, bool isSigned, bool optional) {
    return m_IntegerCastOpWithPred(
        [isSigned](auto fromType, auto toType) {
            return fromType.isSigned() == isSigned && toType.isSigned() == isSigned &&
                   fromType.getWidth() < toType.getWidth();
        },
        matcher, optional);
}

struct ConstantIntBinder {
    enum Kind : unsigned {
        KindMatchOnly = 0,
        KindBindAPSInt = 1,
        KindBindInt = 2,
        KindMatchAPSint = 3,
        KindMatchInt = 4,
    };

    llvm::APSInt *bindValue = nullptr;
    llvm::APSInt matchValue;
    unsigned *bindValueInt = nullptr;
    unsigned matchValueInt = 0;
    unsigned kind;
    bool matchBool;

    ConstantIntBinder(bool matchBool)
        : bindValue(nullptr), bindValueInt(nullptr), kind(KindMatchOnly), matchBool(matchBool) {}

    ConstantIntBinder(llvm::APSInt *bindValue, bool matchBool)
        : bindValue(bindValue), kind(KindBindAPSInt), matchBool(matchBool) {}

    ConstantIntBinder(unsigned *bindValueInt, bool matchBool)
        : bindValueInt(bindValueInt), kind(KindBindInt), matchBool(matchBool) {}

    ConstantIntBinder(llvm::APSInt matchValue, bool matchBool)
        : matchValue(matchValue), kind(KindMatchAPSint), matchBool(matchBool) {}

    ConstantIntBinder(unsigned matchValueInt, bool matchBool)
        : matchValueInt(matchValueInt), kind(KindMatchInt), matchBool(matchBool) {}

    bool match(mlir::Operation *op) {
        mlir::Attribute attr;
        if (!matchPattern(op, m_Constant(&attr))) return false;
        auto val = P4::P4MLIR::P4HIR::getConstantInt(attr);
        if (!val || (mlir::isa<P4::P4MLIR::P4HIR::BoolAttr>(attr) && !matchBool)) return false;

        switch (kind) {
            case KindMatchOnly: {
                return true;
            }
            case KindBindAPSInt: {
                assert(bindValue && "Expected valid pointer");
                *bindValue = *val;
                return true;
            }
            case KindBindInt: {
                assert(bindValueInt && "Expected valid pointer");
                if (!val->isIntN(32)) return false;
                *bindValueInt = val->getLimitedValue();
                return true;
            }
            case KindMatchAPSint: {
                return val == matchValue;
            }
            case KindMatchInt: {
                if (!val->isIntN(32)) return false;
                return val->getLimitedValue() == matchValueInt;
            }
            default: {
                llvm_unreachable("Impossible kind");
                return false;
            }
        }
    }
};

}  // namespace detail

inline auto m_ConstantInt(unsigned *bindValue, bool matchBool = false) {
    return detail::ConstantIntBinder(bindValue, matchBool);
}

inline auto m_ConstantInt(llvm::APSInt *bindValue, bool matchBool = false) {
    return detail::ConstantIntBinder(bindValue, matchBool);
}

inline auto m_ConstantInt(bool matchBool = true) { return detail::ConstantIntBinder(matchBool); }

inline auto m_ConstantInt(unsigned matchValue, bool matchBool = false) {
    return detail::ConstantIntBinder(matchValue, matchBool);
}

inline auto m_ConstantInt(llvm::APSInt matchValue, bool matchBool = false) {
    return detail::ConstantIntBinder(matchValue, matchBool);
}

inline auto m_ZeroInt(bool matchBool = false) { return m_ConstantInt((unsigned)0, matchBool); }

template <typename Matcher>
inline auto m_UnaryOp(P4::P4MLIR::P4HIR::UnaryOpKind kind, Matcher matcher) {
    return detail::m_OpWithPred<P4::P4MLIR::P4HIR::UnaryOp>(
        [kind](P4::P4MLIR::P4HIR::UnaryOp op) { return op.getKind() == kind; }, matcher);
}

template <typename LhsMatcher, typename RhsMatcher>
inline auto m_BinaryOp(P4::P4MLIR::P4HIR::BinOpKind kind, LhsMatcher lhsMatcher,
                       RhsMatcher rhsMatcher) {
    return detail::m_OpWithPred<P4::P4MLIR::P4HIR::BinOp>(
        [kind](P4::P4MLIR::P4HIR::BinOp op) { return op.getKind() == kind; }, lhsMatcher,
        rhsMatcher);
}

template <typename LhsMatcher, typename RhsMatcher>
inline auto m_ShlOp(LhsMatcher lhsMatcher, RhsMatcher rhsMatcher) {
    return mlir::m_Op<P4::P4MLIR::P4HIR::ShlOp>(lhsMatcher, rhsMatcher);
}

template <typename LhsMatcher, typename RhsMatcher>
inline auto m_ShrOp(LhsMatcher lhsMatcher, RhsMatcher rhsMatcher) {
    return mlir::m_Op<P4::P4MLIR::P4HIR::ShrOp>(lhsMatcher, rhsMatcher);
}

template <typename Matcher>
inline auto m_SignExt(Matcher matcher, bool optional = false) {
    return detail::m_IntegerExt(matcher, true, optional);
}

template <typename Matcher>
inline auto m_MaybeSignExt(Matcher matcher) {
    return m_SignExt(matcher, true);
}

template <typename Matcher>
inline auto m_ZeroExt(Matcher matcher, bool optional = false) {
    return detail::m_IntegerExt(matcher, false, optional);
}

template <typename Matcher>
inline auto m_MaybeZeroExt(Matcher matcher) {
    return m_ZeroExt(matcher, true);
}

#endif  // P4MLIR_DIALECT_P4HIR_MATCHERS_H
