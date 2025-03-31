#ifndef LIB_UTILITIES_INTERLEAVE_WITH_ERROR_H_
#define LIB_UTILITIES_INTERLEAVE_WITH_ERROR_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/Support/LLVM.h"
#pragma GCC diagnostic pop

namespace P4::P4MLIR::Utilities {

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
/// Copied from
/// https://github.com/llvm/llvm-project/blob/30013872190ca05eb00333adb989c9f74b1cf3ac/mlir/lib/Target/Cpp/TranslateToCpp.cpp#L36.
template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline mlir::LogicalResult interleaveWithError(ForwardIterator begin, ForwardIterator end,
                                               UnaryFunctor eachFn, NullaryFunctor betweenFn) {
    if (begin == end) return mlir::success();
    if (failed(eachFn(*begin))) return mlir::failure();
    ++begin;
    for (; begin != end; ++begin) {
        betweenFn();
        if (failed(eachFn(*begin))) return mlir::failure();
    }
    return mlir::success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline mlir::LogicalResult interleaveWithError(const Container &c, UnaryFunctor eachFn,
                                               NullaryFunctor betweenFn) {
    return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename StreamT, typename UnaryFunctor>
inline mlir::LogicalResult interleaveCommaWithError(const Container &c, StreamT &os,
                                                    UnaryFunctor eachFn) {
    return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_INTERLEAVE_WITH_ERROR_H_
