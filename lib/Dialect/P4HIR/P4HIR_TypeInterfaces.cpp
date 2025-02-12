#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"

using namespace mlir;
using namespace P4::P4MLIR::P4HIR;
using namespace FieldIdImpl;

mlir::Type FieldIdImpl::getFinalTypeByFieldID(mlir::Type type, unsigned fieldID) {
    std::pair<Type, unsigned> pair(type, fieldID);
    while (pair.second) {
        if (auto ftype = dyn_cast<FieldIDTypeInterface>(pair.first))
            pair = ftype.getSubTypeByFieldID(pair.second);
        else
            llvm::report_fatal_error("fieldID indexing into a non-aggregate type");
    }
    return pair.first;
}

std::pair<Type, unsigned> FieldIdImpl::getSubTypeByFieldID(mlir::Type type, unsigned fieldID) {
    if (!fieldID) return {type, 0};
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
        return ftype.getSubTypeByFieldID(fieldID);

    llvm::report_fatal_error("fieldID indexing into a non-aggregate type");
}

unsigned FieldIdImpl::getMaxFieldID(mlir::Type type) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type)) return ftype.getMaxFieldID();
    return 0;
}

std::pair<unsigned, bool> FieldIdImpl::projectToChildFieldID(mlir::Type type, unsigned fieldID,
                                                             unsigned index) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
        return ftype.projectToChildFieldID(fieldID, index);
    return {0, fieldID == 0};
}

unsigned FieldIdImpl::getIndexForFieldID(mlir::Type type, unsigned fieldID) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type)) return ftype.getIndexForFieldID(fieldID);
    return 0;
}

unsigned FieldIdImpl::getFieldID(mlir::Type type, unsigned fieldID) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type)) return ftype.getFieldID(fieldID);
    return 0;
}

std::pair<unsigned, unsigned> FieldIdImpl::getIndexAndSubfieldID(mlir::Type type,
                                                                 unsigned fieldID) {
    if (auto ftype = dyn_cast<FieldIDTypeInterface>(type))
        return ftype.getIndexAndSubfieldID(fieldID);
    return {0, fieldID == 0};
}

#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.cpp.inc"
