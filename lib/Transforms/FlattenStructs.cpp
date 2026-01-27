#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_FLATTENSTRUCTS
#include "p4mlir/Transforms/Passes.cpp.inc"
namespace {

// This TypeConverter flattens structs and headers, using `_` as a seperator for the new field
// names, e.g. From: !ingress_metadata_t = !p4hir.struct<"ingress_metadata_t", vrf: !b12i, bd:
// !b16i, nexthop_index: !b16i> !metadata = !p4hir.struct<"metadata", ingress_metadata:
// !ingress_metadata_t> To: !metadata = !p4hir.struct<"metadata", ingress_metadata_vrf: !b12i,
// ingress_metadata_bd: !b16i, ingress_metadata_nexthop_index: !b16i> Structs are flattened into
// headers but headers aren't flattened into structs
class StructFlatteningTypeConverter : public P4HIRTypeConverter {
 public:
    StructFlatteningTypeConverter() {
        addConversion([](P4HIR::StructLikeTypeInterface type) -> std::optional<Type> {
            return flattenStructLikeType(type);
        });
    }

    static bool isHeader(Type type) {
        return isa<P4HIR::HeaderType, P4HIR::HeaderUnionType, P4HIR::HeaderStackType>(type);
    }

    static bool isScalar(Type type) {
        return isa<P4HIR::BitsType, P4HIR::VarBitsType, P4HIR::BoolType, P4HIR::ErrorType,
                   P4HIR::EnumType>(type);
    }

 private:
    static P4HIR::StructLikeTypeInterface flattenStructLikeType(
        P4HIR::StructLikeTypeInterface structLikeType) {
        if (auto headerStack = dyn_cast<P4HIR::HeaderStackType>(structLikeType)) {
            auto flattened = flattenStructLikeType(headerStack.getArrayElementType());
            return P4HIR::HeaderStackType::get(headerStack.getContext(), headerStack.getArraySize(),
                                               flattened);
        }

        SmallVector<P4HIR::FieldInfo> flattenedFields;

        for (auto [fieldName, fieldType, annotations] : structLikeType.getFields()) {
            if (auto nestedStruct = dyn_cast<P4HIR::StructType>(fieldType)) {
                auto flattened = flattenStructLikeType(nestedStruct);

                for (auto [nestedName, nestedType, annotations] : flattened.getFields()) {
                    std::string newName =
                        (fieldName.getValue() + "_" + nestedName.getValue()).str();
                    auto newNameAttr = StringAttr::get(structLikeType.getContext(), newName);
                    flattenedFields.push_back({newNameAttr, nestedType, annotations});
                }
            } else if (isHeader(fieldType)) {
                auto flattened =
                    flattenStructLikeType(cast<P4HIR::StructLikeTypeInterface>(fieldType));
                flattenedFields.push_back({fieldName, cast<Type>(flattened), annotations});
            } else if (isScalar(fieldType)) {
                flattenedFields.push_back({fieldName, fieldType, annotations});
            } else if (isa<P4HIR::ValidBitType>(fieldType)) {
                continue;
            } else {
                llvm::errs() << "Unexpected type " << fieldType << "\n";
                llvm_unreachable("Unexpected field type during flattening");
            }
        }

        return llvm::TypeSwitch<P4HIR::StructLikeTypeInterface, P4HIR::StructLikeTypeInterface>(
                   structLikeType)
            .Case([&](P4HIR::StructType structType) {
                return P4HIR::StructType::get(structType.getContext(), structType.getName(),
                                              flattenedFields, structType.getAnnotations());
            })
            .Case([&](P4HIR::HeaderUnionType headerUnionType) {
                return P4HIR::HeaderUnionType::get(headerUnionType.getContext(),
                                                   headerUnionType.getName(), flattenedFields,
                                                   headerUnionType.getAnnotations());
            })
            .Case([&](P4HIR::HeaderType headerType) {
                return P4HIR::HeaderType::get(headerType.getContext(), headerType.getName(),
                                              flattenedFields, headerType.getAnnotations());
            });
    }
};

// This pattern handles StructFieldRefOps and StructExtractOps, it traverses the tree of
// struct accesses, constructs the new field names as it goes (using `_` as separator) and
// replaces the "leaf" accesses with new ones that use the new field names.
template <typename OpTy>
class FlattenStructAccess : public OpConversionPattern<OpTy> {
 public:
    using OpConversionPattern<OpTy>::OpConversionPattern;
    using AdaptorTy = typename OpConversionPattern<OpTy>::OpAdaptor;

    LogicalResult matchAndRewrite(OpTy op, AdaptorTy adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Operation *> eraseList;
        DenseMap<Operation *, Value> replacements;

        if (failed(processStructAccessTree(op.getOperation(), adaptor.getInput(), "", eraseList,
                                           replacements, rewriter)))
            return failure();

        for (auto [oldOp, newValue] : replacements) {
            rewriter.replaceOp(oldOp, newValue);
        }

        for (auto *opToErase : eraseList) {
            rewriter.eraseOp(opToErase);
        }

        return success();
    }

 private:
    LogicalResult processStructAccessTree(Operation *op, Value convertedInput,
                                          StringRef parentFieldPath,
                                          SmallVector<Operation *> &eraseList,
                                          DenseMap<Operation *, Value> &replacements,
                                          ConversionPatternRewriter &rewriter) const {
        return llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](P4HIR::StructFieldRefOp fieldRefOp) {
                return processStructAccessOp(fieldRefOp, convertedInput, parentFieldPath, eraseList,
                                             replacements, rewriter);
            })
            .Case([&](P4HIR::StructExtractOp extractOp) {
                return processStructAccessOp(extractOp, convertedInput, parentFieldPath, eraseList,
                                             replacements, rewriter);
            })
            .Case([&](P4HIR::ReadOp readOp) {
                return processRead(readOp, convertedInput, parentFieldPath, eraseList, replacements,
                                   rewriter);
            })
            .Default([](Operation *op) {
                return op->emitError("Unexpected operation in struct access tree");
            });
    }

    // Helper function for StructFieldRefOp and StructExtractOp
    template <typename StructAccessOpTy>
    LogicalResult processStructAccessOp(StructAccessOpTy op, Value convertedInput,
                                        StringRef parentFieldPath,
                                        SmallVector<Operation *> &eraseList,
                                        DenseMap<Operation *, Value> &replacements,
                                        ConversionPatternRewriter &rewriter) const {
        std::string currentFieldPath = parentFieldPath.empty()
                                           ? op.getFieldName().str()
                                           : (parentFieldPath + "_" + op.getFieldName()).str();

        auto resultTy = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultTy) return op.emitError("Unable to convert result type");

        auto isPredicateOrRefToPredicate = [](Type ty,
                                              llvm::function_ref<bool(Type)> pred) -> bool {
            if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty)) return pred(refTy.getObjectType());
            return pred(ty);
        };

        bool isLeaf =
            isPredicateOrRefToPredicate(resultTy, StructFlatteningTypeConverter::isHeader) ||
            isPredicateOrRefToPredicate(resultTy, StructFlatteningTypeConverter::isScalar) ||
            isPredicateOrRefToPredicate(resultTy,
                                        [](Type ty) { return isa<P4HIR::ValidBitType>(ty); });

        // If this is the leaf of a chain of struct accesses, we create the final replacement
        // (we have to be careful and use FieldRef/Extract/Read based on whether the input and
        // output are references or not)
        if (isLeaf) {
            bool outputIsRef = isa<P4HIR::ReferenceType>(op.getResult().getType());
            Value newResult =
                llvm::TypeSwitch<Type, Value>(convertedInput.getType())
                    .Case<P4HIR::ReferenceType>([&](auto) {
                        Value res = rewriter
                                        .create<P4HIR::StructFieldRefOp>(
                                            op.getLoc(), convertedInput, currentFieldPath)
                                        .getResult();
                        if (!outputIsRef)
                            res = rewriter.create<P4HIR::ReadOp>(op.getLoc(), res).getResult();
                        return res;
                    })
                    .Default([&](auto) {
                        return rewriter
                            .create<P4HIR::StructExtractOp>(op.getLoc(), convertedInput,
                                                            currentFieldPath)
                            .getResult();
                    });

            replacements[op.getOperation()] = newResult;
            return success();
        }

        // This is not a leaf, process the rest of the tree and add the op the erase list
        for (auto user : op.getResult().getUsers()) {
            if (!isa<P4HIR::StructFieldRefOp, P4HIR::StructExtractOp, P4HIR::ReadOp>(user)) {
                return user->emitError(
                    "Expected struct access or read operation as user of intermediate struct "
                    "access");
            }

            if (failed(processStructAccessTree(user, convertedInput, currentFieldPath, eraseList,
                                               replacements, rewriter)))
                return failure();
        }

        eraseList.push_back(op);
        return success();
    }

    LogicalResult processRead(P4HIR::ReadOp op, Value convertedInput, StringRef parentFieldPath,
                              SmallVector<Operation *> &eraseList,
                              DenseMap<Operation *, Value> &replacements,
                              ConversionPatternRewriter &rewriter) const {
        // For intermediate reads we just process the rest of the tree and add the read to the erase
        // list
        for (auto user : op.getResult().getUsers()) {
            if (!isa<P4HIR::StructFieldRefOp, P4HIR::StructExtractOp>(user)) {
                return user->emitError("Expected struct access operation as user of read");
            }

            if (failed(processStructAccessTree(user, convertedInput, parentFieldPath, eraseList,
                                               replacements, rewriter)))
                return failure();
        }

        eraseList.push_back(op);
        return success();
    }
};

struct StructFlatteningPass : public P4::P4MLIR::impl::FlattenStructsBase<StructFlatteningPass> {
    void runOnOperation() override {
        auto moduleOp = getOperation();

        StructFlatteningTypeConverter typeConverter;

        ConversionTarget target(getContext());
        target.addLegalDialect<P4HIR::P4HIRDialect>();

        target.addDynamicallyLegalOp<P4HIR::StructFieldRefOp>([&](P4HIR::StructFieldRefOp op) {
            return typeConverter.isLegal(op.getInput().getType());
        });

        target.addDynamicallyLegalOp<P4HIR::StructExtractOp>([&](P4HIR::StructExtractOp op) {
            return typeConverter.isLegal(op.getInput().getType());
        });

        target.addDynamicallyLegalOp<P4HIR::FuncOp>(
            [&](P4HIR::FuncOp op) { return typeConverter.isLegal(op.getFunctionType()); });

        target.addDynamicallyLegalOp<P4HIR::TableKeyOp>(
            [&](P4HIR::TableKeyOp op) { return typeConverter.isLegal(op.getApplyType()); });

        target.addDynamicallyLegalOp<P4HIR::ControlOp>(
            [&](P4HIR::ControlOp op) { return typeConverter.isLegal(op.getApplyType()); });

        target.addDynamicallyLegalOp<P4HIR::ParserOp>(
            [&](P4HIR::ParserOp op) { return typeConverter.isLegal(op.getApplyType()); });

        target.addDynamicallyLegalOp<P4HIR::CallOp>([&](P4HIR::CallOp op) {
            return typeConverter.isLegal(op.getResultTypes()) &&
                   typeConverter.isLegal(op.getOperandTypes());
        });

        target.addDynamicallyLegalOp<P4HIR::CallMethodOp>([&](P4HIR::CallMethodOp op) {
            return typeConverter.isLegal(op.getResultTypes()) &&
                   typeConverter.isLegal(op.getOperandTypes());
        });

        target.addDynamicallyLegalOp<P4HIR::TableApplyOp>(
            [&](P4HIR::TableApplyOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

        target.addDynamicallyLegalOp<P4HIR::ConstructOp>(
            [&](P4HIR::ConstructOp op) { return typeConverter.isLegal(op.getType()); });

        target.addDynamicallyLegalOp<P4HIR::VariableOp>(
            [&](P4HIR::VariableOp op) { return typeConverter.isLegal(op.getType()); });

        target.addDynamicallyLegalOp<P4HIR::ControlLocalOp>(
            [&](P4HIR::ControlLocalOp op) { return typeConverter.isLegal(op.getVal().getType()); });

        target.addDynamicallyLegalOp<P4HIR::SymToValueOp>(
            [&](P4HIR::SymToValueOp op) { return typeConverter.isLegal(op.getType()); });

        target.addDynamicallyLegalOp<P4HIR::ReadOp>(
            [&](P4HIR::ReadOp op) { return typeConverter.isLegal(op.getType()); });

        target.addDynamicallyLegalOp<P4HIR::InstantiateOp>([&](P4HIR::InstantiateOp op) {
            auto types = op.getTypeParameters();
            if (!types) return true;
            return llvm::all_of(types.value(), [&](Attribute ty) {
                return typeConverter.isLegal(cast<TypeAttr>(ty).getValue());
            });
        });

        RewritePatternSet patterns(&getContext());
        patterns.add<FlattenStructAccess<P4HIR::StructFieldRefOp>,
                     FlattenStructAccess<P4HIR::StructExtractOp>>(typeConverter, &getContext());
        populateOpTypeConversionPattern<
            P4HIR::FuncOp, P4HIR::ConstructOp, P4HIR::CallOp, P4HIR::CallMethodOp,
            P4HIR::TableApplyOp, P4HIR::ControlApplyOp, P4HIR::VariableOp, P4HIR::ControlLocalOp,
            P4HIR::TableKeyOp, P4HIR::ControlOp, P4HIR::ParserOp, P4HIR::ReadOp,
            P4HIR::InstantiateOp, P4HIR::SymToValueOp>(patterns, typeConverter);

        populateFunctionOpInterfaceTypeConversionPattern<P4HIR::FuncOp>(patterns, typeConverter);

        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<Pass> createFlattenStructsPass() {
    return std::make_unique<StructFlatteningPass>();
}
}  // namespace P4::P4MLIR
