#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-enum-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_ENUMELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct EnumEliminationPass : public P4::P4MLIR::impl::EnumEliminationBase<EnumEliminationPass> {
    EnumEliminationPass() = default;
    void runOnOperation() override;
};

class EnumTypeConverter : public P4HIRTypeConverter {
 public:
    explicit EnumTypeConverter() {
        // Converts EnumType to SerEnumType
        addConversion([](P4HIR::EnumType enumType) -> Type {
            if (!enumType.shouldConvert()) {
                return enumType;
            }

            auto underlyingType = mlir::cast<P4HIR::BitsType>(enumType.getUnderlyingType());

            llvm::SmallVector<mlir::NamedAttribute> fields;
            for (auto const &[index, field] : llvm::enumerate(enumType.getFields())) {
                auto name = mlir::cast<StringAttr>(field);
                auto value =
                    P4HIR::IntAttr::get(underlyingType, enumType.getEncodingForField(name, index));
                fields.emplace_back(name, value);
            }
            return P4HIR::SerEnumType::get(enumType.getName(), underlyingType, fields,
                                           enumType.getAnnotations());
        });

        addTypeAttributeConversion([this](P4HIR::EnumType enumType, P4HIR::EnumFieldAttr attr) {
            return P4HIR::EnumFieldAttr::get(convertType(enumType), attr.getField());
        });
    }
};

}  // namespace

void EnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    EnumTypeConverter typeConverter;
    ConversionTarget target(context);

    target.addDynamicallyLegalOp<P4HIR::OverloadSetOp>([&](P4HIR::OverloadSetOp ovl) {
        for (auto opFunc : ovl.getOps<P4HIR::FuncOp>()) return !target.isIllegal(opFunc);
        return true;
    });

    target.addDynamicallyLegalOp<P4HIR::CaseOp>([&](P4HIR::CaseOp caseOp) {
        return llvm::all_of(caseOp.getValue(), [&](Attribute val) {
            if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(val)) {
                return typeConverter.isLegal(typedAttr.getType());
            }
            return true;
        });
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (auto func = dyn_cast<FunctionOpInterface>(op)) {
            auto fnType = func.getFunctionType();
            return typeConverter.isLegal(fnType);
        }

        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(&context);

    populateP4HIRAnyCallOpTypeConversionPattern(patterns, typeConverter);
    populateP4HIRFunctionOpTypeConversionPattern<P4HIR::FuncOp, P4HIR::ParserOp, P4HIR::ControlOp>(
        patterns, typeConverter);
    patterns.add<TypeConversionPattern>(typeConverter, &context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createEnumEliminationPass() {
    return std::make_unique<EnumEliminationPass>();
}
