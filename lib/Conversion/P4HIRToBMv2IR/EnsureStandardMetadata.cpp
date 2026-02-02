#include <array>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "set-corelib"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_ENSURESTANDARDMETADATA
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

BMv2IR::HeaderType getStandardMetadataTy(MLIRContext *ctx) {
    std::array<std::pair<StringRef, unsigned>, 16> fields{{{"ingress_port", 9},
                                                           {"egress_spec", 9},
                                                           {"egress_port", 9},
                                                           {"instance_type", 32},
                                                           {"packet_length", 32},
                                                           {"enq_timestamp", 32},
                                                           {"enq_qdepth", 19},
                                                           {"deq_timedelta", 32},
                                                           {"deq_qdepth", 19},
                                                           {"ingress_global_timestamp", 48},
                                                           {"egress_global_timestamp", 48},
                                                           {"mcast_grp", 16},
                                                           {"egress_rid", 16},
                                                           {"checksum_error", 1},
                                                           {"priority", 3},
                                                           {"_padding", 3}}};
    SmallVector<BMv2IR::FieldInfo, 16> fieldInfos;

    for (auto &field : fields) {
        fieldInfos.emplace_back(StringAttr::get(ctx, field.first),
                                P4HIR::BitsType::get(ctx, field.second, false));
    }

    return BMv2IR::HeaderType::get(ctx, BMv2IR::standardMetadataNewStructName, fieldInfos);
}

struct EnsureStandardMetadataPass
    : public P4::P4MLIR::impl::EnsureStandardMetadataBase<EnsureStandardMetadataPass> {
    void runOnOperation() override {
        bool hasStandardMetadata = false;
        auto moduleOp = getOperation();
        moduleOp.walk([&](BMv2IR::HeaderInstanceOp instance) {
            if (instance.getSymName() == BMv2IR::standardMetadataNewStructName)
                hasStandardMetadata = true;
        });
        if (hasStandardMetadata) return;
        // We don't have standard_metadata header instance, need to add one
        auto &ctx = getContext();
        OpBuilder builder(moduleOp.getBodyRegion());
        auto ty = getStandardMetadataTy(&ctx);
        builder.create<BMv2IR::HeaderInstanceOp>(moduleOp.getLoc(),
                                                 BMv2IR::standardMetadataNewStructName, ty, true);
    };
};

}  // anonymous namespace
