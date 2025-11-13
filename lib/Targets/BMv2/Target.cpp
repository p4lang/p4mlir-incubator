#include "p4mlir/Targets/BMv2/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "p4mlir/Common/Registration.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"

using namespace mlir;
using namespace P4::P4MLIR;

mlir::FailureOr<llvm::json::Value> P4::P4MLIR::bmv2irToJson(ModuleOp moduleOp) {
    moduleOp->emitError("MLIR to BMv2 NYI");
    return failure();
}

void P4::P4MLIR::registerToBMv2JSONTranslation() {
    TranslateFromMLIRRegistration registration(
        "p4hir-to-bmv2-json", "Translate MLIR to BMv2 JSON",
        [](Operation *op, raw_ostream &output) {
            auto moduleOp = dyn_cast<ModuleOp>(op);
            if (!moduleOp) return failure();
            if (failed(bmv2irToJson(moduleOp, output))) return failure();
            return success();
        },
        [](DialectRegistry &registry) { P4::P4MLIR::registerAllDialects(registry); });
}

LogicalResult P4::P4MLIR::bmv2irToJson(ModuleOp moduleOp, raw_ostream &output) {
    auto maybeJsonModule = bmv2irToJson(moduleOp);
    if (failed(maybeJsonModule)) return failure();

    output << *maybeJsonModule;
    return success();
}
