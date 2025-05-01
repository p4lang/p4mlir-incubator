#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/ParserGraph.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-print-parsers-graph"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_PRINTPARSERSGRAPH
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct PrintParsersGraphPass
    : public P4::P4MLIR::impl::PrintParsersGraphBase<PrintParsersGraphPass> {
    explicit PrintParsersGraphPass(raw_ostream &os) : os(os) {}
    void runOnOperation() override {
        getOperation()->walk(
            [&](P4HIR::ParserOp parser) { llvm::WriteGraph(os, parser, /*ShortNames=*/false); });
    }
    raw_ostream &os;
};
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createPrintParsersGraphPass() {
    return std::make_unique<PrintParsersGraphPass>(llvm::errs());
}
