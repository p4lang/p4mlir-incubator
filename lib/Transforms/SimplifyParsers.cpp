#include <llvm/Support/ErrorHandling.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-simplify-parsers"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SIMPLIFYPARSERS
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {
struct SimplifyParsers : public impl::SimplifyParsersBase<SimplifyParsers> {
    void runOnOperation() override;

 private:
    /// Finds all states reachable from the 'start' state.
    llvm::DenseSet<P4HIR::ParserStateOp> findReachableStates(P4HIR::ParserOp parser);

    /// Collapses linear sequences of states without branches or annotations.
    void collapseChains(P4HIR::ParserOp parser);
};
}  // end anonymous namespace

llvm::DenseSet<P4HIR::ParserStateOp> SimplifyParsers::findReachableStates(P4HIR::ParserOp parser) {
    auto start = parser.getStartState();
    llvm::SmallVector<P4HIR::ParserStateOp> worklist{start};
    llvm::DenseSet<P4HIR::ParserStateOp> visited{start};

    bool acceptReachable = false, rejectReachable = false;

    while (!worklist.empty()) {
        auto state = worklist.pop_back_val();
        for (auto next : state.getNextStates()) {
            if (next.isAccept()) acceptReachable = true;
            else if (next.isReject()) rejectReachable = true;
            if (!visited.contains(next)) worklist.push_back(next);
        }
    }

    if (!acceptReachable && !rejectReachable)
        parser.emitWarning("Parser never reaches the 'accept' or 'reject' state.");

    return visited;
}

void SimplifyParsers::collapseChains(P4HIR::ParserOp parser) {
    mlir::DenseMap<P4HIR::ParserStateOp, unsigned> indegree;
    // succ[s1] = s2 if there is exactly one outgoing edge from s1 to s2.
    mlir::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> succ;

    // We diconnect any annotated states since they can't be collapsed
    // and if we kept them they'd stop any merging downstream.
    for (auto state : parser.states()) {
        if (state.getAnnotations()) continue;

        for (auto next : state.getNextStates()) {
            if (!next.getAnnotations()) ++indegree[next];
        }
    }

    // Process each chain head, collapsing states whenever possible.
    for (auto &[head, _] : succ) {
        if (indegree[head] == 1) continue;  // not a chain head

        // Walk forward through the chain using successor map ensuring the next state
        // has only one incoming edge.
        for (auto it = succ.find(head); it != succ.end(); it = succ.find(it->second)) {
            P4HIR::ParserStateOp next = it->second;
            if (indegree[next] != 1) break;
            auto &headOps = head.getBody().back().getOperations();
            auto &nextOps = next.getBody().back().getOperations();

            // Remove the terminator (transition to 'next') from the current head body.
            headOps.back().erase();

            // Splice all operations from 'next' into the head body.
            headOps.splice(headOps.end(), nextOps);

            // Remove the now-empty 'next' state.
            next.erase();
        }
    }
}

void SimplifyParsers::runOnOperation() {
    getOperation()->walk([&](P4HIR::ParserOp parser) {
        llvm::DenseSet<P4HIR::ParserStateOp> reachable = findReachableStates(parser);
        for (auto s : llvm::make_early_inc_range(parser.states()))
            if (!reachable.contains(s)) s.erase();

        collapseChains(parser);

        return WalkResult::advance();
    });
}

std::unique_ptr<Pass> createSimplifyParsersPass() { return std::make_unique<SimplifyParsers>(); }
}  // namespace P4::P4MLIR
