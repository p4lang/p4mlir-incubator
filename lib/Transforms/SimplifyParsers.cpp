#include <llvm/Support/ErrorHandling.h>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "p4mlir/Dialect/P4HIR/ParserGraph.h"
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
    /// Collapses linear sequences of states without branches or annotations.
    void collapseChains(P4HIR::ParserOp parser);
};
}  // end anonymous namespace

void SimplifyParsers::collapseChains(P4HIR::ParserOp parser) {
    // TODO: Revisit this to use ParserCallGraph instead
    mlir::DenseMap<P4HIR::ParserStateOp, unsigned> indegree;
    // Initialize indegree[start] to 1 to account for the implicit parser entry edge.
    indegree[parser.getStartState()] = 1;
    for (auto state : parser.states()) {
        for (auto next : state.getNextStates()) {
            ++indegree[next];
        }
    }

    // succ[s1] = s2 if there is exactly one outgoing edge from s1 to s2
    // and s2 has exactly one incoming edge from s1.
    mlir::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> succ;
    mlir::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> pred;

    // We diconnect any annotated states since they can't be collapsed
    // and if we kept them they'd poison downstream merging.
    for (auto state : parser.states()) {
        if (!llvm::hasNItems(state.getNextStates(), 1)) continue;
        // Name annotations are special, if they are the only annotation
        // then they can be merged into as heads.
        auto ann = state.getAnnotations();
        if (ann && !(ann->size() == 1 && ann->contains("name"))) continue;
        P4HIR::ParserStateOp successor = *state.getNextStates().begin();
        if (indegree[successor] != 1 || successor.getAnnotations()) continue;

        succ[state] = successor;
        pred[successor] = state;
    }

    // Process each chain head, collapsing states whenever possible.
    for (auto [head, _] : succ) {
        if (pred.contains(head)) continue;
        LLVM_DEBUG(llvm::dbgs() << "Chaining states into '" << head.getName() << "'\n");

        // Walk forward through the chain using successor map
        for (auto it = succ.find(head); it != succ.end(); it = succ.find(it->second)) {
            P4HIR::ParserStateOp next = it->second;
            LLVM_DEBUG(llvm::dbgs() << "\tAdding '" << next.getName() << "' to chain\n");
            auto &headOps = head.getBody().front().getOperations();

            // Remove the terminator (transition to 'next') from the current head body.
            head.getNextTransition()->erase();

            // Splice all operations from 'next' into the head body.
            headOps.splice(headOps.end(), next.getBody().front().getOperations());

            // Remove the now-empty 'next' state.
            next.erase();
        }
    }
}

void SimplifyParsers::runOnOperation() {
    getOperation()->walk([&](P4HIR::ParserOp parser) {
        LLVM_DEBUG(llvm::dbgs() << "\n--- Simplifying parser '" << parser.getName() << "' ---\n");
        llvm::df_iterator_default_set<llvm::GraphTraits<P4HIR::ParserOp>::NodeRef> reachable;
        bool acceptReachable = false, rejectReachable = false;

        /// Finds all states reachable from the start state and deletes unreachable states.
        for (auto stateOp : llvm::depth_first_ext(parser, reachable)) {
            auto state = mlir::cast<P4HIR::ParserStateOp>(stateOp);
            LLVM_DEBUG(llvm::dbgs() << "DFS visiting " << state.getName() << "\n");
            if (state.isAccept()) acceptReachable = true;
            if (state.isReject()) rejectReachable = true;
        }
        if (!acceptReachable && !rejectReachable)
            parser.emitError("Parser never reaches the 'accept' or 'reject' state.");
        LLVM_DEBUG(llvm::dbgs() << "Parser '" << parser.getName() << "' has " << reachable.size()
                                << " reachable states\n");

        for (auto state : llvm::make_early_inc_range(parser.states())) {
            if (!reachable.contains(state)) {
                if (state.isAccept()) {
                    parser.emitWarning()
                        << "Parser has unreachable accept state " << state.getName();
                } else {
                    LLVM_DEBUG(llvm::dbgs()
                               << "Removing unreachable state '" << state.getName() << "'\n");
                    state.erase();
                }
            }
        }

        collapseChains(parser);

        return WalkResult::advance();
    });
}

std::unique_ptr<Pass> createSimplifyParsersPass() { return std::make_unique<SimplifyParsers>(); }
}  // namespace P4::P4MLIR
