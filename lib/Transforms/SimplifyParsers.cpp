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
    llvm::DenseSet<P4HIR::ParserStateOp> findReachableStates(P4HIR::ParserOp &parser,
                                                             mlir::SymbolTable &symbolTable);

    /// Collapses linear sequences of states without branches or annotations.
    void collapseChains(P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable);
};
}  // end anonymous namespace

llvm::DenseSet<P4HIR::ParserStateOp> SimplifyParsers::findReachableStates(
    P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable) {
    auto term = llvm::cast<P4HIR::ParserTransitionOp>(parser.getBody().back().getTerminator());
    auto start = symbolTable.lookup<P4HIR::ParserStateOp>(term.getStateAttr().getLeafReference());
    llvm::SmallVector<P4HIR::ParserStateOp> worklist{start};
    llvm::DenseSet<P4HIR::ParserStateOp> visited{start};

    bool acceptReachable = false, rejectReachable = false;

    while (!worklist.empty()) {
        auto state = worklist.pop_back_val();
        for (auto nextOp : parser.getSuccessors(state, symbolTable)) {
            auto next = llvm::cast<P4HIR::ParserStateOp>(nextOp);
            if (mlir::isa<P4HIR::ParserAcceptOp>(next.getTerminator())) {
                acceptReachable = true;
            } else if (mlir::isa<P4HIR::ParserRejectOp>(next.getTerminator())) {
                rejectReachable = true;
            }
            if (visited.insert(next).second) worklist.push_back(next);
        }
    }

    if (!acceptReachable && !rejectReachable)
        parser.emitWarning("Parser never reaches the 'accept' or 'reject' state.");

    return visited;
}

void SimplifyParsers::collapseChains(P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable) {
    mlir::DenseMap<P4HIR::ParserStateOp, unsigned> indegree;
    // succ[s1] = s2 if there is exactly one outgoing edge from s1 to s2.
    mlir::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> succ;

    // We diconnect any annotated states since they can't be collapsed
    // and if we kept them they'd stop any merging downstream.
    for (auto state : parser.getBody().getOps<P4HIR::ParserStateOp>()) {
        if (state.getAnnotations()) continue;

        llvm::SmallVector<mlir::Operation *> successors = parser.getSuccessors(state, symbolTable);
        for (auto nextOp : successors) {
            auto next = llvm::cast<P4HIR::ParserStateOp>(nextOp);
            if (!next.getAnnotations()) ++indegree[next];
        }
        if (successors.size() == 1) {
            auto next = llvm::cast<P4HIR::ParserStateOp>(successors.front());
            if (!next.getAnnotations()) succ[state] = next;
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

            Block &headBlock = head.getBody().back();
            Block &nextBlock = next.getBody().back();

            // Remove the terminator (transition to 'next') from the current head body.
            headBlock.getTerminator()->erase();

            // Splice all operations from 'next' into the head body.
            headBlock.getOperations().splice(headBlock.end(), nextBlock.getOperations());
            next.erase();
        }
    }
}

void SimplifyParsers::runOnOperation() {
    getOperation()->walk([&](P4HIR::ParserOp parser) {
        mlir::SymbolTable symbolTable(parser);

        llvm::DenseSet<P4HIR::ParserStateOp> reachable = findReachableStates(parser, symbolTable);
        for (auto s : llvm::make_early_inc_range(parser.getBody().getOps<P4HIR::ParserStateOp>()))
            if (!reachable.contains(s)) s.erase();

        collapseChains(parser, symbolTable);

        return WalkResult::advance();
    });
}

std::unique_ptr<Pass> createSimplifyParsersPass() { return std::make_unique<SimplifyParsers>(); }
}  // namespace P4::P4MLIR
