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

    /// Calculates state in-degrees and identifies unique successors.
    void buildIndegreeAndSuccessor(
        P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable,
        llvm::DenseMap<P4HIR::ParserStateOp, unsigned> &indegree,
        llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> &succ);
};
}  // end anonymous namespace

llvm::DenseSet<P4HIR::ParserStateOp> SimplifyParsers::findReachableStates(
    P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable) {
    auto start = symbolTable.lookup<P4HIR::ParserStateOp>("start");
    if (!start) {
        parser.emitOpError("Parser missing 'start' state");
        return {};
    }
    llvm::SmallVector<P4HIR::ParserStateOp> worklist{start};
    llvm::DenseSet<P4HIR::ParserStateOp> visited{start};

    bool acceptReachable = false, rejectReachable = false;

    while (!worklist.empty()) {
        auto state = worklist.pop_back_val();
        Operation *term = state.getBody().back().getTerminator();

        mlir::TypeSwitch<Operation *, void>(term)
            .Case<P4HIR::ParserAcceptOp>([&](auto) { acceptReachable = true; })
            .Case<P4HIR::ParserRejectOp>([&](auto) { rejectReachable = true; })
            .Case<P4HIR::ParserTransitionOp>([&](auto transition) {
                auto next = symbolTable.lookup<P4HIR::ParserStateOp>(
                    transition.getStateAttr().getLeafReference());
                if (visited.insert(next).second) worklist.push_back(next);
            })
            .Case<P4HIR::ParserTransitionSelectOp>([&](P4HIR::ParserTransitionSelectOp select) {
                for (auto selectCase : select.getBody().getOps<P4HIR::ParserSelectCaseOp>()) {
                    auto next = symbolTable.lookup<P4HIR::ParserStateOp>(
                        selectCase.getStateAttr().getLeafReference());
                    if (visited.insert(next).second) worklist.push_back(next);
                }
            })
            .Default([&](auto) { llvm_unreachable("Unknown parser terminator"); });
    }

    if (!acceptReachable && !rejectReachable)
        parser.emitWarning("Parser never reaches the 'accept' or 'reject' state.");

    return visited;
}

void SimplifyParsers::buildIndegreeAndSuccessor(
    P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable,
    llvm::DenseMap<P4HIR::ParserStateOp, unsigned> &indegree,
    llvm::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> &succ) {
    // Helper function to identify special states that should never be collapsed
    auto isSpecial = [](P4HIR::ParserStateOp state) {
        auto name = state.getSymName();
        return name == "start" || name == "accept" || name == "reject";
    };

    for (auto state : parser.getBody().getOps<P4HIR::ParserStateOp>()) {
        if (isSpecial(state)) continue;

        Operation *term = state.getBody().back().getTerminator();
        mlir::TypeSwitch<Operation *, void>(term)
            .Case<P4HIR::ParserAcceptOp, P4HIR::ParserRejectOp>([&](auto) {})
            .Case<P4HIR::ParserTransitionOp>([&](auto tr) {
                auto next =
                    symbolTable.lookup<P4HIR::ParserStateOp>(tr.getStateAttr().getLeafReference());
                if (isSpecial(next)) return;

                ++indegree[next];
                succ[state] = next;
            })
            .Case<P4HIR::ParserTransitionSelectOp>([&](P4HIR::ParserTransitionSelectOp sel) {
                llvm::SmallVector<P4HIR::ParserStateOp> targets;
                for (auto selectCase : sel.getBody().getOps<P4HIR::ParserSelectCaseOp>()) {
                    auto next = symbolTable.lookup<P4HIR::ParserStateOp>(
                        selectCase.getStateAttr().getLeafReference());
                    targets.push_back(next);
                    if (isSpecial(next)) return;

                    ++indegree[next];
                }
                // A select with only one case is treated the same as a direct transition
                if (targets.size() == 1 && !isSpecial(targets.front()))
                    succ[state] = targets.front();
            })
            .Default([&](auto) { llvm_unreachable("Unknown parser terminator"); });
    }
}

void SimplifyParsers::collapseChains(P4HIR::ParserOp &parser, mlir::SymbolTable &symbolTable) {
    mlir::DenseMap<P4HIR::ParserStateOp, unsigned> indegree;
    // succ[s1] = s2 if there is exactly one outgoing edge from s1 to s2
    mlir::DenseMap<P4HIR::ParserStateOp, P4HIR::ParserStateOp> succ;
    buildIndegreeAndSuccessor(parser, symbolTable, indegree, succ);

    // Identify collapsible chain head states
    llvm::SmallVector<P4HIR::ParserStateOp> chainHeads;
    for (auto &[state, next] : succ) {
        if (indegree[state] > 0 || state.getAnnotations()) continue;
        chainHeads.push_back(state);
    }

    // Process each chain, collapsing all states into the head state
    for (auto head : chainHeads) {
        Block &headBody = head.getBody().back();

        // Walk forward through the chain using successor map
        // ensuring the next state has only one incoming edge and no annotations
        for (auto it = succ.find(head); it != succ.end(); it = succ.find(it->second)) {
            P4HIR::ParserStateOp next = it->second;
            if (indegree[next] != 1 || next.getAnnotations()) break;

            Block &nextBody = next.getBody().back();

            // Remove the terminator (transition to 'next') from the current head body
            headBody.getTerminator()->erase();

            // Splice all operations from 'next' into the head body
            headBody.getOperations().splice(headBody.end(), nextBody.getOperations(),
                                            nextBody.begin(), nextBody.end());
            next->erase();
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
