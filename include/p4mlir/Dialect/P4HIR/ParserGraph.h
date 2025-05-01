#ifndef P4MLIR_DIALECT_P4HIR_PARSERGRAPH_H
#define P4MLIR_DIALECT_P4HIR_PARSERGRAPH_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "mlir/Support/LLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR::P4HIR::detail {
// Unfortunately, GraphTraits::NodeRef must be a pointer, therefore we cannot
// have NodeRef to be equal to ParserStateOp.
// Using declaration to avoid polluting global namespace with P4HIR-specific
// graph traits for mlir::Operation.
using P4HIROperation = mlir::Operation;
}  // namespace P4::P4MLIR::P4HIR::detail

template <>
struct llvm::GraphTraits<P4::P4MLIR::P4HIR::ParserOp> {
    using NodeType = P4::P4MLIR::P4HIR::ParserStateOp;
    using NodeRef = P4::P4MLIR::P4HIR::detail::P4HIROperation *;
    using GraphType = P4::P4MLIR::P4HIR::ParserOp;

    static NodeRef getEntryNode(GraphType parser) { return parser.getStartState(); }

    using ChildIteratorType = NodeType::StateIterator;
    static ChildIteratorType child_begin(NodeRef state) {
        return mlir::cast<NodeType>(state).getNextStates().begin();
    }
    static ChildIteratorType child_end(NodeRef state) {
        return mlir::cast<NodeType>(state).getNextStates().end();
    }

    using nodes_iterator = mlir::Block::op_iterator<NodeType>;
    static nodes_iterator nodes_begin(GraphType parser) { return parser.state_begin(); }
    static nodes_iterator nodes_end(GraphType parser) { return parser.state_end(); }
};

template <>
struct llvm::DOTGraphTraits<P4::P4MLIR::P4HIR::ParserOp> : public llvm::DefaultDOTGraphTraits {
    using NodeType = P4::P4MLIR::P4HIR::ParserStateOp;
    using GraphType = P4::P4MLIR::P4HIR::ParserOp;

    using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

    static std::string getGraphName(GraphType parser) { return parser.getName().str(); }

    static std::string getNodeLabel(P4::P4MLIR::P4HIR::detail::P4HIROperation *node,
                                    P4::P4MLIR::P4HIR::ParserOp) {
        auto state = mlir::cast<NodeType>(node);
        return state.getName().str();
    }

    std::string getNodeAttributes(P4::P4MLIR::P4HIR::detail::P4HIROperation *node,
                                  P4::P4MLIR::P4HIR::ParserOp parser) {
        auto state = mlir::cast<NodeType>(node);
        if (state == parser.getStartState()) return "shape=oval,fillcolor=lightblue,style=filled";
        if (state.isAccept()) return "shape=oval,fillcolor=forestgreen,style=filled";
        if (state.isReject()) return "shape=oval,fillcolor=tomato,style=filled";

        return "";
    }

    template <typename Iterator>
    static std::string getEdgeAttributes(P4::P4MLIR::P4HIR::detail::P4HIROperation *, Iterator it,
                                         P4::P4MLIR::P4HIR::ParserOp) {
        P4::P4MLIR::P4HIR::ParserStateOp nextState = *it;
        std::string str;
        llvm::raw_string_ostream os(str);
        if (nextState.isAccept())
            os << "color=green";
        else if (nextState.isReject())
            os << "color=red";

        return os.str();
    }
};

#endif  // P4MLIR_DIALECT_P4HIR_PARSERGRAPH_H
