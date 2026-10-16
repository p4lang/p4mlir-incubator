// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "p4mlir/Transforms/IRUtils.h"

#include <cctype>

#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

using namespace P4::P4MLIR;

mlir::FailureOr<llvm::SmallVector<mlir::BlockArgument>> IRUtils::cloneFunctionLikeArgs(
    mlir::FunctionOpInterface obj, llvm::ArrayRef<mlir::BlockArgument> args,
    llvm::ArrayRef<mlir::Type> newTypes /* = {} */) {
    assert((newTypes.size() == args.size() || newTypes.empty()) && "Expected same count new types");

    auto argIndices =
        llvm::map_to_vector(args, [](mlir::BlockArgument arg) { return arg.getArgNumber(); });
    auto argTypes = llvm::to_vector(newTypes);
    if (argTypes.empty())
        argTypes = llvm::map_to_vector(args, [](mlir::BlockArgument arg) { return arg.getType(); });
    auto argLocs = llvm::map_to_vector(args, [](mlir::BlockArgument arg) { return arg.getLoc(); });
    auto argAttrs = llvm::map_to_vector(args, [&](mlir::BlockArgument arg) {
        return mlir::DictionaryAttr::get(obj.getContext(), obj.getArgAttrs(arg.getArgNumber()));
    });

    if (failed(obj.insertArguments(argIndices, argTypes, argAttrs, argLocs)))
        return mlir::failure();

    auto newArgs = llvm::map_to_vector(
        args, [&](mlir::BlockArgument arg) { return obj.getArgument(arg.getArgNumber() - 1); });
    return newArgs;
}

mlir::LogicalResult IRUtils::eraseFunctionLikeArgs(mlir::FunctionOpInterface obj,
                                                   llvm::ArrayRef<mlir::BlockArgument> args) {
    llvm::BitVector argsToErase(obj.getNumArguments());
    for (auto arg : args) argsToErase.set(arg.getArgNumber());
    return obj.eraseArguments(argsToErase);
}

std::string IRUtils::legalizeName(std::string str) {
    unsigned e = str.size();
    unsigned i1 = 0;
    unsigned i2 = 0;
    while (i1 < e) {
        char c = str[i1++];

        if (std::isalnum(c)) {
            str[i2++] = c;
        } else {
            str[i2++] = '_';
            while (i1 < e && !std::isalnum(str[i1])) i1++;
        }
    }
    str.resize(i2);
    return str;
}

P4HIR::StructType IRUtils::createStructType(mlir::MLIRContext *ctx, const llvm::Twine &name,
                                            const llvm::SmallVectorImpl<P4HIR::FieldPath> &paths) {
    auto fields = llvm::map_to_vector(paths, [&](P4HIR::FieldPath path) {
        auto name = mlir::StringAttr::get(ctx, IRUtils::legalizeName(path.str()));
        return P4HIR::FieldInfo(name, path.getType());
    });
    return P4HIR::StructType::get(ctx, name.str(), fields);
}

mlir::Value IRUtils::getAccessibleValue(mlir::RewriterBase &rewriter, mlir::Operation *op,
                                        mlir::Value val) {
    namespace P4HIR = P4::P4MLIR::P4HIR;
    auto funcOp = op->template getParentOfType<P4HIR::FuncOp>();
    if (!funcOp) return val;

    auto parentOp = funcOp->getParentOfType<mlir::FunctionOpInterface>();
    if (!parentOp) return val;

    // Try to find existing control local for `val`.
    mlir::Block &body = parentOp.getCallableRegion()->front();
    P4HIR::ControlLocalOp valControlLocal;
    for (auto controlLocalOp : body.getOps<P4HIR::ControlLocalOp>()) {
        if (controlLocalOp.getVal() == val) {
            valControlLocal = controlLocalOp;
            break;
        }
    }

    if (!valControlLocal) {
        // If we couldn't find an existing control local then create new one.
        auto getUniqueName = [&](std::string toRename) {
            unsigned counter = 0;
            return mlir::SymbolTable::generateSymbolName<256>(
                toRename,
                [&](llvm::StringRef candidate) {
                    return mlir::SymbolTable::lookupSymbolIn(parentOp, candidate) != nullptr;
                },
                counter);
        };

        std::string valName;
        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
            rewriter.setInsertionPointToStart(&body);
            valName = std::string("arg") + std::to_string(arg.getArgNumber());
        } else if (auto var = val.getDefiningOp<P4HIR::VariableOp>()) {
            rewriter.setInsertionPointAfter(var);
            valName = var.getName().value_or("");
        } else {
            llvm_unreachable("Expected argument or variable to create accessible ref");
        }

        auto controlLocalName = ("__local_" + parentOp.getName() + "_" + valName).str();
        valControlLocal = P4HIR::ControlLocalOp::create(rewriter, parentOp->getLoc(),
                                                        getUniqueName(controlLocalName), val);
    }

    // Finally create sym_to_val at `op`'s location to access `val`.
    rewriter.setInsertionPoint(op);
    return P4HIR::SymToValueOp::create(rewriter, op->getLoc(), val.getType(),
                                       mlir::SymbolRefAttr::get(valControlLocal));
}

void IRUtils::inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp) {
    mlir::Block *block = &scopeOp.getScopeRegion().front();
    mlir::Operation *terminator = block->getTerminator();
    mlir::ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, scopeOp, /*blockArgs=*/{});
    rewriter.replaceOp(scopeOp, results);
    rewriter.eraseOp(terminator);
}

bool IRUtils::canSplitBlockAt(mlir::Block *block, mlir::Operation *op) {
    assert(block->findAncestorOpInBlock(*op) != nullptr);

    while (true) {
        if (op->getBlock() == block) return true;
        op = op->getParentOp();
        if (!op || !mlir::isa<P4HIR::ScopeOp>(op)) return false;
    }
}

std::array<mlir::Block *, 3> IRUtils::splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                                   mlir::Operation *op) {
    assert(canSplitBlockAt(block, op));

    // Inline all scopes surrounding `op`.
    block->walk<mlir::WalkOrder::PostOrder>(
        [&](P4HIR::ScopeOp scopeOp) { inlineScope(rewriter, scopeOp); });

    // Split resulting block.
    mlir::Block *before = op->getBlock();
    mlir::Block *middle = rewriter.splitBlock(before, op->getIterator());
    mlir::Block *after = rewriter.splitBlock(middle, std::next(op->getIterator()));
    return {before, middle, after};
}

namespace {

using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;

// Returns blocks that contain uses of val and are not ancestors of `block`.
BlockSet getEscapingBlocks(mlir::Block *block, mlir::Value val) {
    BlockSet escapingBlocks;
    for (mlir::Operation *user : val.getUsers())
        if (user->getBlock() != block && !block->findAncestorOpInBlock(*user))
            escapingBlocks.insert(user->getBlock());
    return escapingBlocks;
}

// Potentially promote `val` to a local variable.
// The rewriter's insertion point is the location where the new variable is created.
void promoteValToVar(mlir::RewriterBase &rewriter, mlir::Value val, const BlockSet &escapingBlocks,
                     llvm::StringRef varName = "promoted_local") {
    if (escapingBlocks.empty()) return;

    // Create new variable to hold `val`.
    auto newVar = P4HIR::VariableOp::create(
        rewriter, val.getLoc(), P4HIR::ReferenceType::get(val.getType()), varName, true);

    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Insert a new read in all escaping blocks and replace uses of `val` in that block.
    for (mlir::Block *block : escapingBlocks) {
        rewriter.setInsertionPointToStart(block);
        auto newVal = P4HIR::ReadOp::create(rewriter, val.getLoc(), newVar);
        rewriter.replaceUsesWithIf(val, newVal, [block](mlir::OpOperand &use) {
            return use.getOwner()->getBlock() == block;
        });
    }

    // Assign the new variable after `val`'s definition.
    rewriter.setInsertionPointAfterValue(val);
    P4HIR::AssignOp::create(rewriter, val.getLoc(), val, newVar);
}

// Replace uses of `op` with a copy in all escaping blocks.
void copyOp(mlir::RewriterBase &rewriter, mlir::Operation *op, const BlockSet &escapingBlocks) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    for (mlir::Block *block : escapingBlocks) {
        rewriter.setInsertionPointToStart(block);
        auto newOp = rewriter.clone(*op);
        rewriter.replaceOpUsesWithinBlock(op, newOp->getResults(), block);
    }
}

}  // namespace

// Corrects all values that are used outside `block` due to splitting.
// Rewriter position must be in a place where we can create locals (e.g. parser or control locals).
void IRUtils::adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block) {
    // Iterate in reverse so we process uses before their definition in this block.
    // This is needed to properly handle operands after `copyOp` calls.
    for (mlir::Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
        if (mlir::isa<P4HIR::VariableOp>(op)) {
            // Move variables to top-level if needed.
            if (!getEscapingBlocks(block, op.getResult(0)).empty())
                rewriter.moveOpBefore(&op, rewriter.getInsertionBlock(),
                                      rewriter.getInsertionPoint());
        } else if (mlir::isa<P4HIR::ConstOp, P4HIR::StructFieldRefOp, P4HIR::ArrayElementRefOp,
                             P4HIR::ReadSliceOp>(op)) {
            // Copy RefOps and constants.
            auto escapingBlocks = getEscapingBlocks(block, op.getResult(0));
            copyOp(rewriter, &op, escapingBlocks);
        } else {
            // Promote values to variables as needed.
            for (auto val : op.getResults())
                promoteValToVar(rewriter, val, getEscapingBlocks(block, val));
        }
    }
}

P4HIR::ParserStateOp IRUtils::createSubState(mlir::RewriterBase &rewriter,
                                             P4HIR::ParserStateOp state,
                                             const llvm::Twine &suffix) {
    auto parser = state->getParentOfType<P4HIR::ParserOp>();
    auto getUniqueName = [&](const std::string &basename) {
        unsigned counter = 0;
        return mlir::SymbolTable::generateSymbolName<256>(
            basename,
            [&](llvm::StringRef candidate) { return parser.lookupSymbol(candidate) != nullptr; },
            counter);
    };

    auto name = getUniqueName((state.getSymName() + "_" + suffix).str());
    auto newState =
        P4HIR::ParserStateOp::create(rewriter, state.getLoc(), name, mlir::DictionaryAttr());
    rewriter.createBlock(&newState.getBody(), newState.getBody().begin());

    return newState;
}

mlir::LogicalResult IRUtils::SplitStateRewriter::init() {
    assert(step == CREATED || step == CANCELED);
    auto state = getState();

    if (!state) return op->emitError() << "Operation must be within parser state.";

    if (!canSplitBlockAt(state.getBlock(), op))
        return op->emitError() << "Cannot split state at given position.";

    step = INITIALIZED;

    // Create two new sub-states "pre" and "post" that will hold the code before and after `op`
    // respectively.
    setStateInsertionPointAfter(state);
    preState = createSubState("pre");
    postState = createSubState("post");
    setStateInsertionPointAfter(preState);

    return mlir::success();
}

P4HIR::ParserStateOp IRUtils::SplitStateRewriter::createSubState(const llvm::Twine &suffix,
                                                                 P4HIR::ParserStateOp transitionTo,
                                                                 mlir::Block *ops) {
    assert(step == INITIALIZED);
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPointAfter(stateCreationPoint);
    auto newState = IRUtils::createSubState(rewriter, getState(), suffix);
    auto newStateBB = newState.getBlock();

    if (ops) {
        rewriter.inlineBlockBefore(ops, newStateBB, newStateBB->end());
        rewriter.eraseOp(newStateBB->getTerminator());
    }

    if (transitionTo) {
        rewriter.setInsertionPointToEnd(newStateBB);
        P4HIR::ParserTransitionOp::create(rewriter, newState.getLoc(), transitionTo.getSymbolRef());
    }

    stateCreationPoint = newState;

    return newState;
}

void IRUtils::SplitStateRewriter::cancel() {
    assert(step == CREATED || step == INITIALIZED);

    if (step == INITIALIZED) {
        auto createdStates = llvm::make_range(preState->getIterator(), ++postState->getIterator());
        for (mlir::Operation &op : llvm::make_early_inc_range(createdStates)) rewriter.eraseOp(&op);
    }

    step = CANCELED;
}

void IRUtils::SplitStateRewriter::finalize() {
    assert(step == INITIALIZED);

    auto state = getState();
    assert(canSplitBlockAt(state.getBlock(), op) && "Split op moved to invalid location");
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Split code around `op` and move it to "pre" / "post" states.
    auto [beforeBB, opBB, afterBB] = splitBlockAt(rewriter, state.getBlock(), op);

    auto *preStateBB = preState.getBlock();
    rewriter.inlineBlockBefore(beforeBB, preStateBB, preStateBB->begin());
    auto *postStateBB = postState.getBlock();
    rewriter.inlineBlockBefore(afterBB, postStateBB, postStateBB->begin());

    // Replace `op` with a transition to the "pre" state.
    rewriter.setInsertionPoint(op);
    P4HIR::ParserTransitionOp::create(rewriter, op->getLoc(), preState.getSymbolRef());
    rewriter.eraseOp(op);

    // Due to the splitting states, we may have values with uses in other states.
    auto parser = state->getParentOfType<P4HIR::ParserOp>();
    rewriter.setInsertionPoint(parser.getStartState());
    adjustBlockUses(rewriter, preStateBB);

    step = FINALIZED;
}

/// Return an indirect use mapping for P4HIR `control_local` and `symbol_ref`-based
/// accesses.
IRUtils::PathWalker::IndirectUsesMap IRUtils::PathWalker::getIndirectSymbolUses(
    P4HIR::ControlOp control) {
    IndirectUsesMap result;
    control.walk([&](P4HIR::SymToValueOp symbolRef) {
        if (auto referencedControlLocal = symbolRef.getDeclOp<P4HIR::ControlLocalOp>()) {
            auto [it, ins] = result.insert({referencedControlLocal.getVal(), {}});
            it->second.push_back(symbolRef.getResult());
        }
    });
    return result;
}

/// Helper to look through a P4HIR `struct_extract` or `struct_field_ref` operation.
std::pair<mlir::Value, P4HIR::FieldPath> IRUtils::PathWalker::lookThroughStructAccess(
    mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path) {
    return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
        .Case<P4HIR::StructExtractOp, P4HIR::StructFieldRefOp>([&](auto structAccessOp) {
            assert((operand.get() == structAccessOp.getInput()) && "Unexpected operand");
            return std::pair{structAccessOp.getResult(), path[structAccessOp.getFieldIndex()]};
        })
        .Default([](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
}

/// Helper to look through a P4HIR `array_get` or `array_element_ref` operation.
std::pair<mlir::Value, P4HIR::FieldPath> IRUtils::PathWalker::lookThroughArrayAccess(
    mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path) {
    return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
        .Case<P4HIR::ArrayGetOp, P4HIR::ArrayElementRefOp>(
            [&](auto arrayAccessOp) -> std::pair<mlir::Value, P4HIR::FieldPath> {
                assert((operand.get() == arrayAccessOp.getInput()) && "Unexpected operand");

                unsigned cstIndex;
                if (matchPattern(arrayAccessOp.getIndex(), m_ConstantInt(&cstIndex)))
                    return {arrayAccessOp.getResult(), path[cstIndex]};

                return {mlir::Value(), P4HIR::FieldPath()};
            })
        .Default([](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
}

/// `operand` is an operand in `op` that is used to access the root value's field that
/// corresponds to `path`. If `op` is a field access operation and the resulting path can be
/// determined based on `path`, then this function should return the resulting new value and
/// path. Otherwise an empty value must be returned.
std::pair<mlir::Value, P4HIR::FieldPath> IRUtils::PathWalker::lookThroughUser(
    mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path) {
    // Default implementation to handle P4HIR operations. Users should override this to also
    // support operations from other dialects.
    auto lookThroughStruct = lookThroughStructAccess(op, operand, path);
    if (lookThroughStruct.first) return lookThroughStruct;

    auto lookThroughArray = lookThroughArrayAccess(op, operand, path);
    if (lookThroughArray.first) return lookThroughArray;

    return {mlir::Value(), P4HIR::FieldPath()};
}

mlir::WalkResult IRUtils::PathWalker::walkImpl(mlir::Value value, P4HIR::FieldPath path) {
    auto status = nodeCb ? nodeCb(value, path) : mlir::WalkResult::advance();
    if (status.wasSkipped()) return mlir::WalkResult::advance();
    if (status.wasInterrupted()) return mlir::WalkResult::interrupt();

    for (auto &use : value.getUses()) {
        mlir::Operation *user = use.getOwner();

        auto [resValue, resPath] = lookThroughUser(user, use, path);
        if (resValue) {
            if (walkImpl(resValue, resPath).wasInterrupted()) return mlir::WalkResult::interrupt();
        } else {
            auto status = leafCb ? leafCb(user, use, path) : mlir::WalkResult::advance();
            if (status.wasInterrupted()) return mlir::WalkResult::interrupt();
        }
    }

    if (indirectUses) {
        auto it = indirectUses->find(value);
        if (it != indirectUses->end()) {
            for (mlir::Value alias : it->second)
                if (walkImpl(alias, path).wasInterrupted()) return mlir::WalkResult::interrupt();
        }
    }

    return mlir::WalkResult::advance();
}

mlir::Value IRUtils::PathRewriter::getFromField(mlir::Location loc, mlir::Value val,
                                                P4HIR::IndexedField field) {
    // Default implementation to handle P4HIR operations. Users should override this to also
    // support operations from other dialects.
    bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
    if (mlir::isa<P4HIR::StructLikeTypeInterface>(field.getParentType())) {
        if (isRef)
            return P4HIR::StructFieldRefOp::create(rewriter, loc, val, field.getIndex());
        else
            return P4HIR::StructExtractOp::create(rewriter, loc, val, field.getIndex());
    } else if (mlir::isa<P4HIR::ArrayType>(field.getParentType())) {
        auto idxType = P4HIR::BitsType::get(rewriter.getContext(), 32, false);
        auto idx =
            P4HIR::ConstOp::create(rewriter, loc, P4HIR::IntAttr::get(idxType, field.getIndex()));
        if (isRef)
            return P4HIR::ArrayElementRefOp::create(rewriter, loc, val, idx);
        else
            return P4HIR::ArrayGetOp::create(rewriter, loc, val, idx);
    } else {
        llvm_unreachable("Unexpected type");
        return mlir::Value();
    }
}

std::pair<mlir::OpOperand *, P4HIR::FieldPath> IRUtils::PathRewriter::getOperandWithPath(
    mlir::Operation *op) {
    P4HIR::FieldPath resPath;
    mlir::OpOperand *resOperand = nullptr;
    for (mlir::OpOperand &operand : op->getOpOperands()) {
        if (auto path = getPath(operand.get()); !path.isEmpty()) {
            assert(!resOperand && "Unexpected operation with multiple used paths");
            resPath = path;
            resOperand = &operand;
        }
    }
    assert(resOperand && "A path must exist for leaf operations");
    return {resOperand, resPath};
}

mlir::LogicalResult IRUtils::PathRewriter::transform(bool eraseDead /* = false */) {
    /// Use a vector and deduplicate with a set so we get deterministic order.
    llvm::SmallVector<mlir::Operation *> leafOps;
    mlir::DenseSet<mlir::Operation *> seenLeafOps;

    auto nodeCb = [&](mlir::Value value, P4HIR::FieldPath path) {
        mlir::Operation *def = value.getDefiningOp();
        if (eraseDead && def) opsToRemove.push_back(def);

        rewriter.setInsertionPointAfterValue(value);
        mlir::Value valReplacement = replaceValue(value, path);

        if (valReplacement) {
            // Stop early: replace `value` with a new one that doesn't depend on the root.
            rewriter.replaceAllUsesWith(value, valReplacement);
            return mlir::WalkResult::skip();
        }

        valueToPath.insert({value, path});

        return mlir::WalkResult::advance();
    };

    auto leafCb = [&](mlir::Operation *user, const mlir::OpOperand &operand,
                      P4HIR::FieldPath path) {
        // It is reasonable to expect that leaf operations may have multiple uses of
        // root fields, so we don't directly ask for their replacement. Store `user` and
        // process once we have discovered all leaf paths.
        if (seenLeafOps.insert(user).second) leafOps.push_back(user);
        return mlir::WalkResult::advance();
    };

    [[maybe_unused]] auto status = PathWalker(nodeCb, leafCb, indirectUses).walk(root, rootType);
    assert(!status.wasInterrupted() && "Unexpected state");

    for (mlir::Operation *op : leafOps) {
        rewriter.setInsertionPoint(op);
        if (failed(replaceUsesIn(op))) return mlir::failure();
    }

    if (eraseDead) {
        // `opsToRemove` is in DFS preorder, so we need to erase them from back to front.
        while (!opsToRemove.empty()) {
            mlir::Operation *op = opsToRemove.pop_back_val();
            if (op->use_empty()) rewriter.eraseOp(op);
        }
    }

    return mlir::success();
}

std::optional<IRUtils::LocationMapping::SplitFieldLocation> IRUtils::LocationMapping::remap(
    FieldLocation from) const {
    assert((from.first < fromCount()) && "'from' index out of bounds");
    auto [fromIndex, fromPath] = from;
    auto prefixes = llvm::to_vector(fromPath.paths());

    while (!prefixes.empty()) {
        auto fromPrefix = prefixes.pop_back_val();
        if (auto *toLocation = lookup(FieldLocation(fromIndex, fromPrefix))) {
            auto fromSuffix = P4HIR::FieldPath::subtract(fromPath, fromPrefix);
            auto [toIndex, toPath] = *toLocation;
            return SplitFieldLocation(toIndex, toPath, fromSuffix);
        }
    }

    return std::nullopt;
}

bool IRUtils::LocationMapping::addMapping(FieldLocation from, FieldLocation to) {
    assert((from.first < fromCount()) && "'from' index out of bounds");
    assert((to.first < toCount()) && "'to' index out of bounds");
    assert((from.second.getRootType() == fromTypes[from.first]) && "'from' type mismatch");
    assert((!to.second || to.second.getRootType() == toTypes[to.first]) && "'to' type mismatch");
    auto [it, ins] = mapping.insert({from, FieldLocation(0, P4HIR::FieldPath())});
    it->second = to;
    return ins;
}

bool IRUtils::LocationMapping::verify() const {
    for (auto fromType : fromTypes)
        if (!fromType) return false;

    for (auto toType : toTypes)
        if (!toType) return false;

    for (auto [from, to] : mapping) {
        auto [fromIndex, fromPath] = from;
        auto [toIndex, toPath] = to;
        if (fromPath.getRootType() != fromTypes[fromIndex]) return false;
        if (toPath.getRootType() != toTypes[toIndex]) return false;
    }

    return true;
}

void IRUtils::LocationMapping::getUnmappedImpl(unsigned index, P4HIR::FieldPath path,
                                               llvm::SmallVector<FieldLocation> &result) const {
    bool hasMapping = lookup(FieldLocation(index, path));
    if (hasMapping) return;

    auto origSize = result.size();
    bool merge = true;
    for (auto childPath : path.children()) {
        auto lastSize = result.size();
        getUnmappedImpl(index, childPath, result);

        auto newCount = result.size() - lastSize;
        if (newCount != 1 || result.back().second != childPath) merge = false;
    }

    // If there's no mapped field within this path's children, then instead of adding all
    // children just add this path to minimize the output. Also handles the no-children case.
    if (merge) {
        result.resize(origSize);
        result.emplace_back(index, path);
    }
}

bool IRUtils::LocationMapping::isUniqueImpl(unsigned index, P4HIR::FieldPath path,
                                            bool hasLocation) const {
    bool hasMapping = lookup(FieldLocation(index, path));
    if (hasMapping && hasLocation) return false;

    for (auto childPath : path.children())
        if (!isUniqueImpl(index, childPath, hasMapping || hasLocation)) return false;

    return true;
}

mlir::Value IRUtils::LocationRewriter::getRef(mlir::Location loc, FieldLocation from) {
    auto to = mapping.remap(from);
    if (!to) return mlir::Value();

    auto [toIndex, toPath, toField] = *to;
    if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return mlir::Value();

    IRUtils::PathRewriter pathRewriter(rewriter, toValues[toIndex]);
    auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
    return pathRewriter.getFromPath(loc, toValues[toIndex], fullPath);
}

mlir::Value IRUtils::LocationRewriter::getVal(mlir::Location loc, FieldLocation from,
                                              bool allowSplit /* = true */) {
    if (auto to = mapping.remap(from)) {
        auto [toIndex, toPath, toField] = *to;
        IRUtils::PathRewriter pathRewriter(rewriter, toValues[toIndex]);
        auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
        mlir::Value res = pathRewriter.getFromPath(loc, toValues[toIndex], fullPath);
        if (mlir::isa<P4HIR::ReferenceType>(res.getType()))
            res = P4HIR::ReadOp::create(rewriter, loc, res);
        return res;
    } else {
        if (!allowSplit) return mlir::Value();

        auto [index, path] = from;

        // Build the read value by combining individual values for its fields.
        if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
            auto fields = llvm::map_to_vector(path.children(), [&](auto childPath) {
                return getVal(loc, FieldLocation(index, childPath), allowSplit);
            });
            return P4HIR::StructOp::create(rewriter, loc, structType, fields);
        }

        return mlir::Value();
    }
}

bool IRUtils::LocationRewriter::setVal(mlir::Location loc, FieldLocation from, mlir::Value val,
                                       bool allowSplit /* = true */) {
    if (auto to = mapping.remap(from)) {
        auto [toIndex, toPath, toField] = *to;
        if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return false;

        IRUtils::PathRewriter pathRewriter(rewriter, toValues[toIndex]);
        auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
        mlir::Value res = pathRewriter.getFromPath(loc, toValues[toIndex], fullPath);
        P4HIR::AssignOp::create(rewriter, loc, val, res);
        return true;
    } else {
        if (!allowSplit) return false;

        auto [index, path] = from;

        // Break down the assignment to the individual parts.
        if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
            for (auto field : structType.getFields()) {
                mlir::Value fieldVal = P4HIR::StructExtractOp::create(rewriter, loc, val, field);
                setVal(loc, FieldLocation(index, path[field]), fieldVal, allowSplit);
            }
            return true;
        }

        return false;
    }
}

mlir::LogicalResult IRUtils::LocationRewriter::ValueRewriter::replaceUsesIn(mlir::Operation *op) {
    if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
        auto path = getOperandWithPath(op).second;
        auto newVal = parent.getVal(op->getLoc(), FieldLocation(index, path));
        if (!newVal) return op->emitError("Cannot replace read of ") << path.str();
        rewriter.replaceOp(op, newVal);
        return mlir::success();
    } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
        auto path = getOperandWithPath(op).second;
        bool success = parent.setVal(op->getLoc(), FieldLocation(index, path), assignOp.getValue());
        if (!success) return op->emitError("Cannot replace write to ") << path.str();
        rewriter.eraseOp(op);
        return mlir::success();
    } else if (mlir::isa<P4HIR::ControlLocalOp>(op)) {
        rewriter.eraseOp(op);
        return mlir::success();
    } else {
        bool success = true;
        rewriter.modifyOpInPlace(op, [&]() {
            for (mlir::OpOperand &operand : op->getOpOperands()) {
                auto val = operand.get();
                if (auto path = getPath(val); !path.isEmpty()) {
                    FieldLocation from(index, path);
                    bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
                    auto newVal = isRef ? parent.getRef(val.getLoc(), from)
                                        : parent.getVal(val.getLoc(), from);
                    if (!newVal) {
                        op->emitError("Cannot replace access of ") << path.str();
                        success = false;
                        break;
                    }
                    operand.set(newVal);
                }
            }
        });
        return success ? mlir::success() : mlir::failure();
    }
}