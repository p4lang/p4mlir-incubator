// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"
#include "p4mlir/Dialect/P4HIR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::IRUtils {

/// Assuming that `args` are existing arguments of `obj` then add one each new argument to `obj` for
/// each of `args`. Each new argument is placed just before the original and has the same attributes
/// and location. If `newTypes` is non-empty then these types are used for the new arguments,
/// otherwise the same types as `args` are used.
mlir::FailureOr<llvm::SmallVector<mlir::BlockArgument>> cloneFunctionLikeArgs(
    mlir::FunctionOpInterface obj, llvm::ArrayRef<mlir::BlockArgument> args,
    llvm::ArrayRef<mlir::Type> newTypes = {});

/// Assuming that `args` are existing arguments of `obj` then erase all these arguments from `obj`.
mlir::LogicalResult eraseFunctionLikeArgs(mlir::FunctionOpInterface obj,
                                          llvm::ArrayRef<mlir::BlockArgument> args);

/// Legalize `str` to a name that only contains alphanumeric characters and underscores.
std::string legalizeName(std::string str);

/// Create a struct type named `name` and with fields based on `paths`.
P4HIR::StructType createStructType(mlir::MLIRContext *ctx, const llvm::Twine &name,
                                   const llvm::SmallVectorImpl<P4HIR::FieldPath> &paths);

/// Make `val` accessible to `op` regardless of wherher `op` is inside an action.
mlir::Value getAccessibleValue(mlir::RewriterBase &rewriter, mlir::Operation *op, mlir::Value val);

// Inline `scopeOp`'s body to its parent.
void inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp);

// If `op` is an operation somewhere within `block`, check if we can split it in three parts:
// One with operations before `op`, one with `op` and one with operations after `op`.
// This function should be called to check if it's possible to use `splitBlockAt`.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split block in three parts: One with operations before `op`, one with `op` and one with
// operations after `op`. Any scopes surrounding `op` may be inlined to perform the split.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    const llvm::Twine &suffix);

// Helper class to replace one operation in a state with multiple new states.
// During init it creates two "pre" and "post" states in which the code before and after the split
// point is put. Then any number of additional states between pre and post can be created with the
// createXXXState functions. Once done the user should either call finalize to commit changes or
// cancel and any changes will be undone.
class SplitStateRewriter {
 public:
    SplitStateRewriter(mlir::RewriterBase &rewriter, mlir::Operation *op)
        : rewriter(rewriter), op(op), step(CREATED) {}
    SplitStateRewriter(SplitStateRewriter &&) = delete;
    SplitStateRewriter &operator=(SplitStateRewriter &&) = delete;
    SplitStateRewriter(const SplitStateRewriter &) = delete;
    SplitStateRewriter &operator=(const SplitStateRewriter &) = delete;
    ~SplitStateRewriter() {
        assert((step == CANCELED || step == FINALIZED) && "Incorrect state on destruction");
    }

    // Initialize and check if splitting is feasible.
    mlir::LogicalResult init();

    // Create new intermediate state that transitions to `transitionTo` and move `ops` in it.
    // If `ops` is non-null the terminator of the block is erased once moved.
    P4HIR::ParserStateOp createSubState(const llvm::Twine &suffix,
                                        P4HIR::ParserStateOp transitionTo = {},
                                        mlir::Block *ops = nullptr);

    // Same as createSubState but transition to the "post" state specifically.
    P4HIR::ParserStateOp createJoinSubState(const llvm::Twine &suffix, mlir::Block *ops = nullptr) {
        return createSubState(suffix, postState, ops);
    }

    // Undo any changes done so far.
    void cancel();

    // Finalize and commit all changes.
    void finalize();

    mlir::Operation *getSplitOp() { return op; }
    P4HIR::ParserStateOp getState() { return op->getParentOfType<P4HIR::ParserStateOp>(); }

    P4HIR::ParserStateOp getPreState() {
        assert(step == INITIALIZED);
        return preState;
    }

    P4HIR::ParserStateOp getPostState() {
        assert(step == INITIALIZED);
        return postState;
    }

    void setStateInsertionPointAfter(P4HIR::ParserStateOp afterState) {
        stateCreationPoint = afterState;
    }

 private:
    enum RewriteStep { CREATED, INITIALIZED, CANCELED, FINALIZED };

    mlir::RewriterBase &rewriter;
    mlir::Operation *op;
    RewriteStep step;

    P4HIR::ParserStateOp preState;
    P4HIR::ParserStateOp postState;
    mlir::Operation *stateCreationPoint;
};

/// Extensible helper class to walk field-accesses and their uses starting from some root value. The
/// default implementation supports walking struct field and array element with constant index
/// field-access operations.
class PathWalker {
 public:
    using IndirectUsesMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;
    using NodeCallbackFn = std::function<mlir::WalkResult(mlir::Value, P4HIR::FieldPath)>;
    using LeafCallbackFn = std::function<mlir::WalkResult(
        mlir::Operation *, const mlir::OpOperand &, P4HIR::FieldPath)>;

    ///  - `nodeCb` is the callback function for intermediate nodes. This callback is called for
    ///  field-access operations. The result value and the corresponding accessed field are provided
    ///  as arguments to the callback.
    ///  - `leafCb` is the callback function for leaf operations. This callback is called for
    ///  operations that are not field-access operations and that access a value with a known path.
    ///  The operation, the operand and the corresponding path are provided as arguments to the
    ///  callback. If a leaf operation has multiple operands with known paths then the callback will
    ///  be called multiple times.
    ///  - `indirectUses` is a mapping that provides additional uses for values that are accessed
    ///  by means other than SSA def-use chains (e.g. symbol-based access).
    PathWalker(NodeCallbackFn nodeCb, LeafCallbackFn leafCb,
               const IndirectUsesMap *indirectUses = nullptr)
        : nodeCb(nodeCb), leafCb(leafCb), indirectUses(indirectUses) {}
    virtual ~PathWalker() {}

    /// Create a `PathWalker` with a node callback.
    static PathWalker forNodes(NodeCallbackFn nodeCb,
                               const IndirectUsesMap *indirectUses = nullptr) {
        return PathWalker(nodeCb, {}, indirectUses);
    }

    /// Create a `PathWalker` with a leaf callback.
    static PathWalker forLeafs(LeafCallbackFn leafCb,
                               const IndirectUsesMap *indirectUses = nullptr) {
        return PathWalker({}, leafCb, indirectUses);
    }

    /// Return an indirect use mapping for P4HIR `control_local` and `symbol_ref`-based
    /// accesses.
    static IndirectUsesMap getIndirectSymbolUses(P4HIR::ControlOp control);

    /// Helper to look through a P4HIR `struct_extract` or `struct_field_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> lookThroughStructAccess(
        mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path);

    /// Helper to look through a P4HIR `array_get` or `array_element_ref` operation.
    static std::pair<mlir::Value, P4HIR::FieldPath> lookThroughArrayAccess(
        mlir::Operation *op, const mlir::OpOperand &operand, P4HIR::FieldPath path);

    /// `operand` is an operand in `op` that is used to access the root value's field that
    /// corresponds to `path`. If `op` is a field access operation and the resulting path can be
    /// determined based on `path`, then this function should return the resulting new value and
    /// path. Otherwise an empty value must be returned.
    virtual std::pair<mlir::Value, P4HIR::FieldPath> lookThroughUser(mlir::Operation *op,
                                                                     const mlir::OpOperand &operand,
                                                                     P4HIR::FieldPath path);

    /// Walk field-access operations and uses starting from `root`. If `rootType` is additionally
    /// provided it is used to compute field paths instead of `root.getType()`. This is useful when
    /// doing type conversions. If `rootType` is provided it must have the same shape as
    /// `root.getType()`.
    mlir::WalkResult walk(mlir::Value root, mlir::Type rootType = mlir::Type()) {
        if (!rootType) rootType = root.getType();

        P4HIR::FieldPath rootPath(P4HIR::maybeUnref(rootType));
        return walkImpl(root, rootPath);
    }

 protected:
    // Value can only be `root` or otherwise a result of a field access operation.
    mlir::WalkResult walkImpl(mlir::Value value, P4HIR::FieldPath path);

    NodeCallbackFn nodeCb;
    LeafCallbackFn leafCb;
    const IndirectUsesMap *indirectUses;
};

/// A helper class to replace uses of indexable-typed values with new values that may have
/// different layouts, using P4HIR::FieldPath. Users of an indexable-typed value may be
/// field-access operations, which produce new FieldPaths based on their operands, or leaf
/// operations which consume values.
/// The root value may either be a reference or a value of an indexable-type and the values to
/// replace are references or values accordingly.
class PathRewriter {
 public:
    using IndirectUsesMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;

    /// Full constructor.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value root, mlir::Type type,
                 mlir::Value newRoot, mlir::Type newType)
        : rewriter(rewriter), root(root), newRoot(newRoot) {
        rootType = P4HIR::maybeUnref(type);
        if (newType) newRootType = P4HIR::maybeUnref(newType);
    }

    /// Minimal constructor.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue)
        : PathRewriter(rewriter, rootValue, rootValue.getType(), mlir::Value(), mlir::Type()) {}

    /// Constructor for value replacement.
    PathRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue, mlir::Value newRootValue)
        : PathRewriter(rewriter, rootValue, rootValue.getType(), newRootValue,
                       newRootValue.getType()) {}

    PathRewriter(PathRewriter &&) = delete;
    PathRewriter &operator=(PathRewriter &&) = delete;
    PathRewriter(const PathRewriter &) = delete;
    PathRewriter &operator=(const PathRewriter &) = delete;
    virtual ~PathRewriter() {}

    /// Set the indirect use mapping to use. This mapping provides additional uses for values that
    /// are accessed by means other than SSA def-use chains (e.g. symbol-based access).
    PathRewriter &setIndirectUsesMap(const IndirectUsesMap *indirectUsesMap) {
        indirectUses = indirectUsesMap;
        return *this;
    }

    /// This function builds the necessary operations to access the field that corresponds to
    /// `field` from `value`, using `loc` as the location of access.
    virtual mlir::Value getFromField(mlir::Location loc, mlir::Value val,
                                     P4HIR::IndexedField field);

    /// This function builds the necessary operations to access the field that corresponds to
    /// `path` from `value`, using `loc` as the location of access.
    mlir::Value getFromPath(mlir::Location loc, mlir::Value val, P4HIR::FieldPath path) {
        for (auto field : path.fields()) val = getFromField(loc, val, field);
        return val;
    }

    /// `op` is an operation that has at least one operand accessing root value's fields. These
    /// operands and their paths can be found by using `getPath` on `op`'s arguments. If it is
    /// desirable to transform `op` then this should be done here. Return mlir::success() to
    /// continue the rewriting or mlir::failure() to interrupt it.
    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) { return mlir::success(); }

    /// `value` is a value used to access the root value's field that corresponds `path`. If it is
    /// desirable to replace `value` with a new equivalent, then this new value should be returned.
    /// If nothing should be changed then mlir::Value() should be returned.
    virtual mlir::Value replaceValue(mlir::Value value, P4HIR::FieldPath path) {
        return mlir::Value();
    }

    /// If `value` is accessing a field of the root value then return the corresponding field path,
    /// otherwise return the empty path.
    P4HIR::FieldPath getPath(mlir::Value value) {
        auto it = valueToPath.find(value);
        return it != valueToPath.end() ? it->second : P4HIR::FieldPath();
    }

    /// Assuming that `op` can only have a single operand with a corresponding field path, return
    /// that operand and path.
    std::pair<mlir::OpOperand *, P4HIR::FieldPath> getOperandWithPath(mlir::Operation *op);

    /// Walk `root`'s field-access and leaf users and replace values and operations using the
    /// `replace` functions. If `eraseDead` is true, then erase dead intermediate field access
    /// operations after replacing is done.
    mlir::LogicalResult transform(bool eraseDead = false);

 protected:
    mlir::RewriterBase &rewriter;

    /// The value whose uses are being replaced.
    mlir::Value root;
    /// The plain (without p4hir.ref) type to use for field paths of `root`.
    /// Note that although it needs to have the same shape as `root.getType()`, it may differ from
    /// it.
    mlir::Type rootType;
    /// Optional helper: A new root value to use while doing the replacement or `root`.
    mlir::Value newRoot;
    /// Optional helper: A plain (without p4hir.ref) new root type for `value` after replacement.
    mlir::Type newRootType;

    /// Maintains a mapping from values that are used in leaf operations to field paths.
    llvm::DenseMap<mlir::Value, P4HIR::FieldPath> valueToPath;

    /// Maintains field-access operations in safe-to-erase order, if needed.
    llvm::SmallVector<mlir::Operation *> opsToRemove;

    /// Optional mapping of indirect value uses. An indirect use is when a value is accessed in a
    /// way other than normal SSA def-use chains (e.g. through symbols).
    const IndirectUsesMap *indirectUses = nullptr;
};

/// A helper class to store remapped "locations" between a source and a target collections of types.
struct LocationMapping {
    /// Represents a FieldPath in the I-th object of a collection.
    using FieldLocation = std::pair<unsigned, P4HIR::FieldPath>;

    /// Represents a FieldPath in the I-th object of a collection, where the path is split in two
    /// parts.
    using SplitFieldLocation = std::tuple<unsigned, P4HIR::FieldPath, P4HIR::FieldPath>;

    /// Constructor for when both the source and target collection types are known in advance.
    LocationMapping(llvm::SmallVector<mlir::Type> from, llvm::SmallVector<mlir::Type> to)
        : fromTypes(std::move(from)), toTypes(std::move(to)) {}

    /// Constructor for when the target collection types will be computed later.
    LocationMapping(llvm::SmallVector<mlir::Type> from, unsigned toCount)
        : fromTypes(std::move(from)), toTypes(toCount, mlir::Type()) {}

    /// Set or change the target collection types to `to`.
    void setToTypes(llvm::SmallVector<mlir::Type> to) {
        assert((to.size() == toCount()) && "Unexpected 'to' collection size");
        toTypes = std::move(to);
    }

    /// Set or change the target collection type at `index` to `to`.
    void setToType(unsigned index, mlir::Type to) { toTypes[index] = to; }

    /// Get the types of the source collection.
    const llvm::SmallVector<mlir::Type> &getFromTypes() const { return fromTypes; }

    /// Get the types of the target collection.
    const llvm::SmallVector<mlir::Type> &getToTypes() const { return toTypes; }

    /// Get the size of the source collection.
    unsigned fromCount() const { return fromTypes.size(); }

    /// Get the size of the target collection.
    unsigned toCount() const { return toTypes.size(); }

    /// Return a mapping entry if it exists for `from`, nullptr otherwise.
    const FieldLocation *lookup(FieldLocation from) const {
        auto it = mapping.find(from);
        return (it != mapping.end()) ? &(it->second) : nullptr;
    }

    /// Return a mapping entry if it exists for `from`, nullptr otherwise.
    FieldLocation *lookup(FieldLocation from) {
        auto it = mapping.find(from);
        return (it != mapping.end()) ? &(it->second) : nullptr;
    }

    /// If `from` is a field from the source collection that exists as a whole in another place,
    /// return a split location to access it in the target collection. If different subfields of
    /// `from` are located in different places then a single target location does not exist and
    /// std::nullopt is returned.
    std::optional<SplitFieldLocation> remap(FieldLocation from) const;

    /// Record that `from` in the source collection should be remapped to `to` in the target
    /// collection.
    bool addMapping(FieldLocation from, FieldLocation to);

    /// Record that `from` in the source collection should be remapped to the `to`-ith target
    /// collection without yet specifying the exact mapping.
    bool markForMapping(FieldLocation from, unsigned to) {
        return addMapping(from, FieldLocation(to, P4HIR::FieldPath()));
    }

    /// Record that the locations defined by `fromIndex` and `fromPaths` in the source collection
    /// should be remapped to the locations defined by `toIndex` and `toPaths` in the target
    /// collection.
    void addMapping(unsigned fromIndex, llvm::SmallVectorImpl<P4HIR::FieldPath> &fromPaths,
                    unsigned toIndex, llvm::SmallVectorImpl<P4HIR::FieldPath> &toPaths) {
        assert((fromPaths.size() == toPaths.size()) && "Expected same-size path collections");
        for (auto [fromPath, toPath] : llvm::zip_equal(fromPaths, toPaths))
            addMapping(FieldLocation(fromIndex, fromPath), FieldLocation(toIndex, toPath));
    }

    /// Get a range for all paths in the `index`-th object in the source collection.
    auto getFromPaths(unsigned index) const {
        auto filteredKeys = llvm::make_filter_range(
            mapping.keys(), [index](const auto &k) { return k.first == index; });
        return llvm::map_range(filteredKeys, [](const auto &k) { return k.second; });
    }

    /// Get a range for all mapping entries.
    auto getMapping() const { return llvm::make_range(mapping.begin(), mapping.end()); }

    /// Get a range for all mapping entries regarding the `index`-th object in the source
    /// collection.
    auto getMapping(unsigned index) const {
        return llvm::make_filter_range(
            getMapping(), [index](const auto &e) { return e.getFirst().first == index; });
    }

    /// Returns a minimal vector of all FieldLocations that need to be mapped to create a complete
    /// mapping.
    llvm::SmallVector<FieldLocation> getUnmapped() const {
        llvm::SmallVector<FieldLocation> result;
        for (auto [index, type] : llvm::enumerate(fromTypes))
            getUnmappedImpl(index, P4HIR::FieldPath(type), result);

        return result;
    }

    /// Returns a minimal vector of all FieldLocations that need to be mapped to create a complete
    /// mapping of the `index`-th source object.
    llvm::SmallVector<FieldLocation> getUnmapped(unsigned index) const {
        llvm::SmallVector<FieldLocation> result;
        getUnmappedImpl(index, P4HIR::FieldPath(fromTypes[index]), result);
        return result;
    }

    /// Return true iff this mapping is complete. A complete mapping is one where all source leaf
    /// fields can be remapped in the target location.
    bool isComplete() const { return getUnmapped().empty(); }

    /// Return true iff this mapping is unique. A unique mapping is one where each source field has
    /// only one target location to be remapped at.
    bool isUnique() const {
        for (auto [index, type] : llvm::enumerate(fromTypes))
            if (!isUniqueImpl(index, P4HIR::FieldPath(type), false)) return false;

        return true;
    }

    /// Check that the source and target types and paths are valid.
    bool verify() const;

 protected:
    void getUnmappedImpl(unsigned index, P4HIR::FieldPath path,
                         llvm::SmallVector<FieldLocation> &result) const;

    bool isUniqueImpl(unsigned index, P4HIR::FieldPath path, bool hasLocation) const;

    /// Represents a location mapping between the source and the target collections.
    llvm::DenseMap<FieldLocation, FieldLocation> mapping;

    llvm::SmallVector<mlir::Type> fromTypes;
    llvm::SmallVector<mlir::Type> toTypes;
};

/// A helper class to replace uses of a source collection to ones in a target collections of values
/// using a corresponding `LocationMapping`.
struct LocationRewriter {
    using FieldLocation = LocationMapping::FieldLocation;

    LocationRewriter(mlir::RewriterBase &rewriter, LocationMapping &mapping,
                     llvm::SmallVector<mlir::Value> from, llvm::SmallVector<mlir::Value> to)
        : rewriter(rewriter),
          mapping(mapping),
          fromValues(std::move(from)),
          toValues(std::move(to)) {
        assert((fromValues.size() == mapping.fromCount()) &&
               "Value and type size mismatch for the source collection");
        assert((toValues.size() == mapping.toCount()) &&
               "Value and type size mismatch for the target collection");
        for (auto [value, type] : llvm::zip_equal(fromValues, mapping.getFromTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the source collection");
        for (auto [value, type] : llvm::zip_equal(toValues, mapping.getToTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the target collection");
    }

    virtual ~LocationRewriter() {}

    /// Return a remapped reference for `from` if possible, otherwise return an empty value.
    virtual mlir::Value getRef(mlir::Location loc, FieldLocation from);

    /// Return a remapped value for `from` if possible, otherwise return an empty value.
    /// If `allowSplit` is true then multiple values from different places in the target
    /// collection may be combined to do so.
    virtual mlir::Value getVal(mlir::Location loc, FieldLocation from, bool allowSplit = true);

    /// Perform a remapped assignment for `from` if possible and return true, otherwise return
    /// false. If `allowSplit` is true then the assignment may be broken into multiple assignments
    /// in the target collection to do so.
    virtual bool setVal(mlir::Location loc, FieldLocation from, mlir::Value val,
                        bool allowSplit = true);

    /// Replace all uses of the source collection with ones from the target collection.
    mlir::LogicalResult transform() {
        for (unsigned index = 0, e = fromValues.size(); index < e; index++)
            if (failed(ValueRewriter(*this, index).transform(true))) return mlir::failure();

        return mlir::success();
    }

 protected:
    struct ValueRewriter : public IRUtils::PathRewriter {
        ValueRewriter(LocationRewriter &parent, unsigned index)
            : PathRewriter(parent.rewriter, parent.fromValues[index]),
              parent(parent),
              index(index) {}

        virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) override;

        LocationRewriter &parent;
        unsigned index;
    };

    mlir::RewriterBase &rewriter;
    LocationMapping &mapping;
    llvm::SmallVector<mlir::Value> fromValues;
    llvm::SmallVector<mlir::Value> toValues;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
