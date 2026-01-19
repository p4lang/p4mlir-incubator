#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Dialect.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Ops.h"
#include "p4mlir/Dialect/BMv2IR/BMv2IR_Types.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

#define DEBUG_TYPE "lower-to-header-instance"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERTOHEADERINSTANCE
#include "p4mlir/Conversion/P4HIRToBMv2IR/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

P4HIR::StructType isStructOrRefToStruct(mlir::Type ty) {
    if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
        return isStructOrRefToStruct(refTy.getObjectType());
    auto structTy = dyn_cast<P4HIR::StructType>(ty);
    if (!structTy) return nullptr;
    // We assume that struct don't contain other structs
    assert(
        llvm::none_of(structTy.getFields(),
                      [](P4HIR::FieldInfo field) { return isa<P4HIR::StructType>(field.type); }) &&
        "No structs within structs");
    return structTy;
}

P4HIR::HeaderType isHeaderOrRefToHeader(mlir::Type ty) {
    if (auto refTy = dyn_cast<P4HIR::ReferenceType>(ty))
        return isHeaderOrRefToHeader(refTy.getObjectType());
    if (auto headerTy = dyn_cast<P4HIR::HeaderType>(ty)) return headerTy;
    return nullptr;
}

// Map for <struct type name, field name> -> header instance
using InstanceConversionContext =
    std::map<std::pair<mlir::Type *, std::string>, BMv2IR::HeaderInstanceOp>;
// Map used for raw headers in control/parser args
using HeaderConversionContext = llvm::DenseMap<P4HIR::HeaderType, BMv2IR::HeaderInstanceOp>;

struct StructUses {
    // FieldRefs accessessing header fields
    SmallVector<Operation *> fieldRefs;
    // FieldRefs accessing bit fields
    SmallVector<Operation *> bitRefs;
};

LogicalResult handleFieldAccess(StringRef name, Operation *op, P4HIR::StructType structTy,
                                StructUses &structUses) {
    auto fieldTy = structTy.getFieldType(name);
    return llvm::TypeSwitch<Type, LogicalResult>(fieldTy)
        .Case([&](P4HIR::HeaderType) {
            structUses.fieldRefs.push_back(op);
            return success();
        })
        .Case<P4HIR::BitsType, P4HIR::VarBitsType>([&](mlir::Type) {
            structUses.bitRefs.push_back(op);
            return success();
        })
        .Default([&](Type) { return op->emitError("Unsupported field access"); });

    return success();
};

LogicalResult handleStructUse(OpOperand &use, P4HIR::StructType structTy, StructUses &structUses,
                              SmallVector<Operation *> &eraseList) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(use.getOwner())
        .Case([&](P4HIR::StructFieldRefOp fieldRefOp) {
            return handleFieldAccess(fieldRefOp.getFieldName(), fieldRefOp, structTy, structUses);
        })
        .Case([&](P4HIR::StructExtractOp extractOp) {
            return handleFieldAccess(extractOp.getFieldName(), extractOp, structTy, structUses);
        })
        .Case([&](P4HIR::ReadOp readOp) -> LogicalResult {
            for (auto readUser : readOp.getResult().getUsers()) {
                auto extract = dyn_cast<P4HIR::StructExtractOp>(readUser);
                if (!extract) return readUser->emitError("Unsupported read use");
                if (failed(
                        handleFieldAccess(extract.getFieldName(), extract, structTy, structUses)))
                    return failure();
            }
            // Explicitly remove readOp to avoid unrealized_casts
            eraseList.push_back(readOp);
            return success();
        })
        .Case([&](P4HIR::TableApplyOp tableApplyOp) -> LogicalResult {
            // Table apply ops "call" table keys and pass control args to the table as arguments,
            // but we want the key argument to become the same header instance as the control
            // argument.
            auto argIndex = use.getOperandNumber();
            auto moduleOp = tableApplyOp->getParentOfType<ModuleOp>();
            if (!moduleOp) return tableApplyOp->emitError("No parent ModuleOp");
            auto tableOp = dyn_cast<P4HIR::TableOp>(
                mlir::SymbolTable::lookupSymbolIn(moduleOp, tableApplyOp.getTable()));
            if (!tableOp) return tableApplyOp->emitError("No table");
            P4HIR::TableKeyOp tableKey;
            tableOp->walk([&](P4HIR::TableKeyOp keyOp) {
                assert(!tableKey && "Multiple table keys?");
                tableKey = keyOp;
            });
            if (!tableKey) return tableApplyOp->emitError("No table key");
            auto blockArg = tableKey.getBody().getArgument(argIndex);
            for (auto &use : blockArg.getUses()) {
                if (failed(handleStructUse(use, structTy, structUses, eraseList)))
                    return tableKey->emitError("Error processing block argument at index ")
                           << argIndex;
            }
            return success();
        })
        .Default([](Operation *op) { return op->emitError("Unsupported struct use"); });
    return success();
}

FailureOr<StructUses> findFieldRefs(P4HIR::ControlLocalOp controlLocal, P4HIR::StructType structTy,
                                    SmallVector<Operation *> &eraseList) {
    auto parent = controlLocal->getParentOfType<P4HIR::ControlOp>();
    auto moduleOp = controlLocal->getParentOfType<ModuleOp>();
    if (!parent || !moduleOp) return {};
    StructUses structUses;
    auto walkRes = parent->walk([&](P4HIR::SymToValueOp symRef) {
        auto decl = symRef.getDecl();
        auto op = mlir::SymbolTable::lookupSymbolIn(moduleOp, decl);
        if (op == controlLocal.getOperation()) {
            for (auto &use : symRef->getUses()) {
                if (failed(handleStructUse(use, structTy, structUses, eraseList)))
                    return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    if (walkRes.wasInterrupted()) return failure();

    return structUses;
}

// Adds instances from a StructType, splitting the struct to create separate instances for
// the header fields, and creating a new struct containing only the bit fields if necessary
LogicalResult splitStructAndAddInstances(Value val, P4HIR::StructType structTy, Location loc,
                                         StringRef parentName, ModuleOp moduleOp,
                                         PatternRewriter &rewriter,
                                         InstanceConversionContext *instances) {
    auto ctx = rewriter.getContext();
    llvm::StringMap<BMv2IR::HeaderInstanceOp> localInstance;
    auto getOrInsertInstance = [&localInstance, &rewriter, &instances, &moduleOp, &loc,
                                &parentName](P4HIR::StructType ty, StringRef name,
                                             bool referenceFullStruct) -> BMv2IR::HeaderInstanceOp {
        auto instanceTy = referenceFullStruct ? P4HIR::ReferenceType::get(ty)
                                              : P4HIR::ReferenceType::get(ty.getFieldType(name));
        auto symName = referenceFullStruct ? rewriter.getStringAttr(ty.getName())
                                           : rewriter.getStringAttr(ty.getName() + "_" + name);
        if (instances) {
            auto it = instances->find({&ty, name.str()});
            if (it != instances->end()) {
                return it->second;
            }
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            auto instanceOp = rewriter.create<BMv2IR::HeaderInstanceOp>(loc, symName, instanceTy);
            instances->insert({{&ty, name.str()}, instanceOp});
            return instanceOp;
        }
        // We don't want to use the global context for Header Instances (e.g. we are lowering a
        // P4HIR::Variable) We still need to avoid adding duplicate HeaderInstances if the same
        // field of a variable is accessed multiple times, so we use the "local" map
        auto it = localInstance.find(name);
        if (it != localInstance.end()) return it->second;
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto instanceOp = rewriter.create<BMv2IR::HeaderInstanceOp>(
            loc, rewriter.getStringAttr(parentName + "_" + name), instanceTy);
        localInstance.insert({name, instanceOp});
        return instanceOp;
    };

    StructUses structUses;
    SmallVector<Operation *> eraseList;
    P4HIR::ControlLocalOp controlLocal = nullptr;

    // Find the StructFieldRefOps/StructExtractOps that access the struct
    for (auto &use : val.getUses()) {
        auto user = use.getOwner();
        if (auto cLocal = dyn_cast<P4HIR::ControlLocalOp>(user)) {
            if (controlLocal != nullptr) {
                return emitError(loc, "Expected at most one ControlLocalOp for every argument");
            }
            controlLocal = cLocal;
            continue;
        }
        if (failed(handleStructUse(use, structTy, structUses, eraseList))) return failure();
    }

    // Add uses coming from ControlLocalOp
    if (controlLocal) {
        const auto maybeRefs = findFieldRefs(controlLocal, structTy, eraseList);
        if (failed(maybeRefs))
            return controlLocal->emitError("Error while processing control local op");
        auto controlLocalUses = maybeRefs.value();
        for (auto op : controlLocalUses.fieldRefs) structUses.fieldRefs.push_back(op);
        for (auto op : controlLocalUses.bitRefs) structUses.bitRefs.push_back(op);
    }

    // Add HeaderInstanceOps for StructFieldRefOps that reference header fields
    for (auto op : structUses.fieldRefs) {
        StringRef name =
            llvm::TypeSwitch<Operation *, StringRef>(op)
                .Case([](P4HIR::StructFieldRefOp fieldRefOp) { return fieldRefOp.getFieldName(); })
                .Case([](P4HIR::StructExtractOp extractOp) { return extractOp.getFieldName(); });
        auto instanceOp = getOrInsertInstance(structTy, name.str(), false);
        rewriter.setInsertionPointAfter(op);
        Operation *newOp =
            rewriter.create<BMv2IR::SymToValueOp>(op->getLoc(), instanceOp.getHeaderType(),
                                                  SymbolRefAttr::get(ctx, instanceOp.getSymName()));
        if (isa<P4HIR::StructExtractOp>(op)) {
            auto fieldTy =
                cast<P4HIR::ReferenceType>(newOp->getResult(0).getType()).getObjectType();
            newOp = rewriter.create<P4HIR::ReadOp>(op->getLoc(), fieldTy, newOp->getResult(0));
        }
        rewriter.replaceOp(op, newOp->getResult(0));
    }

    if (structUses.bitRefs.empty()) {
        for (auto op : eraseList) rewriter.eraseOp(op);
        return success();
    }

    // Since the struct has bit fields, we create a new type dropping the header fields, and add a
    // header instance for it. Note that this behaves differently from p4c, which creates a single
    // header instance containing all the scalars (and separate header instance for every varbit
    // since an header instance can contain only one varbit field).

    SmallVector<P4HIR::FieldInfo> bitFields;
    unsigned totalSize = 0;
    for (auto field : structTy.getFields()) {
        if (auto bitTy = dyn_cast<P4HIR::BitsType>(field.type)) {
            totalSize += bitTy.getWidth();
            bitFields.push_back(field);
        } else if (auto varBitTy = dyn_cast<P4HIR::VarBitsType>(field.type)) {
            totalSize += varBitTy.getMaxWidth();
            bitFields.push_back(field);
        }
    }
    // Add a padding field if necessary
    unsigned padding = totalSize % 8;
    if (padding != 0) {
        auto padTy = P4HIR::BitsType::get(ctx, 8 - padding, false);
        P4HIR::FieldInfo padInfo{rewriter.getStringAttr("_padding"), padTy};
        bitFields.push_back(padInfo);
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto newTy = P4HIR::StructType::get(rewriter.getContext(), structTy.getName(), bitFields,
                                        structTy.getAnnotations());
    auto newInstance = getOrInsertInstance(newTy, newTy.getName(), true);
    for (auto op : structUses.bitRefs) {
        StringRef name =
            llvm::TypeSwitch<Operation *, StringRef>(op)
                .Case([](P4HIR::StructFieldRefOp fieldRefOp) { return fieldRefOp.getFieldName(); })
                .Case([](P4HIR::StructExtractOp extractOp) { return extractOp.getFieldName(); });
        rewriter.setInsertionPointAfter(op);
        auto symToVal = rewriter.create<BMv2IR::SymToValueOp>(
            op->getLoc(), newInstance.getHeaderType(),
            SymbolRefAttr::get(ctx, newInstance.getSymName()));
        if (isa<P4HIR::StructExtractOp>(op)) {
            auto readOp = rewriter.create<P4HIR::ReadOp>(
                loc, cast<P4HIR::ReferenceType>(symToVal.getType()).getObjectType(), symToVal);
            rewriter.replaceOpWithNewOp<P4HIR::StructExtractOp>(op, readOp, name);
        } else {
            rewriter.replaceOpWithNewOp<P4HIR::StructFieldRefOp>(op, symToVal, name);
        }
    }

    for (auto op : eraseList) rewriter.eraseOp(op);
    return success();
}

bool tableApplyInForwardSlice(Value val) {
    SetVector<Operation *> slice;
    getForwardSlice(val, &slice);
    return llvm::any_of(slice, [](Operation *op) { return isa<P4HIR::TableApplyOp>(op); });
}

LogicalResult addInstanceForHeader(P4HIR::VariableOp op, P4HIR::HeaderType headerTy, StringRef name,
                                   PatternRewriter &rewriter) {
    if (tableApplyInForwardSlice(op.getResult())) return op.emitError("Handling in TableKey NYI");
    PatternRewriter::InsertionGuard guard(rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Expected parent module");
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto uniqueName = BMv2IR::getUniqueNameInParentModule(op, name + "_var");

    auto instance = rewriter.create<BMv2IR::HeaderInstanceOp>(op->getLoc(), uniqueName,
                                                              P4HIR::ReferenceType::get(headerTy));
    rewriter.setInsertionPointAfter(op);
    rewriter.replaceOpWithNewOp<BMv2IR::SymToValueOp>(
        op, instance.getHeaderType(),
        SymbolRefAttr::get(rewriter.getContext(), instance.getSymName()));

    return success();
}

LogicalResult addInstanceForScalarVariable(P4HIR::VariableOp op, P4HIR::BitsType bitTy,
                                           StringRef name, PatternRewriter &rewriter) {
    auto varName = BMv2IR::getUniqueNameInParentModule(op, name + "_var");
    auto fieldName = rewriter.getStringAttr(name);
    P4HIR::FieldInfo info(fieldName, bitTy);
    auto headerWrapper = P4HIR::HeaderType::get(rewriter.getContext(), varName, {info});

    PatternRewriter::InsertionGuard guard(rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Expected parent module");
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto instance = rewriter.create<BMv2IR::HeaderInstanceOp>(
        op->getLoc(), varName, P4HIR::ReferenceType::get(headerWrapper));

    rewriter.setInsertionPointAfter(op);
    auto symToVal = rewriter.create<BMv2IR::SymToValueOp>(
        op.getLoc(), instance.getHeaderType(),
        SymbolRefAttr::get(rewriter.getContext(), instance.getSymName()));

    // Handle uses of the original variable "across" table_apply (we also need to use header
    // instances in the corresponding table_key This is a fairly limited pattern that handles simple
    // cases like:
    //  %var = p4hir.variable ["var"]
    //  %read = p4hir.read %_switch_0_key
    //  p4hir.table_apply @table with key(%read)
    //  TODO: handle more complex cases

    for (auto user : op->getUsers()) {
        auto readOp = dyn_cast<P4HIR::ReadOp>(user);
        if (!readOp) continue;
        for (auto &use : readOp->getUses()) {
            PatternRewriter::InsertionGuard guard(rewriter);
            auto tableApplyOp = dyn_cast<P4HIR::TableApplyOp>(use.getOwner());
            if (!tableApplyOp) continue;
            auto argNum = use.getOperandNumber();
            // Retrieve the corresponding table_key
            auto tableOp = cast<P4HIR::TableOp>(
                SymbolTable::lookupSymbolIn(moduleOp, tableApplyOp.getTable()));
            // TODO: use helper once added
            P4HIR::TableKeyOp keyOp;
            tableOp.walk([&](P4HIR::TableKeyOp tkOp) { keyOp = tkOp; });
            assert(keyOp && "Expected keyop");
            // Insert SymbolToValue -> ReadOp
            auto &keyBlock = keyOp.getRegion().front();
            rewriter.setInsertionPointToStart(&keyBlock);
            auto keySymVal = rewriter.create<BMv2IR::SymToValueOp>(
                op.getLoc(), instance.getHeaderType(),
                SymbolRefAttr::get(rewriter.getContext(), instance.getSymName()));
            auto fRef =
                rewriter.create<P4HIR::StructFieldRefOp>(keyOp.getLoc(), keySymVal, fieldName);
            auto readOp = rewriter.create<P4HIR::ReadOp>(keyOp.getLoc(), fRef);
            rewriter.replaceAllUsesWith(keyOp.getRegion().front().getArgument(argNum), readOp);
        }
    }
    rewriter.replaceOpWithNewOp<P4HIR::StructFieldRefOp>(op, symToVal, fieldName);

    return success();
}

LogicalResult addInstanceForHeader(BlockArgument arg, P4HIR::HeaderType headerTy, ModuleOp moduleOp,
                                   PatternRewriter &rewriter, HeaderConversionContext *instances) {
    if (tableApplyInForwardSlice(arg))
        return arg.getParentRegion()->getParentOp()->emitError("Handling in TableKey NYI");
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    BMv2IR::HeaderInstanceOp newInstance;
    auto it = instances->find(headerTy);
    if (it != instances->end()) {
        newInstance = it->second;
    } else {
        newInstance = rewriter.create<BMv2IR::HeaderInstanceOp>(
            arg.getLoc(), rewriter.getStringAttr(headerTy.getName()),
            P4HIR::ReferenceType::get(headerTy));
        instances->insert({headerTy, newInstance});
    }

    SmallVector<std::pair<Operation *, unsigned>> uses;
    for (auto &use : arg.getUses()) {
        uses.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
    }
    for (auto &use : uses) {
        Operation *user = use.first;
        rewriter.setInsertionPointToStart(user->getBlock());
        auto opIndex = use.second;
        auto symRef = rewriter.create<BMv2IR::SymToValueOp>(
            user->getLoc(), newInstance.getHeaderType(),
            SymbolRefAttr::get(rewriter.getContext(), newInstance.getSymName()));
        user->setOperand(opIndex, symRef.getResult());
    }

    return success();
}

struct AssignBreakdownPattern : public OpRewritePattern<P4HIR::AssignOp> {
    using OpRewritePattern<P4HIR::AssignOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::AssignOp assignOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto loc = assignOp.getLoc();
        auto ref = assignOp.getRef();
        auto val = assignOp.getValue();
        auto valTy = val.getType();
        if (!isStructWithHeaders(valTy)) return failure();
        auto structTy = cast<P4HIR::StructType>(valTy);
        if (llvm::any_of(structTy.getFields(),
                         [](P4HIR::FieldInfo info) { return isa<P4HIR::BitsType>(info.type); }))
            return assignOp.emitError("Bit fields not supported");
        for (auto field : structTy.getFields()) {
            auto fieldRef = rewriter.create<P4HIR::StructFieldRefOp>(loc, ref, field.name);
            auto fieldVal = rewriter.create<P4HIR::StructExtractOp>(loc, val, field.name);
            rewriter.create<P4HIR::AssignOp>(loc, fieldVal, fieldRef);
        }
        rewriter.eraseOp(assignOp);

        return mlir::success();
    }

    static bool isStructWithHeaders(Type ty) {
        auto refTy = dyn_cast<P4HIR::ReferenceType>(ty);
        auto typeToProcess = ty;
        if (refTy) typeToProcess = refTy.getObjectType();
        auto structTy = dyn_cast<P4HIR::StructType>(ty);
        if (!structTy) return false;
        return llvm::any_of(structTy.getFields(), [](P4HIR::FieldInfo info) {
            return isa<P4HIR::HeaderType>(info.type);
        });
    }
};

struct ParserOpPattern : public OpRewritePattern<P4HIR::ParserOp> {
    ParserOpPattern(MLIRContext *context, InstanceConversionContext *instances,
                    HeaderConversionContext *instancesFromHeaderArgs)
        : OpRewritePattern<P4HIR::ParserOp>(context),
          instances(instances),
          instancesFromHeaderArgs(instancesFromHeaderArgs) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserOp parserOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto moduleOp = parserOp->getParentOfType<ModuleOp>();
        if (!moduleOp) return failure();
        for (auto &arg : parserOp.getArguments()) {
            auto ty = arg.getType();
            std::string parentName =
                (parserOp.getSymName() + std::to_string(arg.getArgNumber())).str();
            if (auto headerTy = isHeaderOrRefToHeader(ty)) {
                if (failed(addInstanceForHeader(arg, headerTy, moduleOp, rewriter,
                                                instancesFromHeaderArgs)))
                    return parserOp->emitError("Failed to process parserOp");
            } else if (auto structTy = isStructOrRefToStruct(ty)) {
                if (failed(splitStructAndAddInstances(arg, structTy, parserOp.getLoc(), parentName,
                                                      moduleOp, rewriter, instances)))
                    return parserOp->emitError("Failed to process parserOp");
            }
        }

        return mlir::success();
    }

 private:
    InstanceConversionContext *instances;
    HeaderConversionContext *instancesFromHeaderArgs;
};

struct ControlOpPattern : public OpRewritePattern<P4HIR::ControlOp> {
    ControlOpPattern(MLIRContext *context, InstanceConversionContext *instances,
                     HeaderConversionContext *instancesFromHeaderArgs)
        : OpRewritePattern<P4HIR::ControlOp>(context),
          instances(instances),
          instancesFromHeaderArgs(instancesFromHeaderArgs) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::ControlOp controlOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto moduleOp = controlOp->getParentOfType<ModuleOp>();
        if (!moduleOp) return failure();
        for (auto &arg : controlOp.getArguments()) {
            auto ty = arg.getType();
            std::string parentName =
                (controlOp.getSymName() + std::to_string(arg.getArgNumber())).str();
            if (auto headerTy = isHeaderOrRefToHeader(ty)) {
                if (failed(addInstanceForHeader(arg, headerTy, moduleOp, rewriter,
                                                instancesFromHeaderArgs)))
                    return controlOp->emitError("Failed to process ControlOp");
            } else if (auto structTy = isStructOrRefToStruct(ty)) {
                if (failed(splitStructAndAddInstances(arg, structTy, controlOp.getLoc(), parentName,
                                                      moduleOp, rewriter, instances)))
                    return controlOp->emitError("Failed to process ControlOp");
            }
        }
        // Remove ControlLocalOp and P4HIR::SymToValueOp since they are unused at this point
        SmallVector<Operation *> eraseList;
        controlOp.walk(
            [&](P4HIR::ControlLocalOp controlLocal) { eraseList.push_back(controlLocal); });
        controlOp.walk([&](P4HIR::SymToValueOp symRef) { eraseList.push_back(symRef); });
        for (auto op : eraseList) rewriter.eraseOp(op);

        return mlir::success();
    }

 private:
    InstanceConversionContext *instances;
    HeaderConversionContext *instancesFromHeaderArgs;
};

struct VariableOpPattern : public OpRewritePattern<P4HIR::VariableOp> {
    using OpRewritePattern<P4HIR::VariableOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp variableOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto moduleOp = variableOp->getParentOfType<ModuleOp>();
        auto refTy = variableOp.getType();
        auto ty = refTy.getObjectType();
        auto maybeName = variableOp.getName();
        if (!maybeName.has_value())
            return variableOp.emitError("Unnamed variable can't be lowered to header instance");
        auto name = maybeName.value();

        auto res =
            TypeSwitch<Type, LogicalResult>(ty)
                .Case([&](P4HIR::StructType structTy) -> LogicalResult {
                    if (failed(splitStructAndAddInstances(variableOp.getResult(), structTy,
                                                          variableOp.getLoc(), name, moduleOp,
                                                          rewriter, nullptr)))
                        return variableOp.emitError("Error translating variableOp");
                    return success();
                })
                .Case([&](P4HIR::HeaderType headerTy) -> LogicalResult {
                    if (failed(addInstanceForHeader(variableOp, headerTy, name, rewriter)))
                        return variableOp.emitError("Error translating variableOp");
                    return success();
                })
                .Case([&](P4HIR::BitsType bitTy) -> LogicalResult {
                    if (failed(addInstanceForScalarVariable(variableOp, bitTy, name, rewriter)))
                        return variableOp.emitError("Error translating variableOp");
                    return success();
                })
                .Default([&](Type ty) -> LogicalResult {
                    return variableOp.emitError("Unsupported variable type");
                });
        return res;
    }
};

struct LowerToHeaderInstancePass
    : public P4::P4MLIR::impl::LowerToHeaderInstanceBase<LowerToHeaderInstancePass> {
    void runOnOperation() override {
        auto &context = getContext();
        // First we break down Assign ops that work on full structs into assigns on the fields
        {
            RewritePatternSet patterns(&context);
            ConversionTarget target(context);
            patterns.add<AssignBreakdownPattern>(&context);

            if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
                signalPassFailure();
        }
        RewritePatternSet patterns(&context);
        ConversionTarget target(context);
        target.addLegalDialect<BMv2IR::BMv2IRDialect>();
        target.addLegalDialect<P4HIR::P4HIRDialect>();
        target.addLegalDialect<P4CoreLib::P4CoreLibDialect>();
        target.addDynamicallyLegalOp<P4HIR::ParserOp>([](P4HIR::ParserOp parserOp) {
            auto argsTy = parserOp.getArgumentTypes();
            return !llvm::any_of(argsTy, [](mlir::Type ty) {
                return isHeaderOrRefToHeader(ty) || isStructOrRefToStruct(ty);
            });
        });
        target.addDynamicallyLegalOp<P4HIR::ControlOp>([](P4HIR::ControlOp controlOp) {
            auto argsTy = controlOp.getArgumentTypes();
            return !llvm::any_of(argsTy, [](mlir::Type ty) {
                return isHeaderOrRefToHeader(ty) || isStructOrRefToStruct(ty);
            });
        });
        target.addIllegalOp<P4HIR::VariableOp>();

        InstanceConversionContext instances;
        HeaderConversionContext instancesFromHeaderArgs;
        patterns.add<VariableOpPattern>(patterns.getContext());
        patterns.add<ParserOpPattern, ControlOpPattern>(patterns.getContext(), &instances,
                                                        &instancesFromHeaderArgs);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

}  // anonymous namespace
