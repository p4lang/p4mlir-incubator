set(LLVM_TARGET_DEFINITIONS P4HIR.td)
mlir_tablegen(P4HIR_Dialect.h.inc -gen-dialect-decls -dialect=p4hir)
mlir_tablegen(P4HIR_Dialect.cpp.inc -gen-dialect-defs -dialect=p4hir)
mlir_tablegen(P4HIR_Ops.h.inc -gen-op-decls)
mlir_tablegen(P4HIR_Ops.cpp.inc -gen-op-defs)
mlir_tablegen(P4HIR_Types.h.inc -gen-typedef-decls -typedefs-dialect=p4hir)
mlir_tablegen(P4HIR_Types.cpp.inc -gen-typedef-defs -typedefs-dialect=p4hir)

# Generate extra headers for custom enum and attrs.
mlir_tablegen(P4HIR_OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(P4HIR_OpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(P4HIR_Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=p4hir)
mlir_tablegen(P4HIR_Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=p4hir)

add_public_tablegen_target(P4MLIR_P4HIR_IncGen)
add_dependencies(mlir-headers P4MLIR_P4HIR_IncGen)
