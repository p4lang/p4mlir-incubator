include "mlir/IR/Dialect.td"

def BMv2_Dialect : Dialect {
  let name = "bmv2";
  let cppNamespace = "::p4mlir::bmv2";
}
