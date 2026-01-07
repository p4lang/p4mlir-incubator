include "BMv2Dialect.td"
include "mlir/IR/OpBase.td"

def BMv2_V1SwitchOp : Op<"v1switch", [
  Symbol
]> {
  let summary = "BMv2 V1Switch package instantiation";

  let description = [{
    Explicit representation of a BMv2 V1Switch package.
    Exposes parser, controls, and deparser directly.
  }];

  let arguments = (ins
    AnyType:$parser,
    AnyType:$verify,
    AnyType:$ingress,
    AnyType:$egress,
    AnyType:$update,
    AnyType:$deparser
  );

  let assemblyFormat = [{
    $parser `,` $verify `,` $ingress `,`
    $egress `,` $update `,` $deparser
    attr-dict
  }];
}
