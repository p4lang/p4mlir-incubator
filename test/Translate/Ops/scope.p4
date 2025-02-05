// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL:   p4hir.func action @scope()
action scope() {
    bool res;
    // Outer alloca
    // CHECK-NEXT: %[[RES_0:.*]] = p4hir.variable ["res"] : <!p4hir.bool>

    {
      bit<10> lhs = 1;
      bit<10> rhs = 2;

      res = lhs == rhs;
      // CHECK: p4hir.assign %{{.*}}, %[[RES_0]] : <!p4hir.bool>
      {
         res = lhs != rhs;
      // CHECK: p4hir.assign %{{.*}}, %[[RES_0]] : <!p4hir.bool>
      }
    }

    {
      bit<10> lhs = 1;
      bit<10> rhs = 2;
      // CHECK: %[[RES_1:.*]] = p4hir.variable ["res"] : <!p4hir.bit<10>>      
      bit<10> res;
      // This should store into inner res, not outer
      // CHECK: p4hir.assign %{{.*}}, %[[RES_1]] : <!p4hir.bit<10>>
      res = lhs * rhs;
    }
}
