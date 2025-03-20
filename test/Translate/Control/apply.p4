// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-LABEL:   p4hir.control @InnerPipe(%arg0: !b10i {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "arg1"}, %arg1: !i16i {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "arg2"}, %arg2: !p4hir.ref<!i16i> {p4hir.dir = #p4hir<dir out>, p4hir.param_name = "arg3"})()
control InnerPipe(bit<10> arg1, in int<16> arg2, out int<16> arg3) {
    apply {}
}

// CHECK-LABEL:   p4hir.control @Pipe
control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4) {
// CHECK:     %[[inner:.*]] = p4hir.instantiate @InnerPipe() as "inner" : () -> !InnerPipe
    InnerPipe() inner;

    action bar() {
        int<16> x1;
        return;
    }

    apply {
        bar();
        int<16> x1;
// CHECK:         p4hir.apply %[[inner]](%{{.*}}, %{{.*}}, %{{.*}}) : !InnerPipe        
        inner.apply(1, 2, x1);
    }
}
