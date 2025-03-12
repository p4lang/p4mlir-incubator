// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

// CHECK-LABEL:  p4hir.control @InnerPipe(%arg0: !b10i, %arg1: !i16i, %arg2: !p4hir.ref<!i16i>)(flag: !p4hir.bool) {
control InnerPipe(bit<10> arg1, in int<16> arg2, out int<16> arg3)(bool flag) {
    apply {}
}

// CHECK-LABEL:   p4hir.control @Pipe(%arg0: !b10i, %arg1: !i16i, %arg2: !p4hir.ref<!i16i>, %arg3: !p4hir.ref<!i16i>)(ctr_arg1: !i16i, hdr_arg: !MyHeader) {
control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4)(int<16> ctr_arg1, MyHeader hdr_arg) {
    InnerPipe(true) inner1;
    InnerPipe(false) inner2;
    // CHECK: %[[inner1:.*]] = p4hir.instantiate @InnerPipe(%true) as "inner1" : (!p4hir.bool) -> !InnerPipe
    // CHECK: %[[inner2:.*]] = p4hir.instantiate @InnerPipe(%false) as "inner2" : (!p4hir.bool) -> !InnerPipe
    
    action bar() {
        int<16> x1;
        return;
    }

    apply {
        bar();
        int<16> x1;
        // CHECK: p4hir.apply %[[inner1]](%{{.*}}, %{{.*}}, %{{.*}}) : !InnerPipe
        inner1.apply(1, ctr_arg1, x1);
    }
}
