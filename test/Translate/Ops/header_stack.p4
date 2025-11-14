// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

header h {}
header hh {}
header_union hu {
 h h1;
 hh h2;
};

// CHECK-LABEL: p4hir.parser @p
parser p() {
    state start {
        // CHECK: %[[stack:.*]] = p4hir.variable ["stack"] : <!hs_4xh>
        h[4] stack;

        // CHECK: %[[data_field_ref:.*]] = p4hir.struct_field_ref %[[stack]]["data"] : <!hs_4xh>
        // CHECK: %[[elt_ref:.*]] = p4hir.array_element_ref %[[data_field_ref]][%{{.*}}] : !p4hir.ref<!arr_4xh>, !b32i
        // CHECK: %[[__valid_field_ref:.*]] = p4hir.struct_field_ref %[[elt_ref]]["__valid"] : <!h>
        // CHECK: p4hir.assign %{{.*}}, %[[__valid_field_ref]] : <!validity_bit>
        stack[3].setValid();

        // CHECK: %[[val:.*]] = p4hir.read %[[stack]] : <!hs_4xh>
        // CHECK: %[[data:.*]] = p4hir.struct_extract %[[val]]["data"] : !hs_4xh
        // CHECK: %[[array_elt:.*]] = p4hir.array_get %[[data]][{{.*}}] : !arr_4xh, !b32i
        h b = stack[3];
        
        // CHECK: %[[data_field_ref:.*]] = p4hir.struct_field_ref %[[stack]]["data"] : <!hs_4xh>
        // CHECK: %[[nextIndex_field_ref:.*]] = p4hir.struct_field_ref %[[stack]]["nextIndex"] : <!hs_4xh>
        // CHECK: %[[val:.*]] = p4hir.read %[[nextIndex_field_ref]] : <!b32i>
        // CHECK: %[[elt_ref:.*]] = p4hir.array_element_ref %[[data_field_ref]][%[[val]]] : !p4hir.ref<!arr_4xh>, !b32i        
        stack.next = b;

        b = stack.last;
        stack[3] = b;
        bit<32> e = stack.lastIndex;
        transition accept;
    }
}

// CHECK-LABEL: p4hir.control @c
control c() {
    apply {
        h[4] stack;
        hu[2] hustack;
        stack[3].setValid();
        h b = stack[3];
        hustack[1].h1 = hustack[0].h1;
        stack[2] = b;
        // TODO: Support header stakc operations
        // stack.push_front(2);
        // stack.pop_front(2);
        bit<32> sz = stack.size;
    }
}
