// RUN: p4mlir-translate --typeinference-only %s | sed 's/__corelib = \[\]/corelib/g' | p4mlir-opt --lower-to-p4corelib | FileCheck %s

@__corelib
extern packet_in {
    void extract<T>(out T hdr);
}

header H {
    bit<8> data;
}

struct headers_t {
    H[2] stack;
}

// CHECK-LABEL: p4hir.parser @p
parser p(packet_in pkt, out headers_t hdr) {
    // CHECK: p4hir.state @start
    state start {
        // CHECK: p4hir.scope
        // CHECK: %[[stack_field_ref:.*]] = p4hir.struct_field_ref %arg1["stack"] : <!headers_t>
        // CHECK: %[[data_field_ref:.*]] = p4hir.struct_field_ref %[[stack_field_ref]]["data"] : <!hs_2xH>
        // CHECK: %[[nextIndex_field_ref:.*]] = p4hir.struct_field_ref %[[stack_field_ref]]["nextIndex"] : <!hs_2xH>
        // CHECK: %[[val:.*]] = p4hir.read %[[nextIndex_field_ref]] : <!b32i>
        // CHECK: %[[elt_ref:.*]] = p4hir.array_element_ref %[[data_field_ref]][%[[val]]] : !p4hir.ref<!arr_2xH>, !b32i
        // CHECK: %[[hdr_out_arg:.*]] = p4hir.variable ["hdr_out_arg"] : <!H>

        // CHECK: %[[nextIndex_field_ref_0:.*]] = p4hir.struct_field_ref %[[stack_field_ref]]["nextIndex"] : <!hs_2xH>
        // CHECK: %[[val_1:.*]] = p4hir.read %[[nextIndex_field_ref_0]] : <!b32i>
        // CHECK: %[[c2_b32i:.*]] = p4hir.const #int2_b32i
        // CHECK: %[[lt:.*]] = p4hir.cmp(lt, %[[val_1]] : !b32i, %[[c2_b32i]] : !b32i)
        // CHECK: %[[error:.*]] = p4hir.const #error_StackOutOfBounds
        // CHECK: p4corelib.verify %[[lt]] signalling %[[error]] : !error

        // CHECK: p4corelib.extract_header %[[hdr_out_arg]] : <!H> from %arg0 : !p4corelib.packet_in
        // CHECK: %[[val_2:.*]] = p4hir.read %[[hdr_out_arg]] : <!H>
        // CHECK: p4hir.assign %[[val_2]], %[[elt_ref]] : <!H>

        // CHECK: %[[c1_b32i:.*]] = p4hir.const #int1_b32i
        // CHECK: %[[nextIndex_field_ref_3:.*]] = p4hir.struct_field_ref %[[stack_field_ref]]["nextIndex"] : <!hs_2xH>
        // CHECK: %[[val_4:.*]] = p4hir.read %[[nextIndex_field_ref_3]] : <!b32i>
        // CHECK: %[[add:.*]] = p4hir.binop(add, %[[val_4]], %[[c1_b32i]]) : !b32i
        // CHECK: p4hir.assign %[[add]], %[[nextIndex_field_ref_3]] : <!b32i>
        pkt.extract(hdr.stack.next);
        transition accept;
    }
}
