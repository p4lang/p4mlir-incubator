// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

match_kind {
    exact,
    lpm,
    ternary,
    range
}

header hdr {
    bit<8>  e;
    bit<16> t;
    bit<8>  l;
    bit<8> r;
    bit<8>  v;
}

struct Header_t {
    hdr h;
}
struct Meta_t {
    bit<9> egress_spec;
}

// CHECK-DAG: #[[everything:.*]] = #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
// CHECK-DAG: #[[int_16_b8i:.*]] = #p4hir.int<240> : !b8i
// CHECK-DAG: #[[int1_b8i:.*]] = #p4hir.int<1> : !b8i
// CHECK-DAG: #[[int17_b8i:.*]] = #p4hir.int<17> : !b8i
// CHECK-DAG: #[[int18_b8i:.*]] = #p4hir.int<18> : !b8i
// CHECK-DAG: #[[int2_b8i:.*]] = #p4hir.int<2> : !b8i
// CHECK-DAG: #[[int8_b8i:.*]] = #p4hir.int<8> : !b8i
// CHECK: #[[set_mask_of_int17_b8i_int_16_b8i:.*]] = #p4hir.set<mask : [#[[int17_b8i]], #[[int_16_b8i]]]> : !p4hir.set<!b8i>
// CHECK: #[[set_range_of_int1_b8i_int8_b8i:.*]] = #p4hir.set<range : [#[[int1_b8i]], #[[int8_b8i]]]> : !p4hir.set<!b8i>
// CHECK: #[[set_product_of_set_mask_of_int17_b8i_int_16_b8i:.*]] = #p4hir.set<product : [#[[set_mask_of_int17_b8i_int_16_b8i]]]> : !p4hir.set<tuple<!b8i>>
// CHECK: #[[set_product_of_set_range_of_int1_b8i_int8_b8i:.*]] = #p4hir.set<product : [#[[set_range_of_int1_b8i_int8_b8i]]]> : !p4hir.set<tuple<!b8i>>

// CHECK-LABEL: p4hir.control @ingress
control ingress(inout Header_t h, inout Meta_t m) {
    action a() { m.egress_spec = 0; }
    action a_with_control_params(bit<9> x) { m.egress_spec = x; }

    // CHECK-LABEL: p4hir.table @t_exact
    table t_exact {
  	key = {
            h.h.e : exact;
        }

	actions = {
            a;
            a_with_control_params;
        }

	default_action = a;

        // CHECK-LABEL: p4hir.table_entries const
        const entries = {
            // CHECK: p4hir.table_entry #p4hir.aggregate<[#[[int1_b8i]]]>
            // CHECK:  p4hir.call @ingress::@a_with_control_params
            0x01 : a_with_control_params(1);
            // CHECK: p4hir.table_entry #p4hir.aggregate<[#[[int2_b8i]]]
            // CHECK: p4hir.call @ingress::@a_with_control_params
            0x02 : a_with_control_params(2);
        }
    }

    // CHECK-LABEL: p4hir.table @t_lpm
    table t_lpm {
        key = {
            h.h.l : lpm;
        }

        actions = {
            a;
            a_with_control_params;
        }

        default_action = a;

        // CHECK-LABEL: p4hir.table_entries const
        const entries = {
        // CHECK: p4hir.table_entry #[[set_product_of_set_mask_of_int17_b8i_int_16_b8i]]
            0x11 &&& 0xF0 : a_with_control_params(11);
        // CHECK: p4hir.table_entry #p4hir.aggregate<[#[[int18_b8i]]]>            
            0x12          : a_with_control_params(12);
        // CHECK: p4hir.table_entry #p4hir.aggregate<[#[[everything]]]>
            _             : a_with_control_params(13);
        }
    }

    table t_ternary {
        key = {
            h.h.t : ternary;
        }

        actions = {
            a;
            a_with_control_params;
        }

        default_action = a;

        const entries = {
            0x1111 &&& 0xF    : a_with_control_params(1) @priority(3);
            0x1181            : a_with_control_params(2);
            0x1181 &&& 0xF00F : a_with_control_params(3) @priority(1);
        }
    }

    // CHECK-LABEL: p4hir.table @t_range
    table t_range {
        key = {
            h.h.r : range;
        }

        actions = {
            a;
            a_with_control_params;
        }

        default_action = a;

        // CHECK-LABEL: p4hir.table_entries const
        const entries = {
            // CHECK: p4hir.table_entry #[[set_product_of_set_range_of_int1_b8i_int8_b8i]]
            1..8 : a_with_control_params(21);
            6..12: a_with_control_params(22);
            15   : a_with_control_params(24);
            _    : a_with_control_params(23);
        }
    }
    apply {
        t_exact.apply();
        t_lpm.apply();
        t_ternary.apply();
        t_range.apply();
    }
}
