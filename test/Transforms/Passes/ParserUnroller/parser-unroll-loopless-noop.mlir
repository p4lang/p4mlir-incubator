// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

// Loop-less parsers with nothing to unroll: the pass now runs on acyclic
// parsers, so it must leave these untouched (and not require @reject).

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
#int0_b32i = #p4hir.int<0> : !b32i

// issue4932: a trivial single-transition parser stays as-is.
// CHECK-LABEL: @issue4932_trivial
module @issue4932_trivial {
    // CHECK: p4hir.parser @ParserImpl
    p4hir.parser @ParserImpl()() {
        // CHECK: p4hir.state @start
        // CHECK:   p4hir.transition to @accept
        // CHECK-NOT: @start_1
        p4hir.state @start {
            p4hir.transition to @accept
        }
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }
}

!o1 = !p4hir.header<"O1", byte: !b8i, __valid: !validity_bit>
!arr_1xo1 = !p4hir.array<1x!o1>
!hs1 = !p4hir.header_stack<1x!o1>
!headers = !p4hir.struct<"headers", u: !hs1>

// issue561-7: a *constant* header-stack index (hdr.u[0]) is not a .next access,
// so it must be kept verbatim — no clones, no index substitution.
// CHECK-LABEL: @issue561_const_index
module @issue561_const_index {
    // CHECK: p4hir.parser @P
    p4hir.parser @P(%arg1: !p4hir.ref<!headers>)() {
        p4hir.state @start {
            p4hir.transition to @parseO1
        }
        // CHECK: p4hir.state @parseO1 {
        // CHECK:   %[[C0:.*]] = p4hir.const #int0_b32i
        // CHECK:   p4hir.array_element_ref {{.*}}[%[[C0]]]
        // CHECK:   p4hir.transition to @accept
        // CHECK-NOT: @parseO1_1
        p4hir.state @parseO1 {
            %u_ref = p4hir.struct_field_ref %arg1["u"] : <!headers>
            %data_ref = p4hir.struct_field_ref %u_ref["data"] : <!hs1>
            %c0 = p4hir.const #int0_b32i
            %elt_ref = p4hir.array_element_ref %data_ref[%c0] : !p4hir.ref<!arr_1xo1>, !b32i
            p4hir.transition to @accept
        }
        p4hir.state @accept {
            p4hir.parser_accept
        }
        p4hir.state @reject {
            p4hir.parser_reject
        }
        p4hir.transition to @start
    }
}
