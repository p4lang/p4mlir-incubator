// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s -p4hir-parser-unroll -verify-diagnostics | FileCheck %s


!validity_bit = !p4hir.validity.bit
!hdr = !p4hir.header<"hdr", __valid: !validity_bit>
!hs4 = !p4hir.header_stack<4x!hdr>

// CHECK-LABEL: @test_nextindex_inspection_only
module @test_nextindex_inspection_only {
    p4hir.parser @InspectOnlyParser()() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @inspect
        }
        // CHECK: p4hir.state @inspect
        // CHECK-NOT: @inspect_1
        // expected-warning@below {{parser loop at state 'inspect' has no header stack operations; cannot infer unroll depth}}
        p4hir.state @inspect {
            %stack = p4hir.variable ["stack"] : <!hs4>
            %val = p4hir.read %stack : <!hs4>
            %nextIdx = p4hir.struct_extract %val["nextIndex"] : !hs4
            p4hir.transition to @inspect
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
