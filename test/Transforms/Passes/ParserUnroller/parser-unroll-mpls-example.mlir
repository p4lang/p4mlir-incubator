// RUN: p4mlir-opt %s -p4hir-parser-unroll | FileCheck %s

// start -> parse_ethernet -> parse_mpls -+-> parse_mpls (back-edge, BOS=0)
//                                        +-> parse_ipv4 (exit, BOS=1)

!b1 = !p4hir.bit<1>

#int0_b1 = #p4hir.int<0> : !b1
#int1_b1 = #p4hir.int<1> : !b1

// CHECK-LABEL: @test_mpls_loop
module @test_mpls_loop {
    // CHECK: p4hir.parser @MPLSParser
    p4hir.parser @MPLSParser(%bos: !b1)() {
        // CHECK: p4hir.state @start
        p4hir.state @start {
            p4hir.transition to @parse_ethernet
        }
        // CHECK: p4hir.state @parse_ethernet
        p4hir.state @parse_ethernet {
            p4hir.transition to @parse_mpls
        }
        // bos is a parser param, not a header stack → not unrolled: the
        // self-loop is preserved and no clone is produced.
        // CHECK: p4hir.state @parse_mpls {
        // CHECK:   } to @parse_mpls
        // CHECK-NOT: @parse_mpls_1
        p4hir.state @parse_mpls {
            p4hir.transition_select %bos : !b1 {
                p4hir.select_case {
                    %c0 = p4hir.const #int0_b1
                    %s0 = p4hir.set (%c0) : !p4hir.set<!b1>
                    p4hir.yield %s0 : !p4hir.set<!b1>
                } to @parse_mpls
                p4hir.select_case {
                    %everything = p4hir.const #p4hir.universal_set : !p4hir.set<!p4hir.dontcare>
                    p4hir.yield %everything : !p4hir.set<!p4hir.dontcare>
                } to @parse_ipv4
            }
        }
        // CHECK: p4hir.state @parse_ipv4
        p4hir.state @parse_ipv4 {
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
