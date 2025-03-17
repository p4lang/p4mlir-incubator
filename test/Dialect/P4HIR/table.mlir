// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!ActionProfile = !p4hir.extern<"ActionProfile">
!action_enum = !p4hir.enum<"action_enum", a, b>
!action_enum1 = !p4hir.enum<"action_enum", a, b, aa>
!b32i = !p4hir.bit<32>
!i8i = !p4hir.int<8>
!infint = !p4hir.infint
#exact = #p4hir.match_kind<"exact">
#false = #p4hir.bool<false> : !p4hir.bool
#lpm = #p4hir.match_kind<"lpm">
#undir = #p4hir<dir undir>
!t1_ = !p4hir.struct<"t1", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !action_enum>
!t2_ = !p4hir.struct<"t2", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !action_enum1>
#int10_infint = #p4hir.int<10> : !infint
#int32_b32i = #p4hir.int<32> : !b32i
#int42_infint = #p4hir.int<42> : !infint
// CHECK: module
module {
  p4hir.extern @ActionProfile {
    p4hir.func @ActionProfile(!b32i {p4hir.dir = #undir})
  }
  p4hir.control @c(%arg0: !b32i)() {
    p4hir.func action @a(%arg1: !b32i {p4hir.dir = #undir}) {
      p4hir.implicit_return
    }
    p4hir.func action @aa(%arg1: !b32i {p4hir.dir = #p4hir<dir in>}, %arg2: !i8i {p4hir.dir = #undir}) {
      p4hir.implicit_return
    }
    p4hir.func action @b() {
      p4hir.implicit_return
    }
    p4hir.table @t1 {
      p4hir.table_actions {
        p4hir.table_action @a(%arg1: !b32i) {
          p4hir.call @a (%arg1) : (!b32i) -> ()
        }
        p4hir.table_action @b() {
          p4hir.call @b () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @b () : () -> ()
      }
    }
    p4hir.table @t2 {
      p4hir.table_key {
        p4hir.match_key #exact %arg0 : !b32i
        %false = p4hir.const #false
        p4hir.match_key #lpm %false : !p4hir.bool
      }
      p4hir.table_actions {
        p4hir.table_action @a(%arg1: !b32i) {
          p4hir.call @a (%arg1) : (!b32i) -> ()
        }
        p4hir.table_action @b() {
          p4hir.call @b () : () -> ()
        }
        p4hir.table_action @aa(%arg1: !i8i) {
          p4hir.call @aa (%arg0, %arg1) : (!b32i, !i8i) -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @b () : () -> ()
      }
      %size = p4hir.table_size #int42_infint
      %largest_priority_wins = p4hir.table_entry "largest_priority_wins" {
        %false = p4hir.const #false
        p4hir.yield %false : !p4hir.bool
      } : !p4hir.bool
      %priority_delta = p4hir.table_entry "priority_delta" {
        %c10 = p4hir.const #int10_infint
        p4hir.yield %c10 : !infint
      } : !infint
      %some_entry = p4hir.table_entry "some_entry" {
        %c10 = p4hir.const #int10_infint
        p4hir.yield %c10 : !infint
      } : !infint
      %implementation = p4hir.table_entry "implementation" {
        %c32_b32i = p4hir.const #int32_b32i
        %ActionProfile = p4hir.instantiate @ActionProfile(%c32_b32i) as "ActionProfile" : (!b32i) -> !ActionProfile
        p4hir.yield %ActionProfile : !ActionProfile
      } : !ActionProfile
    }
    p4hir.control_apply {
      %t1_apply_result = p4hir.table_apply @t1 : !t1_
      %hit = p4hir.struct_extract %t1_apply_result["hit"] : !t1_
      p4hir.if %hit {
        %t2_apply_result = p4hir.table_apply @t2 : !t2_
      }
    }
  }
}
