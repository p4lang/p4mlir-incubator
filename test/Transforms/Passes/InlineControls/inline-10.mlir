// RUN: p4mlir-opt  --p4hir-inline-controls %s | FileCheck %s

// Test with control instantiation without applies.
!Random = !p4hir.extern<"Random">
!anon = !p4hir.enum<a, NoAction>
!b10i = !p4hir.bit<10>
!b32i = !p4hir.bit<32>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!string = !p4hir.string
!type_T = !p4hir.type_var<"T">
#exact = #p4hir.match_kind<"exact">
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
!b = !p4hir.struct<"b", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#int3_b10i = #p4hir.int<3> : !b10i
module {
  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @extract_1<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }
  p4hir.extern @packet_out {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  p4hir.func @verify(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, !error {p4hir.dir = #in, p4hir.param_name = "toSignal"})
  p4hir.func action @NoAction() annotations {noWarn = "unused"} {
    p4hir.return
  }
  p4hir.overload_set @static_assert {
    p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool
    p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool
  }
  p4hir.extern @Random {
    p4hir.func @Random(!b10i {p4hir.dir = #undir, p4hir.param_name = "v"})
    p4hir.func @read() -> !b10i
  }
  p4hir.control @callee(%arg0: !Random {p4hir.dir = #undir, p4hir.param_name = "rand"})() {
    %c3_b10i = p4hir.const #int3_b10i
    p4hir.control_local @__local_callee_rand_0 = %arg0 : !Random
    p4hir.instantiate @Random (%c3_b10i : !b10i) as @rand2
    p4hir.func action @a() {
      %__local_callee_rand_0 = p4hir.symbol_ref @callee::@__local_callee_rand_0 : !Random
      %0 = p4hir.call_method @Random::@read of %__local_callee_rand_0 : !Random () : () -> !b10i
      p4hir.return
    }
    p4hir.table @b {
      p4hir.table_key(%arg1: !Random) {
        %0 = p4hir.call_method @Random::@read of %arg1 : !Random () : () -> !b10i
        p4hir.match_key #exact %0 : !b10i annotations {name = "rand"}
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @callee::@a () : () -> ()
        }
        p4hir.table_action @NoAction() annotations {defaultonly} {
          p4hir.call @NoAction () : () -> ()
        }
      }
      p4hir.table_default_action {
        p4hir.call @NoAction () : () -> ()
      }
    }
    p4hir.control_apply {
      %b_apply_result = p4hir.table_apply @callee::@b with key(%arg0) : (!Random) -> !b
      %0 = p4hir.call_method @Random::@read of %arg0 : !Random () : () -> !b10i
    }
  }

  // CHECK-LABEL: p4hir.control @caller
  p4hir.control @caller(%arg0: !Random {p4hir.dir = #undir, p4hir.param_name = "rand"})() {
    // CHECK: p4hir.control_local @__local_caller_rand_0 = %arg0 : !Random
    // CHECK: %[[CONST_3:.*]] = p4hir.const #int3_b10i
    // CHECK: p4hir.instantiate @Random (%[[CONST_3]] : !b10i) as @inst.rand2
    // CHECK-NOT: p4hir.func
    // CHECK-NOT: p4hir.table

    p4hir.control_local @__local_caller_rand_0 = %arg0 : !Random
    p4hir.instantiate @callee () as @inst
    // CHECK:      p4hir.control_apply {
    // CHECK-NEXT: }
    p4hir.control_apply {
    }
  }
}
