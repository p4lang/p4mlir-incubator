// RUN: p4mlir-opt --p4hir-symbol-dce %s | FileCheck %s

!b10i = !p4hir.bit<10>
!b12i = !p4hir.bit<12>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!i32i = !p4hir.int<32>
!infint = !p4hir.infint
!type_H = !p4hir.type_var<"H">
!type_I = !p4hir.type_var<"I">
!type_T = !p4hir.type_var<"T">
!type_U = !p4hir.type_var<"U">
!type_V = !p4hir.type_var<"V">
!void = !p4hir.void
#in = #p4hir<dir in>
#undir = #p4hir<dir undir>
!MyCounter_b10i = !p4hir.extern<"MyCounter"<!b10i>>
#int0_b16i = #p4hir.int<0> : !b16i
#int0_b8i = #p4hir.int<0> : !b8i
#int0_i32i = #p4hir.int<0> : !i32i
#int1024_infint = #p4hir.int<1024> : !infint
#int1_b12i = #p4hir.int<1> : !b12i
#int42_b10i = #p4hir.int<42> : !b10i
#int42_infint = #p4hir.int<42> : !infint
// CHECK: module
module {
  p4hir.extern @Crc16<[!type_T]> {
    p4hir.func @hash<!type_U>(!type_U {p4hir.dir = #in, p4hir.param_name = "input_data"})
    p4hir.func @id<!type_U>(!type_U {p4hir.dir = #in, p4hir.param_name = "x"}) -> !type_U
  }
  p4hir.extern @ext<[!type_H]> {
    p4hir.func @ext(!type_H {p4hir.dir = #undir, p4hir.param_name = "v"})
    p4hir.func @method<!type_T>(!type_H {p4hir.dir = #undir, p4hir.param_name = "h"}, !type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
  }
  p4hir.extern @ext2<[!type_H, !type_V]> {
    p4hir.func @ext2(!type_H {p4hir.dir = #undir, p4hir.param_name = "v"})
    p4hir.overload_set @method {
      p4hir.func @method_0<!type_T>(!type_H {p4hir.dir = #in, p4hir.param_name = "h"}, !type_T {p4hir.dir = #in, p4hir.param_name = "t"}) -> !type_V
      p4hir.func @method_1<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "t"}) -> !type_H
    }
  }
  // CHECK-NOT: unusedfunc
  p4hir.func @unusedfunc(!b32i {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !b32i
  // CHECK: usedfunc
  p4hir.func @usedfunc(!b32i {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !b32i

  // CHECK: p4hir.extern @X
  // CHECK-NOT: unusedmethod
  p4hir.extern @X<[!type_T]> {
    p4hir.func @X(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
    p4hir.func @method(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !type_T
    p4hir.func @unusedmethod(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !type_T
  }
  // CHECK-NOT: UnusedX
  p4hir.extern @UnusedX<[!type_T]> {
    p4hir.func @UnusedX(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
    p4hir.func @method(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !type_T
  }
  p4hir.extern @Y {
    p4hir.func @Y()
    p4hir.func @method<!type_T>(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
  }
  p4hir.extern @UnusedY {
    p4hir.func @UnusedY()
    p4hir.func @method<!type_T>(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
  }
  p4hir.extern @MyCounter<[!type_I]> {
    p4hir.func @MyCounter(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"})
    p4hir.func @count(!type_I {p4hir.dir = #in, p4hir.param_name = "index"})
  }
  p4hir.parser @p()() {
    %c0_i32i = p4hir.const #int0_i32i
    p4hir.instantiate @X::@X<[!i32i]> (%c0_i32i : !i32i) as @x
    p4hir.instantiate @Y () as @y
    %c0_b16i = p4hir.const #int0_b16i
    p4hir.instantiate @ext::@ext<[!b16i]> (%c0_b16i : !b16i) as @ex
    %c0_b16i_0 = p4hir.const #int0_b16i
    p4hir.instantiate @ext2::@ext2<[!b16i, !void]> (%c0_b16i_0 : !b16i) as @ey
    p4hir.state @start {
      %c0_i32i_1 = p4hir.const #int0_i32i
      %0 = p4hir.call_method @X::@method (%c0_i32i_1) of @p::@x : (!i32i) -> !i32i
      %c0_b8i = p4hir.const #int0_b8i
      p4hir.call_method @Y::@method<[!b8i]>(%c0_b8i) of @p::@y : (!b8i) -> ()
      %c0_b16i_2 = p4hir.const #int0_b16i
      %c0_b8i_3 = p4hir.const #int0_b8i
      p4hir.call_method @ext::@method<[!b8i]>(%c0_b16i_2, %c0_b8i_3) of @p::@ex : (!b16i, !b8i) -> ()
      %c1_b12i = p4hir.const #int1_b12i
      %1 = p4hir.call_method @ext2::@method<[!b12i]>(%c1_b12i) of @p::@ey : (!b12i) -> !b16i
      %c0_b8i_4 = p4hir.const #int0_b8i
      p4hir.call_method @ext2::@method<[!b8i]>(%1, %c0_b8i_4) of @p::@ey : (!b16i, !b8i) -> ()
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    // TODO: Ensure we can remove unused / unreachable states
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.state @unused {
      p4hir.parser_reject
    }    
    p4hir.transition to @p::@start
  }
  p4hir.parser @Inner(%arg0: !MyCounter_b10i {p4hir.dir = #undir, p4hir.param_name = "counter_set"})() {
    p4hir.state @start {
      %c42_b10i = p4hir.const #int42_b10i
      p4hir.call_method @MyCounter::@count (%c42_b10i) of %arg0 : !MyCounter_b10i : (!b10i) -> ()
      p4hir.transition to @Inner::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @Inner::@start
  }
  p4hir.parser @Test()() {
    %c1024 = p4hir.const #int1024_infint
    %cast = p4hir.cast(%c1024 : !infint) : !b32i
    p4hir.instantiate @MyCounter::@MyCounter<[!b10i]> (%cast : !b32i) as @counter_set
    p4hir.instantiate @Inner () as @inner
    p4hir.state @start {
      %counter_set = p4hir.symbol_ref @Test::@counter_set : !MyCounter_b10i
      p4hir.apply @Test::@inner(%counter_set) : (!MyCounter_b10i) -> ()
      p4hir.transition to @Test::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @Test::@start
  }
  p4hir.control @Inner2(%arg0: !MyCounter_b10i {p4hir.dir = #undir, p4hir.param_name = "counter_set"}, %arg1 : !b32i)() {
    // CHECK: __local_counter_set_0
    // TODO: Ensure we can transitively remove unused locals (@unused is unused action below)
    // CHECK: __local_foo
    // CHECK: __local_foo2
    p4hir.control_local @__local_counter_set_0 = %arg0 : !MyCounter_b10i
    p4hir.control_local @__local_foo = %arg1 : !b32i
    p4hir.control_local @__local_foo2 = %arg1 : !b32i
    // CHECK-NOT: __local_unused
    p4hir.control_local @__local_unused = %arg1 : !b32i

   // TODO: Ensure we can remove unused action
    p4hir.func action @unused(%arg2: !b32i {p4hir.dir = #undir}) {
      %ref = p4hir.symbol_ref @Inner2::@__local_foo : !b32i
      %val = p4hir.call @usedfunc(%ref) : (!b32i) -> !b32i
      %counter_set = p4hir.symbol_ref @Inner2::@__local_counter_set_0 : !MyCounter_b10i
      %cast = p4hir.cast(%val : !b32i) : !b10i
      p4hir.call_method @MyCounter::@count (%cast) of %counter_set : !MyCounter_b10i : (!b10i) -> ()

      p4hir.return
    }

    // CHECK: p4hir.func action @used
    p4hir.func action @used(%arg3: !b32i {p4hir.dir = #undir}) {
      %ref = p4hir.symbol_ref @Inner2::@__local_foo2 : !b32i
      %counter_set = p4hir.symbol_ref @Inner2::@__local_counter_set_0 : !MyCounter_b10i
      %cast = p4hir.cast(%ref : !b32i) : !b10i
      p4hir.call_method @MyCounter::@count (%cast) of %counter_set : !MyCounter_b10i : (!b10i) -> ()

      p4hir.return
    }    

    p4hir.table @t1 {
      p4hir.table_actions {
        p4hir.table_action @used(%arg4: !b32i) {
          p4hir.call @Inner2::@used (%arg4) : (!b32i) -> ()
        }
      }  
    }
    
    p4hir.control_apply {
      %c42_b10i = p4hir.const #int42_b10i
      p4hir.call_method @MyCounter::@count (%c42_b10i) of %arg0 : !MyCounter_b10i : (!b10i) -> ()
    }
  }
  p4hir.control @Test2()() {
    %c42 = p4hir.const #int42_infint
    %cast = p4hir.cast(%c42 : !infint) : !b32i
    p4hir.instantiate @MyCounter::@MyCounter<[!b10i]> (%cast : !b32i) as @counter_set
    p4hir.instantiate @Inner () as @inner
    // In general cannot remove unused instantiations as these might have side effects
    // CHECK: p4hir.instantiate @UnusedY
    p4hir.instantiate @UnusedY () as @y
    p4hir.control_apply {
      %counter_set = p4hir.symbol_ref @Test2::@counter_set : !MyCounter_b10i
      p4hir.apply @Test2::@inner(%counter_set) : (!MyCounter_b10i) -> ()
    }
  }
}
