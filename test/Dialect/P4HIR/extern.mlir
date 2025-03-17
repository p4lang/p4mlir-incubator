// RUN: p4mlir-opt %s | FileCheck %s

!b10i = !p4hir.bit<10>
!b12i = !p4hir.bit<12>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!i32i = !p4hir.int<32>
!void = !p4hir.void
!i7i = !p4hir.int<7>
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
#int0_b16i = #p4hir.int<0> : !b16i
#int0_b8i = #p4hir.int<0> : !b8i
#int0_i32i = #p4hir.int<0> : !i32i
#int1024_b32i = #p4hir.int<1024> : !b32i
#int1_b12i = #p4hir.int<1> : !b12i
#int42_b10i = #p4hir.int<42> : !b10i

// CHECK: module
module {
  p4hir.extern @Y {
    p4hir.func @Y()
    p4hir.func @method<!p4hir.type_var<"T">>(!p4hir.type_var<"T"> {p4hir.dir = #undir})
  }

  p4hir.extern @ext<[!p4hir.type_var<"H">]> {
    p4hir.func @ext(!p4hir.type_var<"H"> {p4hir.dir = #undir})
    p4hir.func @method<!p4hir.type_var<"T">>(!p4hir.type_var<"H"> {p4hir.dir = #undir}, !p4hir.type_var<"T"> {p4hir.dir = #undir})
  }
  
  p4hir.extern @ext2<[!p4hir.type_var<"H">, !p4hir.type_var<"V">]> {
    p4hir.func @ext2(!p4hir.type_var<"H"> {p4hir.dir = #undir})
    p4hir.overload_set @method {
      p4hir.func @method_0<!p4hir.type_var<"T">>(!p4hir.type_var<"H"> {p4hir.dir = #in}, !p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"V">
      p4hir.func @method_1<!p4hir.type_var<"T">>(!p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"H">
    }
  }
  
  p4hir.extern @X<[!p4hir.type_var<"T">]> {
    p4hir.func @X(!p4hir.type_var<"T"> {p4hir.dir = #undir})
    p4hir.func @method(!p4hir.type_var<"T"> {p4hir.dir = #undir}) -> !p4hir.type_var<"T">
  }

  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!p4hir.type_var<"T">>(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out})
      p4hir.func @extract_1<!p4hir.type_var<"T">>(!p4hir.ref<!p4hir.type_var<"T">> {p4hir.dir = #out}, !b32i {p4hir.dir = #in})
    }
    p4hir.func @lookahead<!p4hir.type_var<"T">>() -> !p4hir.type_var<"T">
    p4hir.func @advance(!b32i {p4hir.dir = #in})
    p4hir.func @length() -> !b32i
  }

  p4hir.func @externFunc<!p4hir.type_var<"T">, !p4hir.type_var<"U">>(!p4hir.type_var<"U"> {p4hir.dir = #in}, !p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"T">

  p4hir.func action @foo(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #p4hir<dir inout>}, %arg1: !i7i {p4hir.dir = #in}) {
    %val = p4hir.read %arg0 : <!b32i>
    %call = p4hir.call @externFunc<[!b32i, !i7i]> (%arg1, %val) : (!i7i, !b32i) -> !b32i
    p4hir.assign %call, %arg0 : <!b32i>
    p4hir.implicit_return
  }

  p4hir.parser @p()() {
    %c0_i32i = p4hir.const #int0_i32i
    %x = p4hir.instantiate @X(%c0_i32i) as "x" : (!i32i) -> !p4hir.extern<"X"<!i32i>>
    %y = p4hir.instantiate @Y() as "y" : () -> !p4hir.extern<"Y">
    %c0_b16i = p4hir.const #int0_b16i
    %ex = p4hir.instantiate @ext(%c0_b16i) as "ex" : (!b16i) -> !p4hir.extern<"ext"<!b16i>>
    %c0_b16i_0 = p4hir.const #int0_b16i
    %ey = p4hir.instantiate @ext2(%c0_b16i_0) as "ey" : (!b16i) -> !p4hir.extern<"ext2"<!b16i, !void>>
    p4hir.state @start {
      %c0_i32i_1 = p4hir.const #int0_i32i
      %0 = p4hir.call_method @X::@method (%x, %c0_i32i_1) : !p4hir.extern<"X"<!i32i>>, (!i32i) -> !i32i
      %c0_b8i = p4hir.const #int0_b8i
      p4hir.call_method @Y::@method<[!b8i]> (%y, %c0_b8i) : !p4hir.extern<"Y">, (!b8i) -> ()
      %c0_b16i_2 = p4hir.const #int0_b16i
      %c0_b8i_3 = p4hir.const #int0_b8i
      p4hir.call_method @ext::@method<[!b8i]> (%ex, %c0_b16i_2, %c0_b8i_3) : !p4hir.extern<"ext"<!b16i>>, (!b16i, !b8i) -> ()
      %c1_b12i = p4hir.const #int1_b12i
      %1 = p4hir.call_method @ext2::@method<[!b12i]> (%ey, %c1_b12i) : !p4hir.extern<"ext2"<!b16i, !void>>, (!b12i) -> !b16i
      %c0_b8i_4 = p4hir.const #int0_b8i
      p4hir.call_method @ext2::@method<[!b8i]> (%ey, %1, %c0_b8i_4) : !p4hir.extern<"ext2"<!b16i, !void>>, (!b16i, !b8i) -> ()
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p::@start
  }
}
