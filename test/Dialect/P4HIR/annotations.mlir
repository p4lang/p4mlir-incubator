// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!Annotated = !p4hir.extern<"Annotated" annotations {size = ["100"]}>
!b10i = !p4hir.bit<10>
!b16i = !p4hir.bit<16>
!b1i = !p4hir.bit<1>
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!empty = !p4hir.struct<"empty">
!i10i = !p4hir.int<10>
!infint = !p4hir.infint
#exact = #p4hir.match_kind<"exact">
#false = #p4hir.bool<false> : !p4hir.bool
#in = #p4hir<dir in>
#lpm = #p4hir.match_kind<"lpm">
#ternary = #p4hir.match_kind<"ternary">
#undir = #p4hir<dir undir>
#int-1_b1i = #p4hir.int<1> : !b1i
#int1_i10i = #p4hir.int<1> : !i10i
#int2_i10i = #p4hir.int<2> : !i10i
#int42_b10i = #p4hir.int<42> : !b10i
#int42_infint = #p4hir.int<42> : !infint
!Foo = !p4hir.struct<"Foo", bar: !b32i {match = #ternary}>
!SomeEnum = !p4hir.enum<"SomeEnum" {p4runtime_translation = ["p4.org/psa/v1/bar", ",", "enum"]}, Field, Field2>
!b9i = !p4hir.bit<9>
!validity_bit = !p4hir.validity.bit
!PortId_32_t = !p4hir.alias<"PortId_32_t" annotations {p4runtime_translation = ["p4.org/psa/v1/PortId_32_t", ",", "32"]}, !b9i>
#int1_b8i = #p4hir.int<1> : !b8i
!PreservedFieldList = !p4hir.ser_enum<"PreservedFieldList" {p4runtime_translation = ["p4.org/psa/v1/foo", ",", "enum"]}, !b8i, Field : #int1_b8i>
!packet_in_header_t = !p4hir.header<"packet_in_header_t" {controller_header = ["packet_in"]}, ingress_port: !PortId_32_t {id = ["1"]}, target_egress_port: !PortId_32_t {id = ["2"]}, __valid: !validity_bit>
#PreservedFieldList_Field = #p4hir.enum_field<Field, !PreservedFieldList> : !PreservedFieldList
!Meta = !p4hir.struct<"Meta" {controller_header = ["foo"]}, b: !b1i {field_list = #PreservedFieldList_Field}, f: !PreservedFieldList>

// CHECK: module
module {
  %b = p4hir.const ["b"] #int-1_b1i annotations {hidden}
  p4hir.extern @Annotated annotations {size = ["100"]} {
    p4hir.func @Annotated() annotations {hidden, name = "annotated", pkginfo = {bar = "42", foo = 10 : i64}}
    p4hir.func @execute(!b8i {p4hir.annotations = {optional}, p4hir.dir = #undir, p4hir.param_name = "index"}) annotations {name = "exe"}
  }
  p4hir.extern @Virtual {
    p4hir.func @Virtual()
    p4hir.func @run(!b16i {p4hir.dir = #in, p4hir.param_name = "ix"})
    p4hir.func @f(!b16i {p4hir.dir = #in, p4hir.param_name = "ix"}) -> !b16i annotations {synchronous = @run}
  }
  p4hir.func @log(!b32i {p4hir.dir = #in, p4hir.param_name = "data"}) -> !b32i annotations {pure}
  p4hir.parser @p(%arg0: !empty {p4hir.dir = #in, p4hir.param_name = "e"}, %arg1: !i10i {p4hir.dir = #in, p4hir.param_name = "sinit"})() annotations {pkginfo = {bar = "42", foo = 10 : i64}} {
    %s = p4hir.variable ["s", init] annotations {name = "var.s"} : <!i10i>
    p4hir.assign %arg1, %s : <!i10i>
    p4hir.state @start annotations {name = "state.start"} {
      %c1_i10i = p4hir.const #int1_i10i
      %cast = p4hir.cast(%c1_i10i : !i10i) : !i10i
      p4hir.assign %cast, %s : <!i10i>
      p4hir.transition to @p::@next
    }
    p4hir.state @next {
      %c2_i10i = p4hir.const #int2_i10i
      %cast = p4hir.cast(%c2_i10i : !i10i) : !i10i
      p4hir.assign %cast, %s : <!i10i>
      p4hir.transition to @p::@accept
    }
    p4hir.state @drop {
      p4hir.transition to @p::@reject
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.state @reject {
      p4hir.parser_reject
    }
    p4hir.transition to @p::@start
  }
  p4hir.control @c(%arg0: !Foo {p4hir.dir = #in, p4hir.param_name = "ff"}, %arg1: !p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "bb"})() annotations {pkginfo = {bar = "42", foo = 10 : i64}} {
    p4hir.func action @a() annotations {hidden} {
      p4hir.implicit_return
    }
    p4hir.func action @b() {
      p4hir.implicit_return
    }
    p4hir.func action @cc() {
      p4hir.implicit_return
    }
    p4hir.table @t annotations {name = "table.t"} {
      p4hir.table_key {
        %c42_b10i = p4hir.const #int42_b10i
        p4hir.match_key #exact %c42_b10i : !b10i annotations {name = "exact.key"}
        %false = p4hir.const #false
        p4hir.match_key #lpm %false : !p4hir.bool
      }
      p4hir.table_actions {
        p4hir.table_action @a() {
          p4hir.call @a () : () -> ()
        }
        p4hir.table_action @b() annotations {tableonly} {
          p4hir.call @b () : () -> ()
        }
        p4hir.table_action @cc() annotations {defaultonly} {
          p4hir.call @cc () : () -> ()
        }
      }
      p4hir.table_default_action annotations {name = "bar"} {
        p4hir.call @cc () : () -> ()
      }
      %implementation = p4hir.table_entry "implementation" annotations {name = "foo"} {
        %Annotated = p4hir.instantiate @Annotated() as "Annotated" : () -> !Annotated
        p4hir.yield %Annotated : !Annotated
      } : !Annotated
      %size = p4hir.table_size #int42_infint annotations {name = "dummy.size"}
    }

    %cond = p4hir.const #false
    p4hir.for : cond {
      p4hir.condition %cond
    } body annotations {unroll} {
      p4hir.yield
    } updates {
      p4hir.yield
    }

    p4hir.control_apply {
      p4hir.scope annotations {unlikely} {
      }
      p4hir.if %arg1 annotations {likely} {
      }
      p4hir.if %arg1 annotations {likely} {
      } else annotations {unlikely} {
      }
    }
  }

  p4hir.func action @foo(%arg0: !Meta {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "m"}, %arg1: !packet_in_header_t {p4hir.annotations = {optional}, p4hir.dir = #p4hir<dir in>, p4hir.param_name = "hdr"}, %arg2: !SomeEnum {p4hir.dir = #p4hir<dir undir>, p4hir.param_name = "e"}) {
    p4hir.implicit_return
  }
}

