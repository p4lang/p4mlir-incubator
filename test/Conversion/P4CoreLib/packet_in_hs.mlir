// RUN: p4mlir-opt %s --lower-to-p4corelib | FileCheck %s
!b32i = !p4hir.bit<32>
!b8i = !p4hir.bit<8>
!error = !p4hir.error<NoError, StackOutOfBounds>
!packet_in = !p4hir.extern<"packet_in" annotations {corelib}>
!validity_bit = !p4hir.validity.bit
!H = !p4hir.header<"H", data: !b8i, __valid: !validity_bit>
!arr_2xH = !p4hir.array<2x!H>
!hs_2xH = !p4hir.header_stack<2x!H>
!headers_t = !p4hir.struct<"headers_t", stack: !hs_2xH>

module {
  p4hir.extern @packet_in annotations {corelib} {
    p4hir.func @extract<!p4hir.type_var<"T">>(!p4hir.ref<!p4hir.type_var<"T">>)
  }

  // CHECK-LABEL: p4hir.parser @p
  p4hir.parser @p(%arg0: !packet_in, %arg1: !p4hir.ref<!headers_t>)() {
    p4hir.state @start {
      p4hir.scope {
        %stack_ref = p4hir.struct_field_ref %arg1["stack"] : <!headers_t>
        %data_ref = p4hir.struct_field_ref %stack_ref["data"] : <!hs_2xH>
        %nextIndex_ref = p4hir.struct_field_ref %stack_ref["nextIndex"] : <!hs_2xH>
        %nextIndex = p4hir.read %nextIndex_ref : <!b32i>
        %elt_ref = p4hir.array_element_ref %data_ref[%nextIndex] : !p4hir.ref<!arr_2xH>, !b32i
        %hdr = p4hir.variable ["hdr"] : <!H>

        // CHECK: p4hir.struct_field_ref {{.*}}["nextIndex"] : <!hs_2xH>
        // CHECK: p4hir.read {{.*}} : <!b32i>
        // CHECK: %[[size:.*]] = p4hir.const #int2_b32i
        // CHECK: %[[check:.*]] = p4hir.cmp(lt, {{.*}} : !b32i, %[[size]] : !b32i)
        // CHECK: %[[error:.*]] = p4hir.const #error_StackOutOfBounds
        // CHECK: p4corelib.verify %[[check]] signalling %[[error]] : !error
        // CHECK: p4corelib.extract_header {{.*}} : <!H> from %arg0 : !p4corelib.packet_in
        p4hir.call_method @packet_in::@extract<[!H]> (%hdr) of %arg0 : !packet_in : (!p4hir.ref<!H>) -> ()

        %val = p4hir.read %hdr : <!H>
        p4hir.assign %val, %elt_ref : <!H>

        // CHECK: %[[one:.*]] = p4hir.const #int1_b32i
        // CHECK: p4hir.struct_field_ref {{.*}}["nextIndex"] : <!hs_2xH>
        // CHECK: %[[curr:.*]] = p4hir.read {{.*}} : <!b32i>
        // CHECK: %[[inc:.*]] = p4hir.binop(add, %[[curr]], %[[one]]) : !b32i
        // CHECK: p4hir.assign %[[inc]], {{.*}} : <!b32i>
      }
      p4hir.transition to @p::@accept
    }
    p4hir.state @accept {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }
}
