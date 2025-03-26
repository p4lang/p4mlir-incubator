// RUN: p4mlir-opt %s | FileCheck %s

!validity_bit = !p4hir.validity.bit
!b32i = !p4hir.bit<32>
!H = !p4hir.header<"H", x: !b32i, y: !b32i, __valid: !validity_bit>

#invalid = #p4hir<validity.bit invalid> : !validity_bit
#valid = #p4hir<validity.bit valid> : !validity_bit
#int10_b32i = #p4hir.int<10> : !b32i
#int12_b32i = #p4hir.int<12> : !b32i
#int42_b32i = #p4hir.int<42> : !b32i
#int36_b32i = #p4hir.int<36> : !b32i

// CHECK: module
module {
  p4hir.func action @test2() {
    %h = p4hir.variable ["h"] : <!H>
    %c10_b32i = p4hir.const #int10_b32i
    %c12_b32i = p4hir.const #int12_b32i
    %valid = p4hir.const #valid
    %hdr_H = p4hir.struct (%c10_b32i, %c12_b32i, %valid) : !H
    p4hir.assign %hdr_H, %h : <!H>
    
    %invalid = p4hir.const #invalid
    %__valid_field_ref = p4hir.struct_extract_ref %h["__valid"] : <!H>
    p4hir.assign %invalid, %__valid_field_ref : <!validity_bit>

    %val = p4hir.read %h : <!H>
    %__valid = p4hir.struct_extract %val["__valid"] : !H
    %eq = p4hir.cmp(eq, %__valid, %invalid) : !validity_bit, !p4hir.bool
    p4hir.if %eq {
      %x_field_ref = p4hir.struct_extract_ref %h["x"] : <!H>
      %c42_b32i = p4hir.const #int42_b32i
      %cast = p4hir.cast(%c42_b32i : !b32i) : !b32i
      p4hir.assign %cast, %x_field_ref : <!b32i>
    } else {
      %y_field_ref = p4hir.struct_extract_ref %h["y"] : <!H>
      %c36_b32i = p4hir.const #int36_b32i
      %cast = p4hir.cast(%c36_b32i : !b32i) : !b32i
      p4hir.assign %cast, %y_field_ref : <!b32i>
    }
    p4hir.return    
  }
}
