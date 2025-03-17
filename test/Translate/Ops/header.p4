// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: !E = !p4hir.header<"E", __valid: !validity_bit>
// CHECK: !H = !p4hir.header<"H", x: !b32i, y: !b32i, __valid: !validity_bit>
// CHECK: !Ethernet = !p4hir.header<"Ethernet", src: !b48i, dest: !b48i, type: !EthTypes, __valid: !validity_bit>
// CHECK: !Headers = !p4hir.struct<"Headers", eth: !Ethernet>

header E {}

enum bit<16> EthTypes {
    IPv4 = 0x0800,
    ARP = 0x0806,
    RARP = 0x8035,
    EtherTalk = 0x809B,
    VLAN = 0x8100,
    IPX = 0x8137,
    IPv6 = 0x86DD
}

header Ethernet {
    bit<48> src;
    bit<48> dest;
    EthTypes type;
}

struct Headers {
    Ethernet eth;
}

header H { bit<32> x; bit<32> y; }

// CHECK-LABEL: p4hir.func action @test2
// CHECK:  %[[H:.*]] = p4hir.variable ["h"] : <!H>
// CHECK:  %[[c10_b32i:.*]] = p4hir.const #int10_b32i
// CHECK:  %[[c12_b32i:.*]] = p4hir.const #int12_b32i
// CHECK:  %[[VALID:.*]] = p4hir.const #valid
// CHECK:  %[[hdr_H:.*]] = p4hir.struct (%[[c10_b32i]], %[[c12_b32i]], %[[VALID]]) : !H
// CHECK:  p4hir.assign %[[hdr_H]], %[[H]] : <!H>
action test2() {
  H h;
  h = { 10, 12 };  // This also makes the header h valid
  h = { y = 12, x = 10 };  // Same effect as above

  h = (H){#};   // This make the header h become invalid
  if (h == (H){#}) {     // This is equivalent to the condition !h.isValid()
    h.x = 42;
  } else {
    h.y = 36;
  }
}

// CHECK-LABEL: p4hir.func action @test3
action test3() {
  H h1 = ...;
  // CHECK:  %[[h1:.*]] = p4hir.variable ["h1", init] : <!H>
  // CHECK:  %[[invalid:.*]] = p4hir.const #invalid
  // CHECK:  %[[__valid_field_ref:.*]] = p4hir.struct_extract_ref %[[h1]]["__valid"] : <!H>
  // CHECK:  p4hir.assign %[[invalid]], %[[__valid_field_ref]] : <!validity_bit>  
  H h2 = { y=5, ... };   // initialize h2 with a header that is valid, field x 0,
                         // field y 5
  H h3 = { ... };        // initialize h3 with a header that is valid, field x 0, field y 0
}

// CHECK-LABEL: p4hir.func action @test1
action test1(inout Headers h) {
    // CHECK: %[[eth_field_ref:.*]] = p4hir.struct_extract_ref %arg0["eth"] : <!Headers>
    // CHECK: %[[__valid_field_ref:.*]] = p4hir.struct_extract_ref %[[eth_field_ref]]["__valid"] : <!Ethernet>
    // CHECK: %[[val:.*]] = p4hir.read %[[__valid_field_ref]] : <!validity_bit>
    // CHECK: %[[valid:.*]] = p4hir.const #valid
    // CHECK: %[[eq:.*]] = p4hir.cmp(eq, %[[val]], %[[valid]]) : !validity_bit, !p4hir.bool
    // CHECK: %[[not:.*]] = p4hir.unary(not, %[[eq]]) : !p4hir.bool
    // CHECK: p4hir.if %[[not]] {
    // CHECK:   p4hir.implicit_return
    // CHECK: }
    if (!h.eth.isValid())
         return;
    if (h.eth.type == EthTypes.IPv4)
         h.eth.setInvalid();
    else {
         h.eth.type = (EthTypes)(bit<16>)0;
         h.eth.setValid();
    }
}

action test4(inout H h, inout E e) {
  h = (H){#};
  e = (E){#};
  bit<32> x = h.x;
  if (e.isValid()) {
    h.setValid();
    h.x = x;
  }
}

// CHECK-LABEL:   p4hir.func action @assign_header
action assign_header() {
  // CHECK:           %[[e1:.*]] = p4hir.variable ["e1"] : <!Ethernet>
  // CHECK:           %[[e2:.*]] = p4hir.variable ["e2"] : <!Ethernet>
  // CHECK:           %[[val_e2:.*]] = p4hir.read %[[e2]] : <!Ethernet>
  // CHECK:           p4hir.assign %[[val_e2]], %[[e1]] : <!Ethernet>
  // CHECK:           p4hir.implicit_return

  Ethernet e1;
  Ethernet e2;

  e1 = e2;
}

// CHECK-LABEL:   p4hir.func action @assign_invalid_header
action assign_invalid_header() {
  // CHECK:           %[[e:.*]] = p4hir.variable ["e"] : <!Ethernet>
  // CHECK:           %[[invalid:.*]] = p4hir.const #invalid
  // CHECK:           %[[__valid_field_ref:.*]] = p4hir.struct_extract_ref %[[e]]["__valid"] : <!Ethernet>
  // CHECK:           p4hir.assign %[[invalid]], %[[__valid_field_ref]] : <!validity_bit>
  // CHECK:           p4hir.implicit_return

  Ethernet e;

  e = (Ethernet){#};
}
