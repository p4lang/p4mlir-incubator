// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

struct P {
    bit<32> f1;
    bit<32> f2;
}

struct T {
    int<32> t1;
    int<32> t2;
}

struct S {
    T s1;
    T s2;
}

struct Empty {};

// CHECK: !Empty = !p4hir.struct<"Empty">
// CHECK: !PortId_t = !p4hir.struct<"PortId_t", _v: !b9i>
// CHECK: !T = !p4hir.struct<"T", t1: !i32i, t2: !i32i>
// CHECK: !S = !p4hir.struct<"S", s1: !T, s2: !T>
// CHECK: !metadata_t = !p4hir.struct<"metadata_t", foo: !PortId_t>

// CHECK-LABEL: module

const T t = { 32s10, 32s20 };
const S s = { { 32s15, 32s25}, t };

// CHECK: %t = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T
// CHECK: %s = p4hir.const ["s"] #p4hir.aggregate<[#p4hir.aggregate<[#int15_i32i, #int25_i32i]> : !T, #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !T]> : !S

const int<32> x = t.t1;
const int<32> y = s.s1.t2;

const int<32> w = .t.t1;

// CHECK: %x = p4hir.const ["x"] #int10_i32i
// CHECK: %y = p4hir.const ["y"] #int25_i32i
// CHECK: %w = p4hir.const ["w"] #int10_i32i

const T tt1 = s.s1;
const Empty e = {};

// CHECK: %tt1 = p4hir.const ["tt1"] #p4hir.aggregate<[#int15_i32i, #int25_i32i]> : !T
// CHECK: %e = p4hir.const ["e"] #p4hir.aggregate<[]> : !Empty

const T t1 = { 10, 20 };
const S s1 = { { 15, 25 }, t1 };

const int<32> x1 = t1.t1;
const int<32> y1 = s1.s1.t2;

const int<32> w1 = .t1.t1;

const T t2 = s1.s1;

struct PortId_t { bit<9> _v; }

const PortId_t PSA_CPU_PORT = { _v = 9w192 };

struct metadata_t {
    PortId_t foo;
}

action test2(inout PortId_t port) {
  port._v = port._v + 1; 
}

// CHECK-LABEL: p4hir.func action @test2(%arg0: !p4hir.ref<!PortId_t> {p4hir.dir = #p4hir<dir inout>}) {
// CHECK: %[[_V_REF:.*]] = p4hir.struct_extract_ref %arg0["_v"] : <!PortId_t>
// CHECK: %[[VAL:.*]] = p4hir.read %arg0 : <!PortId_t>
// CHECK: %[[_V_VAL:.*]]  = p4hir.struct_extract %[[VAL]]["_v"] : !PortId_t
// CHECK: p4hir.assign %{{.*}}, %[[_V_REF]]
// CHECK: p4hir.implicit_return

// CHECK-LABEL: p4hir.func action @test(%arg0: !p4hir.ref<!metadata_t> {p4hir.dir = #p4hir<dir inout>}) {
// Just few important bits here        
action test(inout metadata_t meta) {
   bit<9> vv;

   PortId_t p1 = { _v = vv };

   // CHECK: %[[VV_VAR:.*]] = p4hir.variable ["vv"] : <!b9i>
   // CHECK: %[[VV_VAL:.*]] = p4hir.read %[[VV_VAR]] : <!b9i>
   // CHECK: %[[STRUCT:.*]] = p4hir.struct (%[[VV_VAL]]) : !PortId_t
   // CHECK: %[[P_VAR:.*]] = p4hir.variable ["p1", init] : <!PortId_t>
   // CHECK: p4hir.assign %[[STRUCT]], %[[P_VAR]] : <!PortId_t>
            
   PortId_t p;
   bit<9> v;
   v = p._v;

   v = meta.foo._v;

   meta.foo._v = 1;

   // CHECK: p4hir.scope {
   // CHECK: p4hir.call @test2            
   test2(meta.foo);
   // CHECK: }
                
   // CHECK: %[[METADATA_VAL:.*]] = p4hir.read %arg0 : <!metadata_t>
   // CHECK: %[[FOO:.*]] = p4hir.struct_extract %[[METADATA_VAL]]["foo"] : !metadata_t
   // CHECK: %[[PSA_CPU_PORT:.*]] = p4hir.const ["PSA_CPU_PORT"] #p4hir.aggregate<[#int192_b9i]> : !PortId_t
   // CHECK: %eq = p4hir.cmp(eq, %[[FOO]], %[[PSA_CPU_PORT]]) : !PortId_t, !p4hir.bool
   if (meta.foo == PSA_CPU_PORT) {
       meta.foo._v = meta.foo._v + 1;
   }
}
