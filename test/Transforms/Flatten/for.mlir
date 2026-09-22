// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --p4hir-flatten-cfg %s | FileCheck %s

!b32i = !p4hir.bit<32>
!infint = !p4hir.infint
!arr = !p4hir.array<16 x !b32i>
!h = !p4hir.header<"h", f: !b32i, __valid: !p4hir.validity.bit>
!hs = !p4hir.header_stack<4 x !h>
#int1_b32i = #p4hir.int<1> : !b32i
#int10_infint = #p4hir.int<10> : !infint

module {
  // CHECK-LABEL: @for
  // CHECK-NOT: p4hir.for
  p4hir.func @for() {
    %i = p4hir.variable ["i", init] : <!b32i>
    %c0 = p4hir.const #p4hir.int<0> : !b32i
    p4hir.assign %c0, %i : <!b32i>
    %sum = p4hir.variable ["sum", init] : <!b32i>
    p4hir.assign %c0, %sum : <!b32i>

    // CHECK:      %[[IV:.*]] = p4hir.variable ["i", init] : <!b32i>
    // CHECK:      p4hir.assign %{{.*}}, %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND:.*]]
    // CHECK:    ^[[COND]]:
    // CHECK:      %[[C10:.*]] = p4hir.const #int10_infint
    // CHECK:      %[[CAST:.*]] = p4hir.cast(%[[C10]] : !infint) : !b32i
    // CHECK:      %[[I_COND:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[CMP:.*]] = p4hir.cmp(lt, %[[I_COND]] : !b32i, %[[CAST]] : !b32i)
    // CHECK:      p4hir.cond_br %[[CMP]] ^[[BODY:.*]], ^[[EXIT:.*]]
    // CHECK:    ^[[BODY]]:
    // CHECK:      %[[BODY_C1:.*]] = p4hir.const #int1_b32i
    // CHECK:      %[[SUM_VAL:.*]] = p4hir.read %sum : <!b32i>
    // CHECK:      %[[BODY_ADD:.*]] = p4hir.binop(add, %[[SUM_VAL]], %[[BODY_C1]]) : !b32i
    // CHECK:      p4hir.assign %[[BODY_ADD]], %sum : <!b32i>
    // CHECK:      p4hir.br ^[[UPDATES:.*]]
    // CHECK:    ^[[UPDATES]]:
    // CHECK:      %[[ONE:.*]] = p4hir.const #int1_b32i
    // CHECK:      %[[I_UPD:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[INC:.*]] = p4hir.binop(add, %[[I_UPD]], %[[ONE]]) : !b32i
    // CHECK:      p4hir.assign %[[INC]], %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND]]
    // CHECK:    ^[[EXIT]]:
    // CHECK:      p4hir.return
    p4hir.for : cond {
      %c10 = p4hir.const #int10_infint
      %cast = p4hir.cast(%c10 : !infint) : !b32i
      %val = p4hir.read %i : <!b32i>
      %cond = p4hir.cmp(lt, %val : !b32i, %cast : !b32i)
      p4hir.condition %cond
    } body {
      %c1 = p4hir.const #int1_b32i
      %val = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %val, %c1) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    } updates {
      %c1 = p4hir.const #int1_b32i
      %val = p4hir.read %i : <!b32i>
      %add = p4hir.binop(add, %val, %c1) : !b32i
      p4hir.assign %add, %i : <!b32i>
      p4hir.yield
    }
    p4hir.return
  }

  // CHECK-LABEL: @foreach_range
  // CHECK-NOT: p4hir.foreach
  // CHECK-NOT: p4hir.for
  // CHECK-NOT: p4hir.range
  p4hir.func @foreach_range() {
    %sum = p4hir.variable ["sum", init] : <!b32i>
    %c0 = p4hir.const #p4hir.int<0> : !b32i
    p4hir.assign %c0, %sum : <!b32i>
    %c9 = p4hir.const #p4hir.int<9> : !b32i
    %range = p4hir.range(%c0, %c9) : !p4hir.set<!b32i>
    // CHECK:      %[[IV:.*]] = p4hir.variable ["i", init] : <!b32i>
    // CHECK:      p4hir.assign %{{.*}}, %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND:.*]]
    // CHECK:    ^[[COND]]:
    // CHECK:      %[[I_COND:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[CMP:.*]] = p4hir.cmp(le, %[[I_COND]] : !b32i, %{{.*}} : !b32i)
    // CHECK:      p4hir.cond_br %[[CMP]] ^[[BODY:.*]], ^[[EXIT:.*]]
    // CHECK:    ^[[BODY]]:
    // CHECK:      %[[ELEM:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[SUM_VAL:.*]] = p4hir.read %sum : <!b32i>
    // CHECK:      %[[ADD:.*]] = p4hir.binop(add, %[[SUM_VAL]], %[[ELEM]]) : !b32i
    // CHECK:      p4hir.assign %[[ADD]], %sum : <!b32i>
    // CHECK:      p4hir.br ^[[UPDATES:.*]]
    // CHECK:    ^[[UPDATES]]:
    // CHECK:      %[[I_UPD:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[ONE:.*]] = p4hir.const
    // CHECK:      %[[INC:.*]] = p4hir.binop(add, %[[I_UPD]], %[[ONE]]) : !b32i
    // CHECK:      p4hir.assign %[[INC]], %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND]]
    // CHECK:    ^[[EXIT]]:
    // CHECK:      p4hir.return
    p4hir.foreach %arg : !b32i in %range : !p4hir.set<!b32i> {
      %v = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %v, %arg) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    }
    p4hir.return
  }

  // CHECK-LABEL: @foreach_array
  // CHECK-NOT: p4hir.foreach
  // CHECK-NOT: p4hir.for
  p4hir.func @foreach_array(%array : !arr) {
    %sum = p4hir.variable ["sum", init] : <!b32i>
    %c0 = p4hir.const #p4hir.int<0> : !b32i
    p4hir.assign %c0, %sum : <!b32i>

    // CHECK:      %[[IV:.*]] = p4hir.variable ["i", init] : <!b5i>
    // CHECK:      p4hir.assign %{{.*}}, %[[IV]] : <!b5i>
    // CHECK:      p4hir.br ^[[COND:.*]]
    // CHECK:    ^[[COND]]:
    // CHECK:      %[[I_COND:.*]] = p4hir.read %[[IV]] : <!b5i>
    // CHECK:      %[[CMP:.*]] = p4hir.cmp(lt, %[[I_COND]] : !b5i, %{{.*}} : !b5i)
    // CHECK:      p4hir.cond_br %[[CMP]] ^[[BODY:.*]], ^[[EXIT:.*]]
    // CHECK:    ^[[BODY]]:
    // CHECK:      %[[I_BODY:.*]] = p4hir.read %[[IV]] : <!b5i>
    // CHECK:      %[[ELEM:.*]] = p4hir.array_get %{{.*}}[%[[I_BODY]]] : !arr_16xb32i, !b5i
    // CHECK:      %[[SUM_VAL:.*]] = p4hir.read %sum : <!b32i>
    // CHECK:      %[[ADD:.*]] = p4hir.binop(add, %[[SUM_VAL]], %[[ELEM]]) : !b32i
    // CHECK:      p4hir.assign %[[ADD]], %sum : <!b32i>
    // CHECK:      p4hir.br ^[[UPDATES:.*]]
    // CHECK:    ^[[UPDATES]]:
    // CHECK:      %[[I_UPD:.*]] = p4hir.read %[[IV]] : <!b5i>
    // CHECK:      %[[ONE:.*]] = p4hir.const
    // CHECK:      %[[INC:.*]] = p4hir.binop(add, %[[I_UPD]], %[[ONE]]) : !b5i
    // CHECK:      p4hir.assign %[[INC]], %[[IV]] : <!b5i>
    // CHECK:      p4hir.br ^[[COND]]
    // CHECK:    ^[[EXIT]]:
    // CHECK:      p4hir.return
    p4hir.foreach %arg : !b32i in %array : !arr {
      %v = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %v, %arg) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    }
    p4hir.return
  }

  // CHECK-LABEL: @foreach_header_stack
  // CHECK-NOT: p4hir.foreach
  // CHECK-NOT: p4hir.for
  p4hir.func @foreach_header_stack() {
    %stack = p4hir.variable ["stack"] : <!hs>
    %sum = p4hir.variable ["sum", init] : <!b32i>
    %c0 = p4hir.const #p4hir.int<0> : !b32i
    p4hir.assign %c0, %sum : <!b32i>
    %val = p4hir.read %stack : <!hs>

    // CHECK:      %[[IV:.*]] = p4hir.variable ["i", init] : <!b3i>
    // CHECK:      p4hir.assign %{{.*}}, %[[IV]] : <!b3i>
    // CHECK:      p4hir.br ^[[COND:.*]]
    // CHECK:    ^[[COND]]:
    // CHECK:      %[[I_COND:.*]] = p4hir.read %[[IV]] : <!b3i>
    // CHECK:      %[[CMP:.*]] = p4hir.cmp(lt, %[[I_COND]] : !b3i, %{{.*}} : !b3i)
    // CHECK:      p4hir.cond_br %[[CMP]] ^[[BODY:.*]], ^[[EXIT:.*]]
    // CHECK:    ^[[BODY]]:
    // CHECK:      %[[I_BODY:.*]] = p4hir.read %[[IV]] : <!b3i>
    // CHECK:      %[[DATA:.*]] = p4hir.struct_extract %{{.*}}["data"] : !hs_4xh
    // CHECK:      %[[ELEM:.*]] = p4hir.array_get %[[DATA]][%[[I_BODY]]] : !arr_4xh, !b3i
    // CHECK:      %[[F:.*]] = p4hir.struct_extract %[[ELEM]]["f"] : !h
    // CHECK:      %[[SUM_VAL:.*]] = p4hir.read %sum : <!b32i>
    // CHECK:      %[[ADD:.*]] = p4hir.binop(add, %[[SUM_VAL]], %[[F]]) : !b32i
    // CHECK:      p4hir.assign %[[ADD]], %sum : <!b32i>
    // CHECK:      p4hir.br ^[[UPDATES:.*]]
    // CHECK:    ^[[UPDATES]]:
    // CHECK:      %[[I_UPD:.*]] = p4hir.read %[[IV]] : <!b3i>
    // CHECK:      %[[ONE:.*]] = p4hir.const
    // CHECK:      %[[INC:.*]] = p4hir.binop(add, %[[I_UPD]], %[[ONE]]) : !b3i
    // CHECK:      p4hir.assign %[[INC]], %[[IV]] : <!b3i>
    // CHECK:      p4hir.br ^[[COND]]
    // CHECK:    ^[[EXIT]]:
    // CHECK:      p4hir.return
    p4hir.foreach %arg : !h in %val : !hs {
      %f = p4hir.struct_extract %arg["f"] : !h
      %v = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %v, %f) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    }
    p4hir.return
  }

  // foreach over a constant range
  // CHECK-LABEL: @foreach_const_range
  // CHECK-NOT: p4hir.foreach
  // CHECK-NOT: p4hir.for
  // CHECK-NOT: p4hir.const #p4hir.set
  p4hir.func @foreach_const_range() {
    %sum = p4hir.variable ["sum", init] : <!b32i>
    %c0 = p4hir.const #p4hir.int<0> : !b32i
    p4hir.assign %c0, %sum : <!b32i>
    %set = p4hir.const #p4hir.set<range : [#p4hir.int<0> : !b32i, #p4hir.int<9> : !b32i]> : !p4hir.set<!b32i>

    // CHECK:      %[[IV:.*]] = p4hir.variable ["i", init] : <!b32i>
    // CHECK:      p4hir.assign %{{.*}}, %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND:.*]]
    // CHECK:    ^[[COND]]:
    // CHECK:      %[[I_COND:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[CMP:.*]] = p4hir.cmp(le, %[[I_COND]] : !b32i, %{{.*}} : !b32i)
    // CHECK:      p4hir.cond_br %[[CMP]] ^[[BODY:.*]], ^[[EXIT:.*]]
    // CHECK:    ^[[BODY]]:
    // CHECK:      %[[ELEM:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[SUM_VAL:.*]] = p4hir.read %sum : <!b32i>
    // CHECK:      %[[ADD:.*]] = p4hir.binop(add, %[[SUM_VAL]], %[[ELEM]]) : !b32i
    // CHECK:      p4hir.assign %[[ADD]], %sum : <!b32i>
    // CHECK:      p4hir.br ^[[UPDATES:.*]]
    // CHECK:    ^[[UPDATES]]:
    // CHECK:      %[[I_UPD:.*]] = p4hir.read %[[IV]] : <!b32i>
    // CHECK:      %[[ONE:.*]] = p4hir.const
    // CHECK:      %[[INC:.*]] = p4hir.binop(add, %[[I_UPD]], %[[ONE]]) : !b32i
    // CHECK:      p4hir.assign %[[INC]], %[[IV]] : <!b32i>
    // CHECK:      p4hir.br ^[[COND]]
    // CHECK:    ^[[EXIT]]:
    // CHECK:      p4hir.return
    p4hir.foreach %arg : !b32i in %set : !p4hir.set<!b32i> {
      %v = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %v, %arg) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    }
    p4hir.return
  }
}
