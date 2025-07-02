// RUN: p4mlir-opt --p4hir-enum-elimination %s | FileCheck %s

#in = #p4hir<dir in>

// CHECK: ![[suits:.*]] = !p4hir.ser_enum<"Suits", !b32i, Clubs : #int0_b32i, Diamonds : #int1_b32i, Spades : #int2_b32i>
!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Spades>

// CHECK-DAG: #[[clubs:.*]] = #p4hir.enum_field<Clubs, ![[suits]]> : ![[suits]]
// CHECK-DAG: #[[diamonds:.*]] = #p4hir.enum_field<Diamonds, ![[suits]]> : ![[suits]]
// CHECK-DAG: #[[spades:.*]] = #p4hir.enum_field<Spades, ![[suits]]> : ![[suits]]
#Suits_Clubs = #p4hir.enum_field<Clubs, !Suits> : !Suits
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits
#Suits_Spades = #p4hir.enum_field<Spades, !Suits> : !Suits


// CHECK-LABEL: module
module {
  // CHECK: p4hir.func @process_suit(%[[arg:.*]]: ![[suits]]) -> ![[suits]]
  // CHECK: p4hir.switch (%[[arg]] : ![[suits]])
  // CHECK: p4hir.case(anyof, [#[[clubs]], #[[spades]]])
  // CHECK: %[[c1:.*]] = p4hir.const ["suit"] #[[diamonds]] annotations {hidden}
  // CHECK: %{{.*}} = p4hir.call @process_suit (%[[c1]]) : (![[suits]]) -> ![[suits]]
  // CHECK: p4hir.case(equal, [#[[diamonds]]])
  p4hir.func @process_suit(%arg0: !Suits) -> !Suits {
    p4hir.switch (%arg0 : !Suits) {
      p4hir.case(anyof, [#Suits_Clubs, #Suits_Spades]) {
        %diamond_1 = p4hir.const ["suit"] #Suits_Diamonds annotations {hidden}
        %call_1 = p4hir.call @process_suit(%diamond_1) : (!Suits) -> (!Suits)
        p4hir.yield
      }
      p4hir.case(equal, [#Suits_Diamonds]) {
        p4hir.yield
      }
      p4hir.yield
    }
    p4hir.return %arg0 : !Suits
  }

  // CHECK: %[[var:.*]] = p4hir.variable ["suit"] : <![[suits]]>
  %var = p4hir.variable ["suit"] : <!Suits>

  // CHECK: %[[c2:.*]] = p4hir.const #[[diamonds]]
  %diamond_2 = p4hir.const #Suits_Diamonds

  // CHECK: p4hir.assign %[[c2]], %[[var]] : <![[suits]]>
  p4hir.assign %diamond_2, %var : <!Suits>

  // CHECK: %[[val:.*]] = p4hir.read %[[var]] : <![[suits]]>
  %val = p4hir.read %var : <!Suits>

  // CHECK: %{{.*}} = p4hir.call @process_suit (%[[val]]) : (![[suits]]) -> ![[suits]]
  %call_2 = p4hir.call @process_suit(%val) : (!Suits) -> (!Suits)

  // CHECK: p4hir.parser @p(%[[arg:.*]]: ![[suits]] {p4hir.dir = #p4hir<dir in>, p4hir.param_name = "test"})()
  // CHECK: %{{.*}} = p4hir.const #[[spades]]
  p4hir.parser @p(%arg0: !Suits {p4hir.dir = #in, p4hir.param_name = "test"})() {
    p4hir.state @start {
      %spades = p4hir.const #Suits_Spades
      p4hir.transition to @p::@next
    }
    p4hir.state @next {
      p4hir.parser_accept
    }
    p4hir.transition to @p::@start
  }

  // CHECK: p4hir.control @InnerPipe(%arg0: !Suits)()
  p4hir.control @InnerPipe(%arg0: !Suits)() {
    p4hir.control_apply {
    }
  }
}

