// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s | FileCheck %s

module {
  // CHECK: "parsers":[{
  // CHECK: "id":0
  // CHECK: "init_state":"start"
  // CHECK: "name":"test_parser"
  // CHECK: "parse_states":[

  bmv2ir.parser @test_parser init_state @start {
    // CHECK: "id":0
    // CHECK: "name":"start"
    // CHECK: "parser_ops":[
    // CHECK: "op":"extract"
    // CHECK: "type":"regular"
    // CHECK: "value":"hdr"
    bmv2ir.state @start
    transition_key {
    ^bb0:
    }
    transitions {
      // CHECK: "next_state":"accept"
      // CHECK: "type":"default"
      bmv2ir.transition type default next_state @accept
    }
    parser_ops {
      bmv2ir.extract regular "hdr"
    }

    // CHECK: "id":1
    // CHECK: "name":"accept"
    bmv2ir.state @accept
    transition_key {
    ^bb0:
    }
    transitions {
      // CHECK: "next_state":null
      // CHECK: "type":"default"
      bmv2ir.transition type default
    }
    parser_ops {
    ^bb0:
    }
  }
}
