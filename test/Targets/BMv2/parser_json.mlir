// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s | FileCheck %s

module {
  // CHECK: "parsers"
  // CHECK: "name": "test_parser"
  // CHECK: "id": 0
  // CHECK: "init_state": "test_parser::@start"
  // CHECK: "parse_states"
  
  bmv2ir.parser @test_parser init_state @start {
    // CHECK: "name": "start"
    // CHECK: "id": 0
    // CHECK: "parser_ops"
    // CHECK: "op": "extract"
    // CHECK: "type": "regular"
    // CHECK: "value": "hdr"
    bmv2ir.state @start parser_ops {
      bmv2ir.extract regular "hdr"
    } transitions {
      // CHECK: "type": "default"
      // CHECK: "next_state": "accept"
      bmv2ir.transition type default next_state @accept
    } transition_keys {
    }
    
    // CHECK: "name": "accept"
    // CHECK: "id": 1
    bmv2ir.state @accept parser_ops {
    } transitions {
      // CHECK: "type": "default"
      // CHECK: "next_state": null
      bmv2ir.transition type default
    } transition_keys {
    }
  }
}
