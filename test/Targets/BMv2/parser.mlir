// RUN: p4mlir-to-json --p4hir-to-bmv2-json %s --split-input-file | FileCheck %s

!b8i = !p4hir.bit<8>
#int1_b8i = #p4hir.int<1> : !b8i
#int2_b8i = #p4hir.int<2> : !b8i
module {
  bmv2ir.header_instance @prs1_top : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
  bmv2ir.header_instance @prs1_one : !bmv2ir.header<"header_one", [type:!p4hir.bit<8>, data:!p4hir.bit<8>], max_length = 2>
  bmv2ir.header_instance @prs1_two : !bmv2ir.header<"header_two", [type:!p4hir.bit<8>, data:!p4hir.bit<16>], max_length = 3>
  bmv2ir.header_instance @prs_e_0 : !bmv2ir.header<"header_one", [type:!p4hir.bit<8>, data:!p4hir.bit<8>], max_length = 2>
  bmv2ir.parser @prs init_state @prs::@start {
    bmv2ir.state @start
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @prs::@parse_headers
    }
     parser_ops {
      bmv2ir.extract  regular @prs1_top
    }
    bmv2ir.state @parse_headers
     transition_key {
      bmv2ir.lookahead<0, 8>
    }
     transitions {
      bmv2ir.transition type  hexstr value #int1_b8i next_state @prs::@parse_one
      bmv2ir.transition type  hexstr value #int2_b8i next_state @prs::@parse_two
      bmv2ir.transition type  hexstr value #int1_b8i mask #int2_b8i next_state @prs::@parse_two
      bmv2ir.transition type  default next_state @prs::@parse_bottom
    }
     parser_ops {
    }
    bmv2ir.state @parse_one
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @prs::@parse_two
    }
     parser_ops {
      bmv2ir.extract  regular @prs_e_0
      bmv2ir.assign_header @prs_e_0 to @prs1_one
    }
    bmv2ir.state @parse_two
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @prs::@parse_bottom
    }
     parser_ops {
      bmv2ir.extract  regular @prs1_two
    }
    bmv2ir.state @parse_bottom
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @prs::@accept
    }
     parser_ops {
    }
    bmv2ir.state @accept
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default
    }
     parser_ops {
    }
  }
}


// CHECK:  "header_types": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "fields": [
// CHECK-NEXT:        [
// CHECK-NEXT:          "skip",
// CHECK-NEXT:          8,
// CHECK-NEXT:          false
// CHECK-NEXT:        ]
// CHECK-NEXT:      ],
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "name": "header_top"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "fields": [
// CHECK-NEXT:        [
// CHECK-NEXT:          "type",
// CHECK-NEXT:          8,
// CHECK-NEXT:          false
// CHECK-NEXT:        ],
// CHECK-NEXT:        [
// CHECK-NEXT:          "data",
// CHECK-NEXT:          8,
// CHECK-NEXT:          false
// CHECK-NEXT:        ]
// CHECK-NEXT:      ],
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "name": "header_one"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "fields": [
// CHECK-NEXT:        [
// CHECK-NEXT:          "type",
// CHECK-NEXT:          8,
// CHECK-NEXT:          false
// CHECK-NEXT:        ],
// CHECK-NEXT:        [
// CHECK-NEXT:          "data",
// CHECK-NEXT:          16,
// CHECK-NEXT:          false
// CHECK-NEXT:        ]
// CHECK-NEXT:      ],
// CHECK-NEXT:      "id": 2,
// CHECK-NEXT:      "name": "header_two"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "headers": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "header_type": "header_top",
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "metadata": false,
// CHECK-NEXT:      "name": "prs1_top"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "header_type": "header_one",
// CHECK-NEXT:      "id": 1,
// CHECK-NEXT:      "metadata": false,
// CHECK-NEXT:      "name": "prs1_one"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "header_type": "header_two",
// CHECK-NEXT:      "id": 2,
// CHECK-NEXT:      "metadata": false,
// CHECK-NEXT:      "name": "prs1_two"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "header_type": "header_one",
// CHECK-NEXT:      "id": 3,
// CHECK-NEXT:      "metadata": false,
// CHECK-NEXT:      "name": "prs_e_0"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "parsers": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "init_state": "start",
// CHECK-NEXT:      "name": "prs",
// CHECK-NEXT:      "parse_states": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 0,
// CHECK-NEXT:          "name": "start",
// CHECK-NEXT:          "parser_ops": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "op": "extract",
// CHECK-NEXT:              "parameters": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "regular",
// CHECK-NEXT:                  "value": "prs1_top"
// CHECK-NEXT:                }
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_headers",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 1,
// CHECK-NEXT:          "name": "parse_headers",
// CHECK-NEXT:          "parser_ops": [],
// CHECK-NEXT:          "transition_key": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "type": "lookahead",
// CHECK-NEXT:              "value": [
// CHECK-NEXT:                0,
// CHECK-NEXT:                8
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_one",
// CHECK-NEXT:              "type": "hexstr",
// CHECK-NEXT:              "value": "0x01"
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_two",
// CHECK-NEXT:              "type": "hexstr",
// CHECK-NEXT:              "value": "0x02"
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": "0x02",
// CHECK-NEXT:              "next_state": "parse_two",
// CHECK-NEXT:              "type": "hexstr",
// CHECK-NEXT:              "value": "0x01"
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_bottom",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 2,
// CHECK-NEXT:          "name": "parse_one",
// CHECK-NEXT:          "parser_ops": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "op": "extract",
// CHECK-NEXT:              "parameters": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "regular",
// CHECK-NEXT:                  "value": "prs_e_0"
// CHECK-NEXT:                }
// CHECK-NEXT:              ]
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "op": "assign_header",
// CHECK-NEXT:              "parameters": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "header",
// CHECK-NEXT:                  "value": "prs1_one"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "header",
// CHECK-NEXT:                  "value": "prs_e_0"
// CHECK-NEXT:                }
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_two",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 3,
// CHECK-NEXT:          "name": "parse_two",
// CHECK-NEXT:          "parser_ops": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "op": "extract",
// CHECK-NEXT:              "parameters": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "regular",
// CHECK-NEXT:                  "value": "prs1_two"
// CHECK-NEXT:                }
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "parse_bottom",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 4,
// CHECK-NEXT:          "name": "parse_bottom",
// CHECK-NEXT:          "parser_ops": [],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "accept",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 5,
// CHECK-NEXT:          "name": "accept",
// CHECK-NEXT:          "parser_ops": [],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": null,
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  ]

// -----

!b8i = !p4hir.bit<8>
module {
  bmv2ir.header_instance @prs_only_bit2 : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
  bmv2ir.header_instance @prs_only_bit1 : !bmv2ir.header<"bit_only", [bit:!p4hir.bit<8>], max_length = 1>
  bmv2ir.header_instance @prs_only_bit_top_0 : !bmv2ir.header<"header_top", [skip:!p4hir.bit<8>], max_length = 1>
  bmv2ir.parser @prs_only_bit init_state @prs_only_bit::@start {
    bmv2ir.state @start
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @prs_only_bit::@accept
    }
     parser_ops {
      bmv2ir.extract  regular @prs_only_bit_top_0
      %3 = bmv2ir.field @prs_only_bit_top_0["skip"] -> !b8i
      %4 = bmv2ir.field @prs_only_bit1["bit"] -> !b8i
      bmv2ir.assign %3 : !b8i to %4 : !b8i
      %5 = bmv2ir.field @prs_only_bit2["skip"] -> !b8i
      bmv2ir.assign %4 : !b8i to %5 : !b8i
    }
    bmv2ir.state @accept
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default
    }
     parser_ops {
    }
  }
}

// CHECK:   "header_types": [
// CHECK:     {
// CHECK:       "fields": [
// CHECK:         [
// CHECK:           "skip",
// CHECK:           8,
// CHECK:           false
// CHECK:         ]
// CHECK:       ],
// CHECK:       "id": 0,
// CHECK:       "name": "header_top"
// CHECK:     },
// CHECK:     {
// CHECK:       "fields": [
// CHECK:         [
// CHECK:           "bit",
// CHECK:           8,
// CHECK:           false
// CHECK:         ]
// CHECK:       ],
// CHECK:       "id": 1,
// CHECK:       "name": "bit_only"
// CHECK:     }
// CHECK:   ],
// CHECK:   "headers": [
// CHECK:     {
// CHECK:       "header_type": "header_top",
// CHECK:       "id": 0,
// CHECK:       "metadata": false,
// CHECK:       "name": "prs_only_bit2"
// CHECK:     },
// CHECK:     {
// CHECK:       "header_type": "bit_only",
// CHECK:       "id": 1,
// CHECK:       "metadata": false,
// CHECK:       "name": "prs_only_bit1"
// CHECK:     },
// CHECK:     {
// CHECK:       "header_type": "header_top",
// CHECK:       "id": 2,
// CHECK:       "metadata": false,
// CHECK:       "name": "prs_only_bit_top_0"
// CHECK:     }
// CHECK:   ],
// CHECK:   "parsers": [
// CHECK:     {
// CHECK:       "id": 0,
// CHECK:       "init_state": "start",
// CHECK:       "name": "prs_only_bit",
// CHECK:       "parse_states": [
// CHECK:         {
// CHECK:           "id": 0,
// CHECK:           "name": "start",
// CHECK:           "parser_ops": [
// CHECK:             {
// CHECK:               "op": "extract",
// CHECK:               "parameters": [
// CHECK:                 {
// CHECK:                   "type": "regular",
// CHECK:                   "value": "prs_only_bit_top_0"
// CHECK:                 }
// CHECK:               ]
// CHECK:             },
// CHECK:             {
// CHECK:               "op": "assign",
// CHECK:               "parameters": [
// CHECK:                 {
// CHECK:                   "type": "field",
// CHECK:                   "value": [
// CHECK:                     "prs_only_bit1",
// CHECK:                     "bit"
// CHECK:                   ]
// CHECK:                 },
// CHECK:                 {
// CHECK:                   "type": "field",
// CHECK:                   "value": [
// CHECK:                     "prs_only_bit_top_0",
// CHECK:                     "skip"
// CHECK:                   ]
// CHECK:                 }
// CHECK:               ]
// CHECK:             },
// CHECK:             {
// CHECK:               "op": "assign",
// CHECK:               "parameters": [
// CHECK:                 {
// CHECK:                   "type": "field",
// CHECK:                   "value": [
// CHECK:                     "prs_only_bit2",
// CHECK:                     "skip"
// CHECK:                   ]
// CHECK:                 },
// CHECK:                 {
// CHECK:                   "type": "field",
// CHECK:                   "value": [
// CHECK:                     "prs_only_bit1",
// CHECK:                     "bit"
// CHECK:                   ]
// CHECK:                 }
// CHECK:               ]
// CHECK:             }
// CHECK:           ],
// CHECK:           "transition_key": [],
// CHECK:           "transitions": [
// CHECK:             {
// CHECK:               "mask": null,
// CHECK:               "next_state": "accept",
// CHECK:               "type": "default",
// CHECK:               "value": null
// CHECK:             }
// CHECK:           ]
// CHECK:         },
// CHECK:         {
// CHECK:           "id": 1,
// CHECK:           "name": "accept",
// CHECK:           "parser_ops": [],
// CHECK:           "transition_key": [],
// CHECK:           "transitions": [
// CHECK:             {
// CHECK:               "mask": null,
// CHECK:               "next_state": null,
// CHECK:               "type": "default",
// CHECK:               "value": null
// CHECK:             }
// CHECK:           ]
// CHECK:         }
// CHECK:       ]
// CHECK:     }
// CHECK:   ]

// -----


!b32i = !p4hir.bit<32>
#int32_b32i = #p4hir.int<32> : !b32i
module {
  %c32_b32i = p4hir.const #int32_b32i
  bmv2ir.header_instance @headers_h : !bmv2ir.header<"H", [s:!p4hir.bit<8>, v:!p4hir.varbit<32>], max_length = 5>
  bmv2ir.parser @parser init_state @parser::@start {
    bmv2ir.state @start
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default next_state @parser::@accept
    }
     parser_ops {
      bmv2ir.extract_vl  regular @headers_h(%c32_b32i : !b32i)
    }
    bmv2ir.state @accept
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default
    }
     parser_ops {
    }
    bmv2ir.state @reject
     transition_key {
    }
     transitions {
      bmv2ir.transition type  default
    }
     parser_ops {
    }
  }

}

// CHECK:  "parsers": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "id": 0,
// CHECK-NEXT:      "init_state": "start",
// CHECK-NEXT:      "name": "parser",
// CHECK-NEXT:      "parse_states": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 0,
// CHECK-NEXT:          "name": "start",
// CHECK-NEXT:          "parser_ops": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "op": "extract_VL",
// CHECK-NEXT:              "parameters": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "regular",
// CHECK-NEXT:                  "value": "headers_h"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "type": "expression",
// CHECK-NEXT:                  "value": {
// CHECK-NEXT:                    "type": "hexstr",
// CHECK-NEXT:                    "value": "0x00000020"
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              ]
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": "accept",
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 1,
// CHECK-NEXT:          "name": "accept",
// CHECK-NEXT:          "parser_ops": [],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": null,
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "id": 2,
// CHECK-NEXT:          "name": "reject",
// CHECK-NEXT:          "parser_ops": [],
// CHECK-NEXT:          "transition_key": [],
// CHECK-NEXT:          "transitions": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "mask": null,
// CHECK-NEXT:              "next_state": null,
// CHECK-NEXT:              "type": "default",
// CHECK-NEXT:              "value": null
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
