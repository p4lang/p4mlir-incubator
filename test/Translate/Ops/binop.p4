// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

action bit_binops() {
    bit<10> res;

    bit<10> lhs = 1;
    bit<10> rhs = 2;

    bit<10> r1 = lhs + rhs;
    bit<10> r2 = lhs - rhs;
    bit<10> r3 = lhs * rhs;

    bit<10> r4 = lhs * 3;
    bit<10> r5 = 2 * rhs;

    bit<10> r6 = rhs * rhs;

    bit<10> r7 = lhs + rhs * 3 + (2 - rhs);

    bit<10> r8 = lhs / rhs;
    bit<10> r9 = lhs % lhs;

    bit<10> r10 = lhs |+| rhs;
    bit<10> r11 = lhs |-| rhs;

    bit<10> r12 = lhs | rhs;
    bit<10> r13 = lhs & rhs;
    bit<10> r14 = lhs ^ rhs;
}

action int_binops() {
    int<10> res;

    int<10> lhs = 1;
    int<10> rhs = 2;

    int<10> r1 = lhs + rhs;
    int<10> r2 = lhs - rhs;
    int<10> r3 = lhs * rhs;

    int<10> r4 = lhs * 3;
    int<10> r5 = 2 * rhs;

    int<10> r6 = rhs * rhs;
}

// CHECK-LABEL:   p4hir.func action @bit_binops()
// CHECK:         %[[VAL_0:.*]] = p4hir.variable ["res"] : <!b10i>
// CHECK:         %[[VAL_1:.*]] = p4hir.const #int1_b10i
// CHECK:         %[[VAL_2:.*]] = p4hir.cast(%[[VAL_1]] : !b10i) : !b10i
// CHECK:         %[[VAL_3:.*]] = p4hir.variable ["lhs", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_2]], %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_4:.*]] = p4hir.const #int2_b10i
// CHECK:         %[[VAL_5:.*]] = p4hir.cast(%[[VAL_4]] : !b10i) : !b10i
// CHECK:         %[[VAL_6:.*]] = p4hir.variable ["rhs", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_5]], %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_7:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_8:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_9:.*]] = p4hir.binop(add, %[[VAL_7]], %[[VAL_8]]) : !b10i
// CHECK:         %[[VAL_10:.*]] = p4hir.variable ["r1", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_9]], %[[VAL_10]] : <!b10i>
// CHECK:         %[[VAL_11:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_12:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_13:.*]] = p4hir.binop(sub, %[[VAL_11]], %[[VAL_12]]) : !b10i
// CHECK:         %[[VAL_14:.*]] = p4hir.variable ["r2", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_13]], %[[VAL_14]] : <!b10i>
// CHECK:         %[[VAL_15:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_16:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_17:.*]] = p4hir.binop(mul, %[[VAL_15]], %[[VAL_16]]) : !b10i
// CHECK:         %[[VAL_18:.*]] = p4hir.variable ["r3", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_17]], %[[VAL_18]] : <!b10i>
// CHECK:         %[[VAL_19:.*]] = p4hir.const #int3_b10i
// CHECK:         %[[VAL_20:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_21:.*]] = p4hir.binop(mul, %[[VAL_20]], %[[VAL_19]]) : !b10i
// CHECK:         %[[VAL_22:.*]] = p4hir.variable ["r4", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_21]], %[[VAL_22]] : <!b10i>
// CHECK:         %[[VAL_23:.*]] = p4hir.const #int2_b10i
// CHECK:         %[[VAL_24:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_25:.*]] = p4hir.binop(mul, %[[VAL_23]], %[[VAL_24]]) : !b10i
// CHECK:         %[[VAL_26:.*]] = p4hir.variable ["r5", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_25]], %[[VAL_26]] : <!b10i>
// CHECK:         %[[VAL_27:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_28:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_29:.*]] = p4hir.binop(mul, %[[VAL_27]], %[[VAL_28]]) : !b10i
// CHECK:         %[[VAL_30:.*]] = p4hir.variable ["r6", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_29]], %[[VAL_30]] : <!b10i>
// CHECK:         %[[VAL_31:.*]] = p4hir.const #int3_b10i
// CHECK:         %[[VAL_32:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_33:.*]] = p4hir.binop(mul, %[[VAL_32]], %[[VAL_31]]) : !b10i
// CHECK:         %[[VAL_34:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_35:.*]] = p4hir.binop(add, %[[VAL_34]], %[[VAL_33]]) : !b10i
// CHECK:         %[[VAL_36:.*]] = p4hir.const #int2_b10i
// CHECK:         %[[VAL_37:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_38:.*]] = p4hir.binop(sub, %[[VAL_36]], %[[VAL_37]]) : !b10i
// CHECK:         %[[VAL_39:.*]] = p4hir.binop(add, %[[VAL_35]], %[[VAL_38]]) : !b10i
// CHECK:         %[[VAL_40:.*]] = p4hir.variable ["r7", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_39]], %[[VAL_40]] : <!b10i>
// CHECK:         %[[VAL_41:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_42:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_43:.*]] = p4hir.binop(div, %[[VAL_41]], %[[VAL_42]]) : !b10i
// CHECK:         %[[VAL_44:.*]] = p4hir.variable ["r8", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_43]], %[[VAL_44]] : <!b10i>
// CHECK:         %[[VAL_45:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_46:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_47:.*]] = p4hir.binop(mod, %[[VAL_45]], %[[VAL_46]]) : !b10i
// CHECK:         %[[VAL_48:.*]] = p4hir.variable ["r9", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_47]], %[[VAL_48]] : <!b10i>
// CHECK:         %[[VAL_49:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_50:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_51:.*]] = p4hir.binop(sadd, %[[VAL_49]], %[[VAL_50]]) : !b10i
// CHECK:         %[[VAL_52:.*]] = p4hir.variable ["r10", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_51]], %[[VAL_52]] : <!b10i>
// CHECK:         %[[VAL_53:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_54:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_55:.*]] = p4hir.binop(ssub, %[[VAL_53]], %[[VAL_54]]) : !b10i
// CHECK:         %[[VAL_56:.*]] = p4hir.variable ["r11", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_55]], %[[VAL_56]] : <!b10i>
// CHECK:         %[[VAL_57:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_58:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_59:.*]] = p4hir.binop(or, %[[VAL_57]], %[[VAL_58]]) : !b10i
// CHECK:         %[[VAL_60:.*]] = p4hir.variable ["r12", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_59]], %[[VAL_60]] : <!b10i>
// CHECK:         %[[VAL_61:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_62:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_63:.*]] = p4hir.binop(and, %[[VAL_61]], %[[VAL_62]]) : !b10i
// CHECK:         %[[VAL_64:.*]] = p4hir.variable ["r13", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_63]], %[[VAL_64]] : <!b10i>
// CHECK:         %[[VAL_65:.*]] = p4hir.read %[[VAL_3]] : <!b10i>
// CHECK:         %[[VAL_66:.*]] = p4hir.read %[[VAL_6]] : <!b10i>
// CHECK:         %[[VAL_67:.*]] = p4hir.binop(xor, %[[VAL_65]], %[[VAL_66]]) : !b10i
// CHECK:         %[[VAL_68:.*]] = p4hir.variable ["r14", init] : <!b10i>
// CHECK:         p4hir.assign %[[VAL_67]], %[[VAL_68]] : <!b10i>
// CHECK-LABEL:   p4hir.func action @int_binops()
// CHECK:         %[[VAL_69:.*]] = p4hir.variable ["res"] : <!i10i>
// CHECK:         %[[VAL_70:.*]] = p4hir.const #int1_i10i
// CHECK:         %[[VAL_71:.*]] = p4hir.cast(%[[VAL_70]] : !i10i) : !i10i
// CHECK:         %[[VAL_72:.*]] = p4hir.variable ["lhs", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_71]], %[[VAL_72]] : <!i10i>
// CHECK:         %[[VAL_73:.*]] = p4hir.const #int2_i10i
// CHECK:         %[[VAL_74:.*]] = p4hir.cast(%[[VAL_73]] : !i10i) : !i10i
// CHECK:         %[[VAL_75:.*]] = p4hir.variable ["rhs", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_74]], %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_76:.*]] = p4hir.read %[[VAL_72]] : <!i10i>
// CHECK:         %[[VAL_77:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_78:.*]] = p4hir.binop(add, %[[VAL_76]], %[[VAL_77]]) : !i10i
// CHECK:         %[[VAL_79:.*]] = p4hir.variable ["r1", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_78]], %[[VAL_79]] : <!i10i>
// CHECK:         %[[VAL_80:.*]] = p4hir.read %[[VAL_72]] : <!i10i>
// CHECK:         %[[VAL_81:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_82:.*]] = p4hir.binop(sub, %[[VAL_80]], %[[VAL_81]]) : !i10i
// CHECK:         %[[VAL_83:.*]] = p4hir.variable ["r2", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_82]], %[[VAL_83]] : <!i10i>
// CHECK:         %[[VAL_84:.*]] = p4hir.read %[[VAL_72]] : <!i10i>
// CHECK:         %[[VAL_85:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_86:.*]] = p4hir.binop(mul, %[[VAL_84]], %[[VAL_85]]) : !i10i
// CHECK:         %[[VAL_87:.*]] = p4hir.variable ["r3", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_86]], %[[VAL_87]] : <!i10i>
// CHECK:         %[[VAL_88:.*]] = p4hir.const #int3_i10i
// CHECK:         %[[VAL_89:.*]] = p4hir.read %[[VAL_72]] : <!i10i>
// CHECK:         %[[VAL_90:.*]] = p4hir.binop(mul, %[[VAL_89]], %[[VAL_88]]) : !i10i
// CHECK:         %[[VAL_91:.*]] = p4hir.variable ["r4", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_90]], %[[VAL_91]] : <!i10i>
// CHECK:         %[[VAL_92:.*]] = p4hir.const #int2_i10i
// CHECK:         %[[VAL_93:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_94:.*]] = p4hir.binop(mul, %[[VAL_92]], %[[VAL_93]]) : !i10i
// CHECK:         %[[VAL_95:.*]] = p4hir.variable ["r5", init] : <!i10i>
// CHECK:         p4hir.assign %[[VAL_94]], %[[VAL_95]] : <!i10i>
// CHECK:         %[[VAL_96:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_97:.*]] = p4hir.read %[[VAL_75]] : <!i10i>
// CHECK:         %[[VAL_98:.*]] = p4hir.binop(mul, %[[VAL_96]], %[[VAL_97]]) : !i10i
// CHECK:         %[[VAL_99:.*]] = p4hir.variable ["r6", init] : <!i10i>
