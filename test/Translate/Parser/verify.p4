// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

error {
    NoError,
    SomeError
}

/// Check a predicate @check in the parser; if the predicate is true do nothing,
/// otherwise set the parser error to @toSignal, and transition to the `reject` state.
extern void verify(in bool check, in error toSignal);

parser p2(in bool check, out bool matches) {
    state start {
        verify(check == true, error.SomeError);
        transition next;
    }

    state next {
        matches = true;
        transition accept;
    }
}

// CHECK-LABEL: p4hir.func @verify
// CHECK-LABEL: p4hir.parser @p2(
// CHECK: p4hir.state @start {
// CHECK:      %[[true:.*]] = p4hir.const #true
// CHECK:      %[[eq:.*]] = p4hir.cmp(eq, %arg0, %[[true]]) : !p4hir.bool, !p4hir.bool
// CHECK:      %[[error_SomeError:.*]] = p4hir.const #error_SomeError
// CHECK:      p4hir.call @verify (%[[eq]], %[[error_SomeError]]) : (!p4hir.bool, !error) -> ()
// CHECK:      p4hir.transition to @p2::@next
// CHECK    }
