// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | diff -u - %s.ref
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | p4test -

error {
    NoError,
    SomeError
}

/// Check a predicate @check in the parser; if the predicate is true do nothing,
/// otherwise set the parser error to @toSignal, and transition to the `reject` state.
@corelib
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
// CHECK:      %[[eq:.*]] = p4hir.cmp(eq, %arg0 : !p4hir.bool, %[[true]] : !p4hir.bool)
// CHECK:      %[[error_SomeError:.*]] = p4hir.const #error_SomeError
// CHECK:      p4hir.call @verify (%[[eq]], %[[error_SomeError]]) : (!p4hir.bool, !error) -> ()
// CHECK:      p4hir.transition to @p2::@next
// CHECK    }
