// SPDX-FileCopyrightText: 2024 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-SAME: p4hir
