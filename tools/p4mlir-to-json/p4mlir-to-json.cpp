// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "p4mlir/Common/Registration.h"

using namespace mlir;

int main(int argc, char **argv) {
    P4::P4MLIR::registerAllTranslations();
    return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
