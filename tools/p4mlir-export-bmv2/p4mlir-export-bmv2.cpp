/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstdlib>
#include <iostream>
#include "options.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#pragma GCC diagnostic pop

#include "json_emitter.h"


int main(int argc, char *const argv[]) {

    P4::AutoCompileContext autoP4MLIRExportContext(new P4::MLIR::ExportContext);
    auto &options = P4::MLIR::ExportContext::get().options();
    options.langVersion = P4::CompilerOptions::FrontendVersion::P4_16;

    if (options.process(argc, argv) == nullptr || P4::errorCount() > 0) return EXIT_FAILURE;

    options.setInputFile();
    
    if (options.jsonOutput && options.file.extension().string() == ".mlir") {
        llvm::errs() << "Direct MLIR to JSON conversion\n";
        
        mlir::MLIRContext context;
        context.getOrLoadDialect<P4::P4MLIR::P4HIR::P4HIRDialect>();
        mlir::ParserConfig config(&context);
        auto moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(options.file.string(), config);
        if (!moduleRef) {
            llvm::errs() << "Failed to parse MLIR file: " << options.file.string() << "\n";
            return EXIT_FAILURE;
        }

        // Generate JSON
        P4::P4MLIR::JsonEmitter emitter;
        llvm::json::Value jsonOutput = emitter.emitModule(*moduleRef);
        llvm::json::OStream J(llvm::outs());
        J.value(jsonOutput);
        llvm::outs() << "\n";
        
        return EXIT_SUCCESS;
    }
    return P4::errorCount() > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}