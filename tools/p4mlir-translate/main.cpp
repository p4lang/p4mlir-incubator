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

#include "frontends/common/constantFolding.h"
#include "frontends/common/parseInput.h"
#include "frontends/common/parser_options.h"
#include "frontends/p4/checkCoreMethods.h"
#include "frontends/p4/checkNamedArgs.h"
#include "frontends/p4/createBuiltins.h"
#include "frontends/p4/defaultArguments.h"
#include "frontends/p4/defaultValues.h"
#include "frontends/p4/deprecated.h"
#include "frontends/p4/directCalls.h"
#include "frontends/p4/entryPriorities.h"
#include "frontends/p4/frontend.h"
#include "frontends/p4/getV1ModelVersion.h"
#include "frontends/p4/removeOpAssign.h"
#include "frontends/p4/removeReturns.h"
#include "frontends/p4/specialize.h"
#include "frontends/p4/specializeGenericFunctions.h"
#include "frontends/p4/specializeGenericTypes.h"
#include "frontends/p4/staticAssert.h"
#include "frontends/p4/structInitializers.h"
#include "frontends/p4/tableKeyNames.h"
#include "frontends/p4/typeChecking/bindVariables.h"
#include "frontends/p4/validateMatchAnnotations.h"
#include "frontends/p4/validateParsedProgram.h"
#include "frontends/p4/validateStringAnnotations.h"
#include "frontends/p4/validateValueSets.h"
#include "gc/gc.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/compile_context.h"
#include "lib/crash.h"
#include "lib/error.h"
#include "lib/gc.h"
#include "options.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/lib/Utilities/export_to_p4.h"
#include "p4mlir/Common/Registration.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Dialect/P4HIR/Pipelines/Passes.h"
#include "p4mlir/P4C/translate.h"
#pragma GCC diagnostic pop

namespace {
void log_dump(const P4::IR::Node *node, const char *head) {
    if (node && LOGGING(1)) {
        if (head)
            std::cout << '+' << std::setw(strlen(head) + 6) << std::setfill('-') << "+\n| " << head
                      << " |\n"
                      << '+' << std::setw(strlen(head) + 3) << "+" << std::endl
                      << std::setfill(' ');
        if (LOGGING(2))
            dump(node);
        else
            std::cout << *node << std::endl;
    }
}

/** Changes the value of strictStruct in the typeMap */
class SetStrictStruct : public P4::Inspector {
    P4::TypeMap *typeMap;
    bool strictStruct;

 public:
    SetStrictStruct(P4::TypeMap *typeMap, bool strict) : typeMap(typeMap), strictStruct(strict) {}
    bool preorder(const P4::IR::P4Program *) override { return false; }
    Visitor::profile_t init_apply(const P4::IR::Node *node) override {
        typeMap->setStrictStruct(strictStruct);
        return Inspector::init_apply(node);
    }
};

mlir::LogicalResult handleDiagnostic(mlir::Diagnostic &diag) {
    llvm::raw_ostream &os = llvm::errs();
    auto loc = diag.getLocation();

    {
        // RAII color management
        llvm::WithColor color(os);
        switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Error:
                color.changeColor(llvm::raw_ostream::RED, /*bold=*/true);
                os << "error";
                break;
            case mlir::DiagnosticSeverity::Warning:
                color.changeColor(llvm::raw_ostream::YELLOW, /*bold=*/true);
                os << "warning";
                break;
            case mlir::DiagnosticSeverity::Note:
                // Notes attached to errors/warnings are handled below.
                // Standalone notes might have different formatting.
                color.changeColor(llvm::raw_ostream::BLUE, /*bold=*/true);
                os << "note";
                break;
            case mlir::DiagnosticSeverity::Remark:
                color.changeColor(llvm::raw_ostream::GREEN);
                os << "remark";
                break;
        }
        os << ": ";
        os << loc << ": ";
        os << diag;
    }
    os << "\n";

    for (mlir::Diagnostic &note : diag.getNotes()) {
        llvm::WithColor color(os, llvm::raw_ostream::BLUE, /*bold=*/true);
        os << "note: ";
        os << note.getLocation() << ": ";
        os << note << "\n";
    }

    // Try to get FileLineColLoc for snippet printing
    mlir::FileLineColLoc fileLoc;
    fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc);

    if (!fileLoc) {
        return mlir::success();
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(fileLoc.getFilename().strref());

    if (std::error_code ec = fileOrErr.getError()) {
        return mlir::success();
    }
    llvm::StringRef buffer = fileOrErr.get()->getBuffer();
    unsigned line = fileLoc.getLine();
    unsigned col = fileLoc.getColumn();

    if (line > 0 && col > 0) {
        const int contextLines = 2;
        int firstLine = std::max(1, static_cast<int>(line) - contextLines);
        int lastLine = static_cast<int>(line) + contextLines;

        llvm::SmallVector<llvm::StringRef> lines;
        buffer.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

        // Adjust lastLine if it exceeds the number of lines in the file
        lastLine = std::min(lastLine, static_cast<int>(lines.size()));

        // Ensure firstLine is valid after potential adjustments
        firstLine = std::min(firstLine, lastLine);

        for (int i = firstLine; i <= lastLine; ++i) {
            // Format line number consistently
            os << llvm::format("%*d", 4, i) << " | ";

            // The line index is i-1 because vectors are 0-based
            llvm::StringRef lineContent = lines[i - 1];
            // Handle potential DOS line endings (\r\n) if present
            if (!lineContent.empty() && lineContent.back() == '\r') {
                lineContent = lineContent.drop_back();
            }

            if (i == static_cast<int>(line)) {
                // Print the highlight line
                {
                    llvm::WithColor color(os, llvm::raw_ostream::YELLOW);
                    os << lineContent;
                }
                os << "\n";

                // Print the caret line
                os << "     | ";
                // Account for potential multi-byte characters (basic handling)
                // More accurate handling requires locale/UTF-8 awareness
                for (unsigned c = 1; c < col; ++c) {
                    // Print space or tab appropriately
                    if (c - 1 < lineContent.size() && lineContent[c - 1] == '\t') {
                        os << '\t';
                    } else {
                        os << ' ';
                    }
                }
                llvm::WithColor color(os, llvm::raw_ostream::GREEN, /*bold=*/true);
                os << '^';
                // TODO: Could add ~~~ for ranges if the diagnostic provides it
            } else {
                // Print context line
                os << lineContent;
            }
            os << "\n";
        }
    }

    return mlir::success();
}

}  // namespace

int main(int argc, char *const argv[]) {
    setup_gc_logging();
    P4::setup_signals();

    P4::AutoCompileContext autoP4MLIRTranslateContext(new P4::MLIR::TranslateContext);
    auto &options = P4::MLIR::TranslateContext::get().options();
    options.langVersion = P4::CompilerOptions::FrontendVersion::P4_16;

    if (options.process(argc, argv) == nullptr || P4::errorCount() > 0) return EXIT_FAILURE;

    options.setInputFile();
    const auto *program = P4::parseP4File(options);

    if (program == nullptr || P4::errorCount() > 0) return EXIT_FAILURE;

    log_dump(program, "Parsed program");
    auto hook = options.getDebugHook();
    P4::TypeMap typeMap;
    if (!options.parseOnly) {
        if (options.typeinferenceOnly) {
            P4::FrontEndPolicy policy;

            P4::ParseAnnotations *parseAnnotations = policy.getParseAnnotations();
            if (!parseAnnotations) parseAnnotations = new P4::ParseAnnotations();

            P4::PassManager passes({
                new P4::P4V1::GetV1ModelVersion,
                // Parse annotations
                new P4::ParseAnnotationBodies(parseAnnotations, &typeMap),
                // Simple checks on parsed program
                new P4::ValidateParsedProgram(),
                // Synthesize some built-in constructs
                new P4::CreateBuiltins(),
                new P4::CheckShadowing(),
                // First pass of constant folding, before types are known --
                // may be needed to compute types.
                new P4::ConstantFolding(policy.getConstantFoldingPolicy()),
                // Validate @name/@deprecated/@noWarn. Should run after constant folding.
                new P4::ValidateStringAnnotations(),
                new P4::InstantiateDirectCalls(),
                new P4::Deprecated(),
                new P4::CheckNamedArgs(),
                // Type checking and type inference.  Also inserts
                // explicit casts where implicit casts exist.
                new SetStrictStruct(&typeMap, true),  // Next pass uses strict struct checking
                new P4::TypeInference(&typeMap, false, false),  // insert casts, don't check arrays
                new SetStrictStruct(&typeMap, false),
                new P4::ValidateMatchAnnotations(&typeMap),
                new P4::ValidateValueSets(),
                new P4::DefaultValues(&typeMap),
                new P4::BindTypeVariables(&typeMap),
                new P4::EntryPriorities(),
                new P4::PassRepeated({
                    new P4::SpecializeGenericTypes(&typeMap),
                    new P4::DefaultArguments(
                        &typeMap),  // add default argument values to parameters
                    new SetStrictStruct(&typeMap, true),  // Next pass uses strict struct checking
                    new P4::TypeInference(&typeMap, false),  // more casts may be needed
                    new SetStrictStruct(&typeMap, false),
                    new P4::SpecializeGenericFunctions(&typeMap),
                }),
                new P4::CheckCoreMethods(&typeMap),
                new P4::StaticAssert(&typeMap),
                new P4::StructInitializers(&typeMap),  // TODO: Decide if we can do the same at MLIR
                                                       // level to reduce GC traffic
                new P4::TableKeyNames(&typeMap),
                new P4::TypeChecking(nullptr, &typeMap, true),
            });
            passes.setName("TypeInference");
            passes.setStopOnError(true);
            passes.addDebugHook(hook, true);
            program = program->apply(passes);
        } else {
            // Apply the front end passes. These are usually fixed.
            P4::FrontEnd fe;
            fe.addDebugHook(hook);
            program = fe.run(options, program);

            P4::PassManager post({
                new P4::TypeChecking(nullptr, &typeMap, true),
            });
            post.setName("TypeInference");
            post.setStopOnError(true);
            post.addDebugHook(hook, true);
            program = program->apply(post);
        }
    }

    if (P4::errorCount() > 0) return EXIT_FAILURE;

    // BUG_CHECK(options.typeinferenceOnly, "TODO: fill TypeMap");

    log_dump(program, "After frontend");

    // MLIR uses thread local storage which is not registered by GC causing
    // double frees
#if HAVE_LIBGC
    GC_disable();
#endif

    mlir::DialectRegistry registry;
    registry.insert<P4::P4MLIR::P4HIR::P4HIRDialect, P4::P4MLIR::P4CoreLib::P4CoreLibDialect>();
    P4::P4MLIR::registerCommonFrontEndPipeline();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    context.getDiagEngine().registerHandler(handleDiagnostic);

    auto mod = P4::P4MLIR::toMLIR(context, program, &typeMap);
    if (!mod) return EXIT_FAILURE;

    if (options.mlirFrontend) {
        P4::P4MLIR::CommonFrontEndPipelineOpts mlirFEOptions;
        mlir::PassManager pm(&context);
        P4::P4MLIR::buildCommonFrontEndPassPipeline(pm, mlirFEOptions);

        if (failed(pm.run(*mod))) {
            llvm::errs() << "Failed to run MLIR passes\n";
            return EXIT_FAILURE;
        }
    }

    mlir::OpPrintingFlags flags;
    if (!options.noDump) mod->print(llvm::outs(), flags.enableDebugInfo(options.printLoc));

    if (P4::Log::verbose()) std::cerr << "Done." << std::endl;

    P4::P4MLIR::Utilities::P4HirToP4ExporterOptions formatterOutput;
    formatterOutput.mainPackageOnly = options.mainPackageOnly;
    if (!options.p4OutputFile.empty()) {
        if (failed(P4::P4MLIR::Utilities::writeP4HirToP4File(*mod, options.p4OutputFile))) {
            return EXIT_FAILURE;
        }
    } else if (options.dumpToP4) {
        auto result = P4::P4MLIR::Utilities::exportP4HirToP4(*mod, formatterOutput);
        if (failed(result)) {
            return EXIT_FAILURE;
        }
        llvm::outs() << *result;
    }

    return P4::errorCount() > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
