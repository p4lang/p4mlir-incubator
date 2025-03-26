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
#include "frontends/p4/specialize.h"
#include "frontends/p4/specializeGenericFunctions.h"
#include "frontends/p4/specializeGenericTypes.h"
#include "frontends/p4/staticAssert.h"
#include "frontends/p4/structInitializers.h"
#include "frontends/p4/tableKeyNames.h"
#include "frontends/p4/toP4/toP4.h"
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#pragma GCC diagnostic pop

#include "translate.h"

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

    mlir::MLIRContext context;
    context.getOrLoadDialect<P4::P4MLIR::P4HIR::P4HIRDialect>();

    auto mod = P4::P4MLIR::toMLIR(context, program, &typeMap);
    if (!mod) return EXIT_FAILURE;

    mlir::OpPrintingFlags flags;
    if (!options.noDump) mod->print(llvm::outs(), flags.enableDebugInfo(options.printLoc));

    if (P4::Log::verbose()) std::cerr << "Done." << std::endl;
    return P4::errorCount() > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
