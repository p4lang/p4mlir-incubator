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

#ifndef TOOLS_P4MLIR_TRANSLATE_OPTIONS_H_
#define TOOLS_P4MLIR_TRANSLATE_OPTIONS_H_

#include "frontends/common/options.h"
#include "frontends/common/parser_options.h"

namespace P4::MLIR {

class TranslateOptions : public CompilerOptions {
 public:
    bool parseOnly = false;
    bool typeinferenceOnly = false;
    bool printLoc = false;
    bool noDump = false;
    /// Export P4MLIR as P4 to the specified file.
    std::filesystem::path p4OutputFile = std::filesystem::path();
    /// Write P4MLIR as P4 to the command line.
    bool dumpToP4 = false;
    /// Only emit code relevant to the main package.
    bool mainPackageOnly = false;

    virtual ~TranslateOptions() = default;

    TranslateOptions();
    TranslateOptions(const TranslateOptions &) = default;
    TranslateOptions(TranslateOptions &&) = delete;
    TranslateOptions &operator=(const TranslateOptions &) = default;
    TranslateOptions &operator=(TranslateOptions &&) = delete;
};

using TranslateContext = P4CContextWithOptions<TranslateOptions>;

}  // namespace P4::MLIR

#endif /* TOOLS_P4MLIR_TRANSLATE_OPTIONS_H_ */
