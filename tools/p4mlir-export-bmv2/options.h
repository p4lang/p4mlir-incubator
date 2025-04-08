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

#ifndef _P4MLIR_OPTIONS_H_
#define _P4MLIR_OPTIONS_H_

#include "frontends/common/options.h"
#include "frontends/common/parser_options.h"

namespace P4::MLIR {

class ExportOptions : public CompilerOptions {
 public:

    bool printLoc = false;
    bool jsonOutput = false;

    virtual ~ExportOptions() = default;

    ExportOptions();
    ExportOptions(const ExportOptions &) = default;
    ExportOptions(ExportOptions &&) = delete;
    ExportOptions &operator=(const ExportOptions &) = default;
    ExportOptions &operator=(ExportOptions &&) = delete;
};

using ExportContext = P4CContextWithOptions<ExportOptions>;

}  // namespace P4::MLIR

#endif /* _P4MLIR_OPTIONS_H_ */
