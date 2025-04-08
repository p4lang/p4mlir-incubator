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

#include "options.h"

using namespace P4::MLIR;

ExportOptions::ExportOptions() {
    registerOption(
        "--print-loc", nullptr,
        [this](const char *) {
            printLoc = true;
            return true;
        },
        "print location information in MLIR dump");
    registerOption(
        "--json-output", nullptr,
        [this](const char *) {
            jsonOutput = true;
            return true;
        },
        "output MLIR module in JSON format");
}
