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

TranslateOptions::TranslateOptions() {
    registerOption(
        "--parse-only", nullptr,
        [this](const char *) {
            parseOnly = true;
            return true;
        },
        "only parse the P4 input, without any further processing");
    registerOption(
        "--typeinference-only", nullptr,
        [this](const char *) {
            typeinferenceOnly = true;
            return true;
        },
        "parse the P4 input and run minimal set of frontend passes for type inference");
    registerOption(
        "--print-loc", nullptr,
        [this](const char *) {
            printLoc = true;
            return true;
        },
        "print location information in MLIR dump");
    registerOption(
        "--no-dump", nullptr,
        [this](const char *) {
            noDump = true;
            return true;
        },
        "do not dump module on exit");
}
