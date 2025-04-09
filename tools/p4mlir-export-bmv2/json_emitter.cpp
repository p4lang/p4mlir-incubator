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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop


#include "json_emitter.h"


using namespace llvm::json;

namespace P4::P4MLIR {

llvm::json::Value JsonEmitter::emitModule(mlir::ModuleOp module) {
  Object moduleObj;
  moduleObj["type"] = "module";
  moduleObj["name"] = module.getName() ? module.getName()->str() : "";
  
  Array operations;
  for (auto &op : module.getOps()) {
    operations.push_back(emitOperation(&op));
  }
  moduleObj["operations"] = std::move(operations);
  
  return moduleObj;
}

llvm::json::Value JsonEmitter::emitOperation(mlir::Operation *op) {
  Object opObj;
  opObj["name"] = op->getName().getStringRef().str();
  
  Object attrs;
  for (auto namedAttr : op->getAttrs()) {
    attrs[namedAttr.getName().str()] = emitAttribute(namedAttr.getValue());
  }
  opObj["attributes"] = std::move(attrs);
  
  Array resultTypes;
  for (auto type : op->getResultTypes()) {
    resultTypes.push_back(emitType(type));
  }
  opObj["result_types"] = std::move(resultTypes);
  
  Array regions;
  for (auto &region : op->getRegions()) {
    regions.push_back(emitRegion(region));
  }
  opObj["regions"] = std::move(regions);
  
  return opObj;
}

llvm::json::Value JsonEmitter::emitAttribute(mlir::Attribute attr) {
  Object attrObj;
  
  if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
    attrObj["value"] = strAttr.getValue().str();
    attrObj["type"] = "string";
  } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
    attrObj["value"] = std::to_string(intAttr.getInt());
    attrObj["type"] = "integer";
  } else {
    std::string attrStr;
    llvm::raw_string_ostream os(attrStr);
    attr.print(os);
    attrObj["value"] = os.str();
    attrObj["type"] = "unknown";
  }
  
  return attrObj;
}

llvm::json::Value JsonEmitter::emitType(mlir::Type type) {
  Object typeObj;
  typeObj["dialect"] = type.getDialect().getNamespace().str();
  
  std::string typeStr;
  llvm::raw_string_ostream os(typeStr);
  type.print(os);
  typeObj["value"] = os.str();
  
  return typeObj;
}

llvm::json::Value JsonEmitter::emitRegion(mlir::Region &region) {
  Object regionObj;
  
  Array blocks;
  for (auto &block : region) {
    blocks.push_back(emitBlock(block));
  }
  regionObj["blocks"] = std::move(blocks);
  
  return regionObj;
}

llvm::json::Value JsonEmitter::emitBlock(mlir::Block &block) {
  Object blockObj;
  
  Array arguments;
  for (auto &arg : block.getArguments()) {
    Object argObj;
    argObj["type"] = emitType(arg.getType());
    arguments.push_back(std::move(argObj));
  }
  blockObj["arguments"] = std::move(arguments);
  
  Array operations;
  for (auto &op : block.getOperations()) {
    operations.push_back(emitOperation(&op));
  }
  blockObj["operations"] = std::move(operations);
  
  return blockObj;
}

} // namespace P4::P4MLIR

