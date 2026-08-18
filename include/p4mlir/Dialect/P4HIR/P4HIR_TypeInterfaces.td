// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def IndexableTypeInterface : TypeInterface<"IndexableTypeInterface"> {
  let description = [{
    Interface for a type that consists of fields that can be indexed.
  }];

  let methods = [
    InterfaceMethod<[{
      Get the number of fields that this type has.
    }],
      "unsigned", "getFieldCount", (ins)>,

    InterfaceMethod<[{
      Get the type for the field at a specific index.
    }],
      "mlir::Type", "getFieldType", (ins "unsigned":$index)>,

    InterfaceMethod<[{
      Get the name for the field at a specific index.
    }],
      "std::string", "getFieldName", (ins "unsigned":$index)>,

    InterfaceMethod<[{
      Get the name for the field at a specific index as a StringAttr.
      Override to provide more optimized access for cases where the indexable type
      already has a StringAttr for field names.
    }],
      "mlir::StringAttr", "getFieldNameAttr", (ins "unsigned":$index),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return mlir::StringAttr::get($_type.getContext(), $_type.getFieldName(index));
      }]>,
  
    InterfaceMethod<[{
      Get additional annotations for the field at a specific index.
    }],
      "mlir::DictionaryAttr", "getFieldAnnotations", (ins "unsigned":$index),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return {};
      }]>,
  ];

  let extraClassDeclaration = [{
    IndexedField getField(unsigned idx) {
      return P4HIR::IndexedField(*this, idx);
    }

    auto getFields() {
      return llvm::map_range(llvm::seq(0u, getFieldCount()), [this](unsigned idx) {
        return getField(idx);
      });
    }

    auto getFieldTypes() {
      return llvm::map_range(llvm::seq(0u, getFieldCount()), [this](unsigned idx) {
        return getField(idx).getType();
      });
    }
  }];

  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def StructLikeTypeInterface : TypeInterface<"StructLikeTypeInterface", [IndexableTypeInterface]> {
  let description = [{
    Common methods for struct-like types that could be viewed as a collection
    of named fields
  }];

  let methods = [
    InterfaceMethod<[{
      Access the array of struct field info.
    }],
      "llvm::ArrayRef<FieldInfo>", "getElements", (ins)>,

   InterfaceMethod<[{
      Get the field given the name in StringRef
    }],
     "std::optional<IndexedField>", "getFieldByName", (ins "mlir::StringRef":$fieldName)>,
  ];
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def AnnotatedType : TypeInterface<"AnnotatedType"> {
  let description = [{
    Common methods for for all operations that might have annotations attached to them
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Checks if the op has the annotation given the name",
      /*returnType=*/  "bool",
      /*methodName=*/  "hasAnnotation",
      /*args=*/        (ins "mlir::StringRef":$name),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        if (auto annotations = $_type.getAnnotationsAttr())
           return bool(annotations.get(name));

        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Get the annotation given the name, null if it is not present",
      /*returnType=*/  "mlir::Attribute",
      /*methodName=*/  "getAnnotation",
      /*args=*/        (ins "mlir::StringRef":$name),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        if (auto annotations = $_type.getAnnotationsAttr())
           return annotations.get(name);

        return nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Get annotations as DictionaryAttr. Returns nullptr if there are no annotations set",
      /*returnType=*/  "mlir::DictionaryAttr",
      /*methodName=*/  "getAnnotationsAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return $_type.getAnnotations();
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Get annotations",
      /*returnType=*/  "mlir::ArrayRef<mlir::NamedAttribute>",
      /*methodName=*/  "annotations",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        if (auto attrs = $_type.getAnnotations())
          return attrs.getValue();
        return {};
      }]
    >
  ];

  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def HasDefaultValue : TypeInterface<"HasDefaultValue"> {
  let description = [{
    Implemented by the types having a default value
    Definition: https://p4.org/p4-spec/docs/P4-16-working-spec.html#sec-default-values
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Provides the \"default\" value for a given type",
      /*returnType=*/  "mlir::TypedAttr",
      /*methodName=*/  "getDefaultValue"
    >
  ];
  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def EnumRepresentationInterface : TypeInterface<"EnumRepresentationInterface"> {
  let description = [{
    Interface to specify the underlying representation of an enum type when it is
    lowered to serialized form. This allows downstream users to provide a policy
    for the bit-width of enums.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Get the underlying type for an enum type",
      /*returnType=*/  "mlir::Type",
      /*methodName=*/  "getUnderlyingType",
      /*args=*/        (ins)
    >,
    InterfaceMethod<
      /*desc=*/        "Whether this enum should be converted to serializable form",
      /*returnType=*/  "bool",
      /*methodName=*/  "shouldConvert",
      /*args=*/        (ins)
    >,
    InterfaceMethod<
      /*desc=*/        "Get the numeric encoding for a specific enum field",
      /*returnType=*/  "llvm::APInt",
      /*methodName=*/  "getEncoding",
      /*args=*/        (ins "llvm::StringRef":$fieldName)
    >
  ];

  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_TD
