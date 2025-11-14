#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_TYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def FieldIDTypeInterface : TypeInterface<"FieldIDTypeInterface"> {
  let description = [{
    Common methods for types which can be indexed by a FieldID.
    FieldID is a depth-first numbering of the elements of a type.  For example:
    ```
    struct a  /* 0 */ {
      int b; /* 1 */
      struct c /* 2 */ {
        int d; /* 3 */
      }
    }

    int e; /* 0 */
    ```
  }];

  let methods = [
    InterfaceMethod<"Get the maximum field ID for this type",
      "unsigned", "getMaxFieldID">,

    InterfaceMethod<[{
      Get the sub-type of a type for a field ID, and the subfield's ID. Strip
      off a single layer of this type and return the sub-type and a field ID
      targeting the same field, but rebased on the sub-type.

      The resultant type *may* not be a FieldIDTypeInterface if the resulting 
      fieldID is zero.  This means that leaf types may be ground without 
      implementing an interface.  An empty aggregate will also appear as a 
      zero.
    }],
      "std::pair<::mlir::Type, unsigned>", "getSubTypeByFieldID", (ins "unsigned":$fieldID)>,

    InterfaceMethod<[{
      Returns the effective field id when treating the index field as the
      root of the type.  Essentially maps a fieldID to a fieldID after a
      subfield op. Returns the new id and whether the id is in the given
      child.
    }],
      "std::pair<unsigned, bool>", "projectToChildFieldID", (ins "unsigned":$fieldID, "unsigned":$index)>,

    InterfaceMethod<[{
      Returns the index (e.g. struct or vector element) for a given FieldID.
      This returns the containing index in the case that the fieldID points to a
      child field of a field.
    }],
      "unsigned", "getIndexForFieldID", (ins "unsigned":$fieldID)>,

    InterfaceMethod<[{
      Return the fieldID of a given index (e.g. struct or vector element).
      Field IDs start at 1, and are assigned
      to each field in a recursive depth-first walk of all
      elements. A field ID of 0 is used to reference the type itself.
    }],
      "unsigned", "getFieldID", (ins "unsigned":$index)>,

    InterfaceMethod<[{
      Find the index of the element that contains the given fieldID.
      As well, rebase the fieldID to the element.
    }],
      "std::pair<unsigned, unsigned>", "getIndexAndSubfieldID", (ins "unsigned":$fieldID)>,

  ];

  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

def StructLikeTypeInterface : TypeInterface<"StructLikeTypeInterface", [FieldIDTypeInterface]> {
  let description = [{
    Common methods for struct-like types that could be viewed as a collection
    of named fields
  }];

  let methods = [
    InterfaceMethod<[{
      Get the type of the field at a given index
    }],
      "mlir::Type", "getFieldType", (ins "mlir::StringRef":$fieldName)>,
   InterfaceMethod<[{
      Get the field index given the name in StringRef
    }],
     "std::optional<unsigned>", "getFieldIndex", (ins "mlir::StringRef":$fieldName)>,
   InterfaceMethod<[{
      Get the field given the name in StringRef
    }],
     "std::optional<FieldInfo>", "getField", (ins "mlir::StringRef":$fieldName)>,
   InterfaceMethod<[{
      Get all the fields.
    }],
     "llvm::ArrayRef<FieldInfo>", "getFields">,
   InterfaceMethod<[{
      Get all types of the fields.
    }],
     "void", "getInnerTypes", (ins "mlir::SmallVectorImpl<mlir::Type> &":$types)>,
   InterfaceMethod<[{
      Get the field given the ID.
    }],
     "FieldInfo", "getFieldByID", (ins "unsigned":$fieldID)>,
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
