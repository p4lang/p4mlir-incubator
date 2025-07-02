#ifndef P4HIR_OPS_INTERFACES
#define P4HIR_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def Annotated : OpInterface<"Annotated"> {
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
        if (auto annotations = $_op.getAnnotationsAttr())
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
        if (auto annotations = $_op.getAnnotationsAttr())
           return annotations.get(name);

        return nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Erase the annotation with the given name from the list. Returns the annotation that was erased, or nullptr if there was no annotation with such name.",
      /*returnType=*/  "mlir::Attribute",
      /*methodName=*/  "removeAnnotation",
      /*args=*/        (ins "mlir::StringRef":$name),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        mlir::NamedAttrList attrs($_op.getAnnotationsAttr());
        auto erased = attrs.erase(name);
        $_op.setAnnotationsAttr(attrs.getDictionary($_op.getContext()));
        return erased;
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "If the an annotation exists with the specified name, change it to the new value. Otherwise, add a new annotation with the specified name/value.",
      /*returnType=*/  "void",
      /*methodName=*/  "setAnnotation",
      /*args=*/        (ins "mlir::StringRef":$name, "mlir::Attribute":$value),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        mlir::NamedAttrList attrs($_op.getAnnotationsAttr());
        attrs.set(name, value);
        $_op.setAnnotationsAttr(attrs.getDictionary($_op.getContext()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Get annotations as DictionaryAttr. Returns nullptr if there are no annotations set",
      /*returnType=*/  "mlir::DictionaryAttr",
      /*methodName=*/  "getAnnotationsAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return $_op.getAnnotationsAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Get annotations",
      /*returnType=*/  "mlir::ArrayRef<mlir::NamedAttribute>",
      /*methodName=*/  "annotations",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        if (auto attrs = $_op.getAnnotationsAttr())
          return attrs.getValue();
        return {};
      }]
    >
  ];

  let cppNamespace = "::P4::P4MLIR::P4HIR";
}

#endif  // P4HIR_OPS_INTERFACES
