#include "mlir_to_p4.h"

#include <iostream>
#include <sstream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#pragma GCC diagnostic pop
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"

namespace P4::P4MLIR {

class FormattedStream {
 private:
    std::stringstream ss_;
    int indent_level_ = 0;
    std::string indent_string_ = "    ";
    bool needs_indent_ = true;

    void apply_indent() {
        if (needs_indent_) {
            for (int i = 0; i < indent_level_; ++i) {
                ss_ << indent_string_;
            }
            needs_indent_ = false;
        }
    }

 public:
    FormattedStream() = default;

    explicit FormattedStream(const std::string &indent_unit) : indent_string_(indent_unit) {}

    FormattedStream &indent() {
        ++indent_level_;
        return *this;
    }

    FormattedStream &dedent() {
        if (indent_level_ > 0) {
            --indent_level_;
        } else {
            std::cerr << "Warning: Attempted to dedent below level 0." << std::endl;
        }
        return *this;
    }

    void set_indent_string(const std::string &indent_unit) { indent_string_ = indent_unit; }

    int get_indent_level() const { return indent_level_; }

    // --- Output Operators ---
    // Generic output operator
    template <typename T>
    FormattedStream &operator<<(const T &value) {
        apply_indent();  // Add indentation if needed before writing
        ss_ << value;
        return *this;
    }

    // --- Special Formatting Methods ---
    FormattedStream &newline() {
        ss_ << '\n';
        needs_indent_ = true;  // Next output needs indentation
        return *this;
    }

    FormattedStream &tab() {
        apply_indent();
        ss_ << indent_string_;
        return *this;
    }

    FormattedStream &space(int count = 1) {
        apply_indent();
        for (int i = 0; i < count; ++i) ss_ << ' ';
        return *this;
    }

    FormattedStream &openBrace(bool newline_after = true) {
        apply_indent();
        ss_ << '{';
        if (newline_after) {
            newline();
        }
        indent();
        return *this;
    }

    FormattedStream &closeBrace(bool newline_after = true, bool indent_before = true) {
        dedent();
        if (indent_before) {
            apply_indent();
        } else {
            needs_indent_ = false;
        }
        ss_ << '}';
        if (newline_after) {
            newline();
        }
        return *this;
    }

    FormattedStream &semicolon(bool newline_after = true) {
        apply_indent();
        ss_ << ';';
        if (newline_after) {
            newline();
        }
        return *this;
    }

    std::string str() const { return ss_.str(); }

    class Scope {
        FormattedStream &cs_;
        bool manage_braces_;
        bool newline_before_close_;
        bool indent_closing_brace_;
        std::string custom_open_;
        std::string custom_close_;

     public:
        Scope(FormattedStream &cs, bool manage_braces = true, bool newline_after_open = true,
              bool indent_closing = true, bool newline_before_close = true)
            : cs_(cs),
              manage_braces_(manage_braces),
              newline_before_close_(newline_before_close),
              indent_closing_brace_(indent_closing) {
            if (manage_braces_) {
                cs_.apply_indent();
                cs_.ss_ << '{';
                if (newline_after_open) {
                    cs_.newline();
                }
            }
            cs_.indent();
        }

        // Constructor for custom delimiters (e.g. parentheses, brackets)
        Scope(FormattedStream &cs, const std::string &open_delim, const std::string &close_delim,
              bool newline_after_open = false, bool indent_closing = false,
              bool newline_before_close = false)
            : cs_(cs),
              manage_braces_(false),  // Don't manage braces by default
              newline_before_close_(newline_before_close),
              indent_closing_brace_(indent_closing),
              custom_open_(open_delim),
              custom_close_(close_delim) {
            cs_.apply_indent();
            cs_.ss_ << custom_open_;
            if (newline_after_open) {
                cs_.newline();
            }
            cs_.indent();
        }

        ~Scope() {
            cs_.dedent();
            if (manage_braces_ || !custom_close_.empty()) {
                if (indent_closing_brace_) {
                    cs_.apply_indent();
                } else {
                    cs_.needs_indent_ = false;  // Prevent indent if not desired
                }
                cs_.ss_ << (manage_braces_ ? "}" : custom_close_);
                if (newline_before_close_) {
                    cs_.newline();
                }
            }
            // No std::uncaught_exceptions() check needed here,
            // dedent is safe even during stack unwinding.
        }

        // Make Scope non-copyable and non-movable
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        Scope(Scope &&) = delete;
        Scope &operator=(Scope &&) = delete;
    };

    // Factory function for Scope for cleaner usage
    [[nodiscard]] Scope createScope(bool manage_braces = true, bool newline_after_open = true,
                                    bool indent_closing = true, bool newline_before_close = true) {
        return Scope(*this, manage_braces, newline_after_open, indent_closing,
                     newline_before_close);
    }
    [[nodiscard]] Scope createScope(const std::string &open_delim, const std::string &close_delim,
                                    bool newline_after_open = false, bool indent_closing = false,
                                    bool newline_before_close = false) {
        return Scope(*this, open_delim, close_delim, newline_after_open, indent_closing,
                     newline_before_close);
    }
};

MlirToP4::MlirToP4(mlir::ModuleOp module) : module(module) {}

namespace {

void complexTypeToP4String(const mlir::Type &type, FormattedStream &ss);
void referenceToP4String(const mlir::Operation &type, FormattedStream &ss);
void constantToP4String(const P4HIR::ConstOp &constant, FormattedStream &ss);
void convertExpression(mlir::Operation &op, FormattedStream &ss);
void convertStatement(mlir::Operation &op, FormattedStream &ss);

void complexTypeToP4String(const mlir::Type &type, FormattedStream &ss) {
    if (auto ref = llvm::dyn_cast<P4HIR::ReferenceType>(type)) {
        complexTypeToP4String(ref.getObjectType(), ss);
    } else if (auto headerType = llvm::dyn_cast<P4HIR::HeaderType>(type)) {
        ss << headerType.getName().str();
    } else if (auto structType = llvm::dyn_cast<P4HIR::StructType>(type)) {
        ss << structType.getName().str();
    } else if (auto headerUnion = llvm::dyn_cast<P4HIR::HeaderUnionType>(type)) {
        ss << headerUnion.getName().str();
    } else if (auto externType = llvm::dyn_cast<P4HIR::ExternType>(type)) {
        ss << externType.getName().str();
    } else {
        llvm::errs() << "Unsupported " << type << "\n";
        llvm::report_fatal_error("Encountered unhandled type.");
    }
}

void referenceToP4String(const mlir::Operation &type, FormattedStream &ss) {
    if (auto var = llvm::dyn_cast<P4HIR::StructExtractRefOp>(type)) {
        if (auto ref = var.getInput().getDefiningOp()) {
            referenceToP4String(*ref, ss);
        } else {
            complexTypeToP4String(var.getInput().getType(), ss);
        }
        ss << "." << var.getFieldName().str();
    } else {
        llvm::errs() << "Unsupported " << type << "\n";
    }
}

void constantToP4String(const P4HIR::ConstOp &constant, FormattedStream &ss) {
    for (auto namedAttr : constant->getAttrs()) {
        mlir::Attribute attrValue = namedAttr.getValue();
        if (auto intAttr = attrValue.dyn_cast<P4HIR::IntAttr>()) {
            // TODO: Cleanup.
            llvm::SmallVector<char, 64> valueBuffer;
            intAttr.getValue().toString(valueBuffer, 10, false);
            std::string valueStr(valueBuffer.begin(), valueBuffer.end());
            ss << valueStr;
            return;
        } else if (auto strAttr = attrValue.dyn_cast<mlir::StringAttr>()) {
            ss << strAttr.getValue().str();
            return;
        } else {
            llvm::outs() << "  -> Unhandled Attribute Type: " << attrValue << "\n";
        }
    }
    // Consider adding error handling if no valid attribute is found
}

void convertExpression(mlir::Operation &op, FormattedStream &ss) {
    if (auto constOp = llvm::dyn_cast<P4HIR::ConstOp>(op)) {
        constantToP4String(constOp, ss);
    } else {
        llvm::errs() << "Unhandled operation: " << op.getName() << "\n";
    }
}

void convertControlApply(P4HIR::ControlApplyOp &op, FormattedStream &ss) {
    ss << "apply ";
    ss.openBrace();
    for (auto &nestedOp : op.getOps()) {
        convertStatement(nestedOp, ss);
    }
    ss.closeBrace();
}

void convertDirection(const P4HIR::ParamDirectionAttr &direction, FormattedStream &ss) {
    if (direction.getValue() == P4HIR::ParamDirection::In) {
        ss << "in ";
    } else if (direction.getValue() == P4HIR::ParamDirection::Out) {
        ss << "out ";
    } else if (direction.getValue() == P4HIR::ParamDirection::InOut) {
        ss << "inout ";
    } else if (direction.getValue() == P4HIR::ParamDirection::None) {
        // No direction keyword in the case of none.
    } else {
        llvm::errs() << "Unhandled direction: " << direction << "\n";
    }
}

void convertArgumentList(mlir::FunctionOpInterface functionInterface, FormattedStream &ss) {
    bool needsComma = false;

    for (auto arg : functionInterface.getArguments()) {
        if (needsComma) {
            ss << ", ";
        }
        auto direction =
            functionInterface.getArgAttr(arg.getArgNumber(), P4HIR::FuncOp::getDirectionAttrName());
        auto castDirection = llvm::dyn_cast<P4HIR::ParamDirectionAttr>(direction);
        convertDirection(castDirection, ss);

        complexTypeToP4String(arg.getType(), ss);
        ss << " ";

        auto nameAttr =
            functionInterface.getArgAttr(arg.getArgNumber(), P4HIR::FuncOp::getParamNameAttrName());
        if (nameAttr) {
            if (auto nameStrAttr = nameAttr.dyn_cast<mlir::StringAttr>()) {
                ss << nameStrAttr.getValue().str();
            } else {
                llvm::errs() << "Param name attribute has unexpected type: " << nameAttr << "\n";
            }
        } else {
            llvm::errs() << "Missing param name attribute for function arg " << arg.getArgNumber()
                         << "\n";
            ss << "_param" << arg.getArgNumber();
        }
        needsComma = true;
    }
}

void convertParser(P4HIR::ParserOp &parserOp, FormattedStream &ss) {
    ss << "parser " << parserOp.getName().str() << "(";
    convertArgumentList(parserOp, ss);
    ss << ")";
    ss.openBrace();
    for (auto &nestedOp : parserOp.getOps()) {
        convertStatement(nestedOp, ss);
    }
    ss.closeBrace();
}

void convertControl(P4HIR::ControlOp &controlOp, FormattedStream &ss) {
    ss << "control " << controlOp.getName().str() << "(";
    convertArgumentList(controlOp, ss);
    ss << ")";
    ss.openBrace();
    for (auto &nestedOp : controlOp.getOps()) {
        convertStatement(nestedOp, ss);
    }
    ss.closeBrace();
}

void convertStatement(mlir::Operation &op, FormattedStream &ss) {
    // llvm::outs() << "#### " << op->getName() << " ####\n";
    // llvm::outs() << op->getLoc() << "\n";

    if (auto funcOp = llvm::dyn_cast<P4HIR::FuncOp>(op)) {
        // TODO: Implement FuncOp conversion.
    } else if (auto packageOp = llvm::dyn_cast<P4HIR::PackageOp>(op)) {
        // TODO: Implement PackageOp conversion.
    } else if (auto assignOp = llvm::dyn_cast<P4HIR::AssignOp>(op)) {
        referenceToP4String(*assignOp.getRef().getDefiningOp(), ss);
        ss << " = ";
        convertExpression(*assignOp.getValue().getDefiningOp(), ss);
        ss.semicolon();
    } else if (auto controlApplyOp = llvm::dyn_cast<P4HIR::ControlApplyOp>(op)) {
        convertControlApply(controlApplyOp, ss);
    } else if (auto parserOp = llvm::dyn_cast<P4HIR::ParserOp>(op)) {
        convertParser(parserOp, ss);
    } else if (auto controlOp = llvm::dyn_cast<P4HIR::ControlOp>(op)) {
        convertControl(controlOp, ss);
    } else if (auto instantiateOp = llvm::dyn_cast<P4HIR::InstantiateOp>(op)) {
        // TODO: Implement InstantiateOp conversion.
    } else if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        llvm::errs() << "Not sure what to do with nested module op yet" << "\n";
    } else {
        llvm::errs() << "Unhandled operation: " << op << "\n";
        // llvm::report_fatal_error("Encountered unhandled operation.");
    }
}

}  // namespace

std::string MlirToP4::convert() {
    FormattedStream ss;
    module.walk([&](mlir::Operation *op) {
        if (auto subModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
            llvm::outs() << "Converting module: " << subModule->getName() << "\n";
            for (auto &op : subModule.getOps()) {
                llvm::outs() << "Converting operation: " << op.getName() << "\n";
                // if (op.hasTrait<mlir::OpTrait::ZeroResults>()) {
                convertStatement(op, ss);
                // }
            }
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return ss.str();
}

}  // namespace P4::P4MLIR
