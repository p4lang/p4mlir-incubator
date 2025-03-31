#ifndef LIB_UTILITIES_EXTENDED_FORMATTED_STREAM_H_
#define LIB_UTILITIES_EXTENDED_FORMATTED_STREAM_H_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop

namespace P4::P4MLIR::Utilities {

class ExtendedFormattedOStream {
 public:
    enum class IndentationStyle { Spaces, Tabs };

 private:
    llvm::formatted_raw_ostream OS;
    int indentLevel = 0;
    IndentationStyle Style;
    /// Width for Spaces, usually 1 for Tabs
    unsigned indentWidth;
    /// Start of a new line requires indentation
    bool needsIndent = true;

    // Helper to write the actual indentation characters
    void writeIndentation() {
        if (!needsIndent) return;

        if (indentLevel > 0) {
            if (Style == IndentationStyle::Spaces) {
                // formatted_raw_ostream::indent expects number of spaces
                OS.indent(indentLevel * indentWidth);
            } else {
                // raw_ostream doesn't have a built-in tab indent repeat, do it manually
                for (int i = 0; i < indentLevel; ++i) {
                    OS << '\t';  // Assumes indentWidth is implicitly 1 for tabs
                }
            }
        }
        needsIndent = false;
    }

 public:
    ExtendedFormattedOStream(llvm::raw_ostream &os,
                             IndentationStyle style = IndentationStyle::Spaces,
                             unsigned indentWidth = 4)
        : OS(os), Style(style), indentWidth(indentWidth) {
        // Ensure sensible width for tabs
        if (Style == IndentationStyle::Tabs) {
            indentWidth = 1;
        }
    }

    /// @brief Increases the indentation level.
    void indent() { indentLevel++; }

    /// @brief Decreases the indentation level. Clamps at 0.
    void unindent() {
        if (indentLevel > 0) {
            indentLevel--;
        } else {
            llvm::errs() << "Warning: Attempting to unindent below level 0.\n";
        }
    }

    /// @brief Gets the current indentation level.
    int getIndentLevel() const { return indentLevel; }

    /// @brief Prints a newline character and flags that the next output needs indentation.
    void newline() {
        OS << '\n';
        needsIndent = true;
    }

    /// @brief Handles indentation, prints '{', increases indent level, and optionally adds a
    /// newline after.
    /// @param addNewlineAfter If true, calls newline() after printing the brace.
    void openBrace(bool addNewlineAfter = true) {
        writeIndentation();  // Indent the brace itself if needed
        OS << '{';
        indent();
        if (addNewlineAfter) {
            newline();
        } else {
            // If no newline, subsequent output on the same line doesn't need *new* indent
            needsIndent = false;
        }
    }

    /// @brief Decreases indent level, optionally adds newline before, handles indentation, prints
    /// '}', and optionally adds newline after.
    /// @param addNewlineBefore If true, calls newline() before printing the brace.
    /// @param addNewlineAfter If true, calls newline() after printing the brace.
    void closeBrace(bool addNewlineBefore = false, bool addNewlineAfter = true) {
        unindent();  // Decrease level first
        if (addNewlineBefore) {
            newline();
        }
        writeIndentation();  // Indent the brace according to the new level
        OS << '}';
        if (addNewlineAfter) {
            newline();
        } else {
            needsIndent = false;
        }
    }

    /// @brief Prints ';', and optionally adds a newline after.
    /// @param addNewlineAfter If true, calls newline() after printing the semicolon.
    void semicolon(bool addNewlineAfter = true) {
        // Semicolon usually follows other content on the same line,
        // so indentation should have already been handled.
        // writeIndentation(); // Usually not needed here
        OS << ';';
        if (addNewlineAfter) {
            newline();
        } else {
            needsIndent = false;
        }
    }

    /// @brief Overload stream operator to handle indentation correctly, including embedded
    /// newlines.
    template <typename T>
    ExtendedFormattedOStream &operator<<(const T &Val) {
        // Convert value to string to handle potential newlines within it
        std::string Str;
        llvm::raw_string_ostream SS(Str);
        SS << Val;
        SS.flush();  // Ensure Str has the content

        llvm::StringRef Data(Str);
        size_t Start = 0;
        size_t Pos = 0;

        while ((Pos = Data.find('\n', Start)) != llvm::StringRef::npos) {
            // Print the part before the newline
            if (Pos > Start) {
                writeIndentation();  // Ensure indent before printing segment
                OS << Data.substr(Start, Pos - Start);
            } else if (needsIndent) {
                // Handle case where the line is empty before the newline,
                // but indentation is still needed (e.g., "{\n}")
                writeIndentation();
            }
            // Print the newline itself
            OS << '\n';
            needsIndent = true;  // Next segment needs indentation
            Start = Pos + 1;     // Move past the newline
        }

        // Print the remaining part of the string (or the whole string if no newline)
        if (Start < Data.size()) {
            writeIndentation();  // Ensure indent before printing final segment
            OS << Data.substr(Start);
            // We just printed non-newline chars, so next direct print
            // on this line doesn't need indent unless newline() is called.
            needsIndent = false;
        } else if (Data.ends_with("\n")) {
            // If the original string ended with \n, the loop handled it
            // and needsIndent is already true. Do nothing extra.
        } else if (!Data.empty() && needsIndent) {
            // Handle case where input was empty string "" after potential newlines
            // or if the *entire* input string was empty to begin with.
            // If needsIndent was true, we might need to print it now if the
            // last action was a newline() followed by << "". We writeIndent
            // defensively. It has an internal check for needsIndent anyway.
            writeIndentation();
        }

        return *this;
    }

    // Specialization for C-style strings.
    ExtendedFormattedOStream &operator<<(const char *Str) {
        // Reuse the templated version with StringRef for efficiency.
        return this->operator<<(llvm::StringRef(Str ? Str : ""));
    }

    // Allow explicit flushing of the underlying stream.
    void flush() { OS.flush(); }
};

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_EXTENDED_FORMATTED_STREAM_H_
