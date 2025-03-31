#ifndef LIB_UTILITIES_FORMATTED_STREAM_H_
#define LIB_UTILITIES_FORMATTED_STREAM_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace P4::P4MLIR::Utilities {

class FormattedStream {
 private:
    std::ostream *out_stream_ = nullptr;
    /// If no external stream is provided, we default to this internal stringstream.
    std::unique_ptr<std::stringstream> internal_ss_owner_ = nullptr;
    std::stringstream *internal_ss_ptr_ = nullptr;

    /// Manage indentation state and width.
    int indent_level_ = 0;
    std::string indent_string_ = "    ";
    bool needs_indent_ = true;

    void apply_indent() {
        if (needs_indent_ && out_stream_) {
            for (int i = 0; i < indent_level_; ++i) {
                *out_stream_ << indent_string_;
            }
            // Check stream state after potentially multiple writes
            if (!out_stream_->good()) {
                throw std::runtime_error(
                    "FormattedStream: Output stream is not good after writing indentation.");
            }
            needs_indent_ = false;
        }
    }

    /// Helper to check stream state after operations
    void check_stream_state() const {
        if (out_stream_ && !out_stream_->good()) {
            throw std::runtime_error("FormattedStream: Output stream is not good.");
        }
    }

 public:
    /// Default: Use internal stringstream buffer
    explicit FormattedStream(const std::string &indent_unit = "    ")
        : indent_string_(indent_unit) {
        internal_ss_owner_ = std::make_unique<std::stringstream>();
        internal_ss_ptr_ = internal_ss_owner_.get();
        out_stream_ = internal_ss_ptr_;
        if (!out_stream_) {
            throw std::runtime_error("FormattedStream: Failed to create internal stringstream.");
        }
    }

    /// Use an external ostream (e.g., std::cout, std::ofstream)
    explicit FormattedStream(std::ostream &external_stream, const std::string &indent_unit = "    ")
        : out_stream_(&external_stream), indent_string_(indent_unit) {
        if (!out_stream_) {
            throw std::runtime_error("FormattedStream: External stream provided is invalid.");
        }
        if (!out_stream_->good()) {
            throw std::runtime_error(
                "FormattedStream: External stream provided is already in a bad state.");
        }
    }

    ~FormattedStream() {
        if (out_stream_ && !internal_ss_ptr_) {
            // Flush, if using an external stream.
            out_stream_->flush();
        }
    }

    FormattedStream(const FormattedStream &) = delete;
    FormattedStream &operator=(const FormattedStream &) = delete;

    FormattedStream(FormattedStream &&other) noexcept
        : out_stream_(other.out_stream_),
          internal_ss_owner_(std::move(other.internal_ss_owner_)),
          internal_ss_ptr_(other.internal_ss_ptr_),
          indent_level_(other.indent_level_),
          indent_string_(std::move(other.indent_string_)),
          needs_indent_(other.needs_indent_) {
        // Null out the moved-from object's pointers to prevent double-deletion
        // or dangling pointers. Note that out_stream_ is intentionally *not*
        // nulled if it pointed to an *external* stream, as the external
        // stream's lifetime is not managed here. But if it pointed to the
        // internal stream, it needs updating.
        if (other.internal_ss_ptr_ == other.out_stream_) {
            other.out_stream_ = nullptr;
        }
        other.internal_ss_ptr_ = nullptr;
        // Other members like indent_level_, needs_indent_ are fundamental types
        // or std::string which handles its own move state.
    }

    FormattedStream &operator=(FormattedStream &&other) noexcept {
        if (this != &other) {
            // Move assign members
            internal_ss_owner_ = std::move(other.internal_ss_owner_);
            internal_ss_ptr_ = other.internal_ss_ptr_;
            out_stream_ = other.out_stream_;

            indent_level_ = other.indent_level_;
            indent_string_ = std::move(other.indent_string_);
            needs_indent_ = other.needs_indent_;

            // Null out the moved-from object's pointers as in move constructor
            if (other.internal_ss_ptr_ == other.out_stream_) {
                other.out_stream_ = nullptr;
            }
            other.internal_ss_ptr_ = nullptr;
        }
        return *this;
    }

    FormattedStream &indent() {
        ++indent_level_;
        return *this;
    }
    FormattedStream &dedent() {
        if (indent_level_ > 0) --indent_level_;
        return *this;
    }
    void set_indent_string(const std::string &indent_unit) { indent_string_ = indent_unit; }
    int get_indent_level() const { return indent_level_; }

    template <typename T>
    FormattedStream &operator<<(const T &value) {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        apply_indent();
        *out_stream_ << value;
        check_stream_state();
        return *this;
    }

    /// Special handling for std::newline manipulator (flushes the stream)
    /// We need to capture the function pointer type
    using Manipulator = std::ostream &(*)(std::ostream &);
    FormattedStream &operator<<(Manipulator manip) {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        apply_indent();
        // Apply the manipulator (e.g., std::newline) to the target stream
        manip(*out_stream_);
        if (manip == static_cast<Manipulator>(std::endl)) {
            needs_indent_ = true;
        } else {
            needs_indent_ = false;
            // Let's stick to explicit newline() method for guaranteed indent trigger.
        }
        check_stream_state();
        return *this;
    }

    FormattedStream &newline() {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        *out_stream_ << '\n';
        needs_indent_ = true;
        // We don't automatically flush here; std::newline used via << would flush.
        check_stream_state();
        return *this;
    }

    FormattedStream &openBrace(bool newline_after = true) {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        apply_indent();
        *out_stream_ << '{';
        check_stream_state();
        if (newline_after) {
            newline();
        }
        indent();
        return *this;
    }

    FormattedStream &closeBrace(bool newline_after = true, bool indent_before = true) {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        dedent();
        if (indent_before) {
            apply_indent();
        } else {
            needs_indent_ = false;
        }
        *out_stream_ << '}';
        check_stream_state();
        if (newline_after) {
            newline();
        }
        return *this;
    }

    FormattedStream &semicolon(bool newline_after = true) {
        if (!out_stream_)
            throw std::runtime_error("FormattedStream: Output stream is not initialized.");
        apply_indent();
        *out_stream_ << ';';
        check_stream_state();
        if (newline_after) {
            newline();
        }
        return *this;
    }

    /// Only works if using the internal stringstream buffer.
    std::string str() const {
        if (internal_ss_ptr_) {
            return internal_ss_ptr_->str();
        } else {
            throw std::logic_error(
                "FormattedStream::str() called but output was directed to an external ostream.");
        }
    }
};

}  // namespace P4::P4MLIR::Utilities

#endif  // LIB_UTILITIES_FORMATTED_STREAM_H_
