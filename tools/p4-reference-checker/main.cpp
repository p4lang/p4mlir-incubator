#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Make sure the Config struct is defined before this function
struct Config {
    std::string input_path = "-";  // Default to stdin
    std::string reference_path;
    std::string compiler_bin;
    std::vector<std::string> compiler_args;  // Arguments for the compiler
    bool update_mode = false;
};
// --- Helper Functions ---

// Reads entire content from an input stream (file or stdin)
std::string readStreamContent(std::istream &stream) {
    if (!stream) {
        throw std::runtime_error("Input stream is invalid.");
    }
    std::ostringstream ss;
    ss << stream.rdbuf();
    if (!stream && !stream.eof()) {
        throw std::runtime_error("Error reading from input stream.");
    }
    return ss.str();
}

// Reads entire content from a file path
std::string readFileContent(const std::string &path) {
    if (path == "-") {
        // Reading from stdin
        return readStreamContent(std::cin);
    } else {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open input file: " + path);
        }
        return readStreamContent(file);
    }
}

// Writes content to a file path
void writeFileContent(const std::string &path, const std::string &content) {
    // Ensure parent directory exists
    std::filesystem::path file_path(path);
    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(
            file_path.parent_path());  // Create dirs if they don't exist
    }

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open output file for writing: " + path);
    }
    file << content;
    if (!file) {
        throw std::runtime_error("Error writing to output file: " + path);
    }
}

// Creates a temporary file, writes content to it, and returns its path
// Uses RAII to ensure cleanup
class TemporaryFile {
    std::filesystem::path path_;

 public:
    // Using Chrono + Random for portable unique name generation
    explicit TemporaryFile(const std::string &content,
                           const std::string &prefix = "checker_temp_") {
        std::error_code ec;
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path(ec);
        if (ec) {
            throw std::runtime_error("Could not get temporary directory path: " + ec.message());
        }

        // Use chrono + random for uniqueness
        std::string timestamp =
            std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> distrib(0, std::numeric_limits<uint64_t>::max());
        std::string random_part = std::to_string(distrib(gen));

        std::string filename = prefix + timestamp + "_" + random_part;
        path_ = temp_dir / filename;

        // Collision check (unlikely but robust)
        int retries = 5;
        while (std::filesystem::exists(path_) && retries > 0) {
            std::cerr << "Warning: Temporary file path collision, retrying: " << path_.string()
                      << std::endl;
            timestamp = std::to_string(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            random_part = std::to_string(distrib(gen));  // Regenerate random part
            filename = prefix + timestamp + "_" + random_part;
            path_ = temp_dir / filename;
            --retries;
        }
        if (std::filesystem::exists(path_)) {
            throw std::runtime_error(
                "Failed to create a unique temporary file name after retries: " + path_.string());
        }

        // std::cerr << "Debug: Creating temporary file: " << path_.string() << std::endl; //
        // Optional Debug
        writeFileContent(path_.string(), content);
    }

    ~TemporaryFile() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
            if (ec) {
                std::cerr << "Warning: Failed to remove temporary file '" << path_.string()
                          << "': " << ec.message() << std::endl;
            }
        }
    }

    // Delete copy constructor and assignment operator
    TemporaryFile(const TemporaryFile &) = delete;
    TemporaryFile &operator=(const TemporaryFile &) = delete;

    // Allow move constructor and assignment
    TemporaryFile(TemporaryFile &&other) noexcept : path_(std::move(other.path_)) {
        other.path_ = "";  // Prevent double deletion
    }
    TemporaryFile &operator=(TemporaryFile &&other) noexcept {
        if (this != &other) {
            if (!path_.empty()) {
                std::error_code ec;
                std::filesystem::remove(path_, ec);
            }
            path_ = std::move(other.path_);
            other.path_ = "";
        }
        return *this;
    }

    const std::filesystem::path &path() const { return path_; }
    const std::string string() const { return path_.string(); }
};

// Runs a command and returns its exit status
int runCommand(const std::string &command) {
    std::cerr << "Executing: " << command << std::endl;
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: Failed to execute command using popen." << std::endl;
        return -1;  // Indicate failure to execute
    }

    // Read command's stdout/stderr to avoid blocking pipes (optional but good practice)
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::cerr << "[Compiler Output] " << buffer;  // Echo compiler output to checker's stderr
    }

    int status = pclose(pipe);

    if (status == -1) {
        std::cerr << "Error: pclose failed." << std::endl;
        return -1;  // Indicate pclose failure
    } else {
        // Check how the command exited (normal vs signal)
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);  // Return the actual exit code
        } else if (WIFSIGNALED(status)) {
            std::cerr << "Error: Command terminated by signal " << WTERMSIG(status) << "."
                      << std::endl;
            return -1;  // Indicate abnormal termination
        } else {
            std::cerr << "Error: Command terminated abnormally (unknown reason)." << std::endl;
            return -1;  // Indicate abnormal termination
        }
    }
}

// Runs diff between two files
void runDiff(const std::string &file1, const std::string &file2) {
    // Using system for diff as its output is the desired result.
    // Ensure paths are quoted.
    std::string command = "diff -u ";
    command += "'" + file1 + "' ";  // Basic quoting
    command += "'" + file2 + "'";
    std::cerr << "Executing: " << command << std::endl;

    int status = std::system(command.c_str());  // Output goes directly to stdout/stderr

    // Check diff's exit status for potential issues
    if (status == 0) {
        std::cerr << "Warning: diff command reported files are identical unexpectedly."
                  << std::endl;
    } else if (WIFEXITED(status) && WEXITSTATUS(status) > 1) {
        std::cerr << "Warning: diff command exited with error status " << WEXITSTATUS(status)
                  << std::endl;
    } else if (!WIFEXITED(status)) {
        std::cerr << "Warning: diff command terminated abnormally." << std::endl;
    }
    // Exit code 1 from diff means files differ, which is expected when this function is called.
}

// --- Argument Parsing ---
Config parseArgs(int argc, char *argv[]) {
    Config config;
    // Create a vector of strings from C-style args, skipping the program name (argv[0])
    std::vector<std::string> args(argv + 1, argv + argc);
    bool input_path_set = false;      // Track if the input path was explicitly set
    bool reference_path_set = false;  // Track if the reference path was explicitly set
    size_t i = 0;

    // --- Phase 1: Parse checker options and main positional arguments up to '--' ---
    while (i < args.size()) {
        const std::string &arg = args[i];

        // Check for the '--' argument separator first
        if (arg == "--") {
            i++;    // Consume the "--" itself
            break;  // Exit Phase 1, move to Phase 2 (compiler args)
        }

        // --- Handle Known Options ---
        if (arg == "--compiler-bin") {
            if (i + 1 < args.size()) {
                // Check next argument isn't another option or separator
                if (args[i + 1] == "--" || args[i + 1].rfind("-", 0) == 0) {
                    throw std::runtime_error("--compiler-bin requires a path argument, not '" +
                                             args[i + 1] + "'");
                }
                config.compiler_bin = args[++i];  // Consume the option's value
            } else {
                throw std::runtime_error("--compiler-bin requires a path argument");
            }
        } else if (arg == "--update") {
            config.update_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            // Print help message and exit successfully
            std::cerr << "Usage: " << argv[0]
                      << " [options] <input_path | -> <reference_path> [-- <compiler_args>...]"
                      << std::endl;
            std::cerr << "Options:" << std::endl;
            std::cerr << "  <input_path>        Path to the input file generated by the "
                         "translator, or '-' for stdin (default)."
                      << std::endl;
            std::cerr << "  <reference_path>    Path to the reference (golden) file (required)."
                      << std::endl;
            std::cerr << "  --compiler-bin <path> Path to the binary used to compile the "
                         "translator's output."
                      << std::endl;
            std::cerr
                << "                      If set, compilation is checked before comparison/update."
                << std::endl;
            std::cerr << "  --update            Update the reference file with the input content "
                         "instead of comparing."
                      << std::endl;
            std::cerr << "  -h, --help          Show this help message." << std::endl;
            std::cerr << "  --                  Signals that subsequent arguments should be passed "
                         "directly to the compiler-bin."
                      << std::endl;
            std::cerr << "\nWorkflow Example (check):\n";
            std::cerr << "  translator input.p4 | " << argv[0]
                      << " - --compiler-bin p4-compiler input.ref.p4 -- --opt1 -v\n";
            std::cerr << "\nWorkflow Example (update):\n";
            std::cerr << "  translator input.p4 | " << argv[0] << " - --update input.ref.p4\n";
            exit(EXIT_SUCCESS);
        }

        // --- Handle Positional Arguments OR Unknown Options ---
        // Check if it's NOT an option OR if it IS exactly "-"
        else if (arg.rfind("-", 0) != 0 || arg == "-") {
            // Treat as a positional argument
            if (!input_path_set) {
                // First positional argument is the input path (can be "-")
                config.input_path = arg;
                input_path_set = true;
            } else if (!reference_path_set) {
                // Second positional argument is the reference path
                config.reference_path = arg;
                reference_path_set = true;
            } else {
                // We already have input and reference paths, this is unexpected before "--"
                throw std::runtime_error("Unexpected positional argument before '--': " + arg);
            }
        }
        // --- Handle Unknown Options ---
        else {
            // It starts with '-' but wasn't matched as a known option
            throw std::runtime_error("Unknown option: " + arg);
        }

        // Move to the next argument for the next iteration of the loop
        i++;
    }  // --- End Phase 1 loop ---

    // --- Phase 2: Collect all remaining arguments after '--' as compiler arguments ---
    // The loop above breaks when '--' is encountered or all args are processed.
    // 'i' now points to the first argument *after* '--' (or is == args.size()).
    while (i < args.size()) {
        config.compiler_args.push_back(args[i]);
        i++;
    }

    // --- Final Validation Checks ---
    if (config.reference_path.empty()) {
        // This catches the case where the second required positional argument was missing
        throw std::runtime_error("Missing required argument: <reference_path>");
    }
    // Note: We don't strictly need to check input_path_set because it defaults to "-",
    // and if the user provided *something* else as the first positional arg,
    // input_path_set would be true. If they provided nothing, reference_path would
    // be empty and caught above.

    // If compiler arguments were provided via '--', ensure a compiler binary was also specified
    if (!config.compiler_args.empty() && config.compiler_bin.empty()) {
        throw std::runtime_error(
            "Compiler arguments provided after '--' but --compiler-bin was not specified.");
    }

    // Return the populated and validated configuration object
    return config;
}

// --- Main Logic ---
int main(int argc, char *argv[]) {
    Config config;
    try {
        config = parseArgs(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Use -h or --help for usage." << std::endl;
        return EXIT_FAILURE;
    }

    try {
        // 1. Read input content
        std::string input_content = readFileContent(config.input_path);

        // 2. Optional: Compile check
        if (!config.compiler_bin.empty()) {
            // Create temporary file RAII object only if needed
            TemporaryFile temp_input_file(input_content, "check_compile_");

            // Build the compile command string
            std::stringstream cmd_ss;
            cmd_ss << config.compiler_bin;  // The compiler binary itself

            // Add the extra compiler arguments (passed after --)
            for (const auto &arg : config.compiler_args) {
                // Basic quoting: wrap in single quotes. Handles many cases.
                // For complex args needing more robust shell escaping, a library or
                // direct use of execvp family might be better, but popen + quoting is simpler here.
                cmd_ss << " '" << arg << "'";
            }

            // Add the temporary input file path as the last argument (quoted)
            cmd_ss << " '" << temp_input_file.string() << "'";

            std::string compile_command = cmd_ss.str();

            int compile_status =
                runCommand(compile_command);  // runCommand handles execution and stderr logging

            if (compile_status != 0) {
                std::cerr << "Error: Compilation check FAILED with exit code " << compile_status
                          << "." << std::endl;
                // The command itself was already printed by runCommand
                // std::cerr << "Failing command: " << compile_command << std::endl; // Optional
                // repeat
                return EXIT_FAILURE;
            }
            std::cerr << "Compilation check successful." << std::endl;
            // temp_input_file is automatically cleaned up by RAII when it goes out of scope
        }  // End of compile check block

        // 3. Update mode or Compare mode
        if (config.update_mode) {
            // Update mode: Write input content to reference file
            std::cerr << "Update mode: Writing input to reference file '" << config.reference_path
                      << "'." << std::endl;
            writeFileContent(config.reference_path, input_content);
            std::cerr << "Reference file updated successfully." << std::endl;
        } else {
            // Compare mode: Read reference file and compare
            std::string reference_content;
            try {
                // Check if reference file exists before trying to read
                if (!std::filesystem::exists(config.reference_path)) {
                    std::cerr << "Error: Reference file '" << config.reference_path
                              << "' does not exist." << std::endl;
                    std::cerr << "Perhaps you need to run with --update first?" << std::endl;
                    return EXIT_FAILURE;
                }
                // Use ifstream directly to read reference file
                std::ifstream ref_stream(config.reference_path);
                if (!ref_stream) {
                    throw std::runtime_error("Cannot open reference file: " +
                                             config.reference_path);
                }
                reference_content = readStreamContent(ref_stream);
            } catch (const std::exception &e) {
                std::cerr << "Error reading reference file: " << e.what() << std::endl;
                return EXIT_FAILURE;
            }

            // Perform the comparison
            if (input_content == reference_content) {
                std::cerr << "Success: Input matches reference file '" << config.reference_path
                          << "'." << std::endl;
            } else {
                std::cerr << "Error: Input MISMATCHES reference file '" << config.reference_path
                          << "'." << std::endl;
                std::cerr << "Generating diff..." << std::endl;

                // Create temporary files for diff using RAII
                try {
                    TemporaryFile temp_ref_file(reference_content, "ref_");
                    TemporaryFile temp_input_file(
                        input_content, "input_");  // Can reuse if compile check didn't happen

                    // Run diff - output goes to stdout/stderr
                    runDiff(temp_ref_file.string(), temp_input_file.string());
                    // Temp files are cleaned up automatically by RAII
                } catch (const std::exception &e) {
                    std::cerr << "Error during diff generation: " << e.what() << std::endl;
                    // Fallback? Could print truncated content, but diff failure is often due to
                    // temp file issues.
                }

                return EXIT_FAILURE;  // Indicate mismatch
            }
        }  // End of compare mode block

    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
