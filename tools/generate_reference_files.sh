#!/bin/bash

# Example usage: ./generate_reference_files.sh -c "../install/bin/p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4" -d ../test/Translate/


# --- Default Values ---
DEFAULT_EXTENSION=".p4"
DEFAULT_COMMAND="p4c"
# Filename (used by -o) is now ONLY relevant for a potential future combined mode.
# For now, -o is effectively ignored. Let's keep a default for consistency.
DEFAULT_OUTPUT_FILENAME="p4.ref"
# Directory to start searching (recursively)
DEFAULT_SEARCH_DIR="."
# Global output directory (if set, mirrors structure here)
GLOBAL_OUTPUT_DIR=""
# Verbose flag
VERBOSE=false
# Suffix for reference files in both modes
DEFAULT_REF_SUFFIX=".ref" # Not currently configurable via flag

usage() {
    echo "Usage: $0 [-e EXT] [-c 'CMD args'] [-d SEARCH_DIR] [-O OUT_DIR] [-v] [-h]"
    echo ""
    echo "Recursively finds files with EXTENSION in SEARCH_DIR and runs COMMAND on them,"
    echo "generating an individual reference file for each source file."
    echo ""
    echo "Output Modes:"
    echo "  1. Default (No -O): For each 'path/file.EXT', creates 'path/file.EXT${DEFAULT_REF_SUFFIX}'"
    echo "                      in the *same* directory ('path/')."
    echo "  2. Global Dir (-O): For each 'path/file.EXT' found within SEARCH_DIR,"
    echo "                      creates 'OUT_DIR/path/file.EXT${DEFAULT_REF_SUFFIX}',"
    echo "                      mirroring the directory structure inside OUT_DIR."
    echo ""
    echo "Options:"
    echo "  -e EXT          File extension to process (default: ${DEFAULT_EXTENSION})"
    echo "  -c 'CMD args'   Command to run (quote if args/spaces). Default: '${DEFAULT_COMMAND}'"
    echo "  -d SEARCH_DIR   Top-level directory to search recursively (default: ${DEFAULT_SEARCH_DIR})"
    echo "  -O OUT_DIR      Enable Global Dir Mode: Specify directory to mirror structure and place"
    echo "                  individual reference files."
    echo "  -o OUT_FILE     (Ignored in current version) Filename for combined output."
    echo "  -v              Enable verbose output."
    echo "  -h              Display this help message."
    # Note: -o FILENAME is intentionally left out of the main usage line as it's ignored.
    exit 1
}

# --- Parse Command-Line Options ---
EXTENSION=$DEFAULT_EXTENSION
COMMAND=$DEFAULT_COMMAND
OUTPUT_FILENAME=$DEFAULT_OUTPUT_FILENAME # Kept for potential future use, but ignored now
SEARCH_DIR=$DEFAULT_SEARCH_DIR

# Reset GLOBAL_OUTPUT_DIR at start
GLOBAL_OUTPUT_DIR=""

while getopts ":e:c:d:O:o:vh" opt; do
    case ${opt} in
        e ) EXTENSION="$OPTARG" ;;
        c ) COMMAND="$OPTARG" ;;
        d ) SEARCH_DIR="$OPTARG" ;;
        O ) GLOBAL_OUTPUT_DIR="$OPTARG" ;;
        o ) OUTPUT_FILENAME="$OPTARG" ;; # Parse it, but note it's ignored
        v ) VERBOSE=true ;;
        h ) usage ;;
        \? ) echo "Invalid option: -$OPTARG" >&2; usage ;;
        : ) echo "Invalid option: -$OPTARG requires an argument" >&2; usage ;;
    esac
done
shift $((OPTIND -1)) # Remove parsed options

# --- Basic Input Validation ---
# Check base command existence
BASE_COMMAND=$(echo $COMMAND | awk '{print $1;}')
if ! command -v "$BASE_COMMAND" &> /dev/null; then
    echo "Error: Base command '$BASE_COMMAND' from '$COMMAND' not found or not executable." >&2
    exit 1
fi

# Check search directory existence
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Search directory '$SEARCH_DIR' not found." >&2
    exit 1
fi

# Validate Global Output Directory Mode settings
if [ -n "$GLOBAL_OUTPUT_DIR" ]; then
    # Global mode is active
    if [ ! -d "$GLOBAL_OUTPUT_DIR" ]; then
        # Attempt to create it? Or error out? Let's error out for safety.
        # Use mkdir -p "$GLOBAL_OUTPUT_DIR" if auto-creation is desired.
        echo "Error: Global output directory '$GLOBAL_OUTPUT_DIR' (-O) not found." >&2
        echo "Please create the directory first." >&2
        exit 1
    fi
    if ! touch "$GLOBAL_OUTPUT_DIR/.tmp_write_test" 2>/dev/null; then
        echo "Error: Global output directory '$GLOBAL_OUTPUT_DIR' (-O) is not writable." >&2
        exit 1
    fi
    rm "$GLOBAL_OUTPUT_DIR/.tmp_write_test"
    if $VERBOSE; then
        echo "--- Global Output Mode (-O) Activated ---"
        echo "Output directory    : '$GLOBAL_OUTPUT_DIR'"
        echo "Structure mirroring : Enabled"
        echo "Output suffix       : '${DEFAULT_REF_SUFFIX}'"
        echo "Option '-o' ('$OUTPUT_FILENAME') is ignored."
        echo "-----------------------------------------"
    fi
else
    if $VERBOSE; then
        echo "--- Default (Per-File) Output Mode Activated ---"
        echo "Output suffix       : '${DEFAULT_REF_SUFFIX}' (next to source files)"
        echo "------------------------------------------------"
    fi
fi

# --- Main Processing Logic ---
if $VERBOSE; then
    echo "Searching recursively in: '$SEARCH_DIR'"
    echo "For files matching    : '*$EXTENSION'"
    echo "Running command       : '$COMMAND'"
    echo "------------------------------------"
fi

processed_count=0

# Use find to locate files recursively
# -printf '%p\0' gives the full path null-terminated
# -printf '%P\0' gives the path relative to SEARCH_DIR, null-terminated
# We read both using the while loop
find "$SEARCH_DIR" -type f -name "*$EXTENSION" -printf '%p\0%P\0' | while IFS= read -r -d $'\0' file && IFS= read -r -d $'\0' rel_file; do
    ((processed_count++))
    SOURCE_DIR=$(dirname "$file") # Directory containing the source file

    TARGET_REF_FILE="" # Define target file path variable

    if [ -n "$GLOBAL_OUTPUT_DIR" ]; then
        # === Global Output Mode (-O) ===
        # Target file mirrors structure under GLOBAL_OUTPUT_DIR
        TARGET_REF_FILE="$GLOBAL_OUTPUT_DIR/$rel_file$DEFAULT_REF_SUFFIX"
        TARGET_REF_DIR=$(dirname "$TARGET_REF_FILE")

        # Ensure the target directory exists within the global output dir
        if ! mkdir -p "$TARGET_REF_DIR"; then
             echo "Warning: Failed to create target directory '$TARGET_REF_DIR'. Skipping '$file'." >&2
             continue # Skip to the next file
        fi
        if $VERBOSE; then
            echo "Processing '$file' -> Generating '$TARGET_REF_FILE'"
        fi
        # Run command and OVERWRITE (>) stdout/stderr to the specific target .ref file
        $COMMAND "$file" > "$TARGET_REF_FILE" 2>&1

    else
        # === Default (Per-File) Output Mode ===
        # Target file is next to the source file
        TARGET_REF_FILE="${file}${DEFAULT_REF_SUFFIX}"

        # Check if source directory is writable for the output file
        # Use the actual target filename for the write test for more accuracy
        if ! touch "$TARGET_REF_FILE.tmp" 2>/dev/null; then
             # Try one level up if dirname failed (less likely but possible)
             if ! touch "$SOURCE_DIR/.tmp_write_test" 2>/dev/null; then
                 echo "Warning: Cannot write to output location for '$file' (tried '$TARGET_REF_FILE' and dir '$SOURCE_DIR'). Skipping." >&2
                 # Clean up the test file if the first touch somehow worked but second failed
                 rm -f "$TARGET_REF_FILE.tmp"
                 continue # Skip to the next file
             else
                 rm "$SOURCE_DIR/.tmp_write_test" # Clean up successful dir test file
             fi
        else
             rm "$TARGET_REF_FILE.tmp" # Clean up successful file test file
        fi


        if $VERBOSE; then
            echo "Processing '$file' -> Generating '$TARGET_REF_FILE'"
        fi
        # Run command and OVERWRITE (>) stdout/stderr to the specific .ref file
        $COMMAND "$file" > "$TARGET_REF_FILE" 2>&1
    fi

    # Optional: Check command exit status after redirection
    if [ $? -ne 0 ] && $VERBOSE; then
        echo "Warning: Command exited with non-zero status for '$file' -> '$TARGET_REF_FILE'" >&2
    fi

done

# --- Completion Message ---
echo "------------------------------------"
if [ "$processed_count" -eq 0 ]; then
    echo "Warning: No files found matching '*$EXTENSION' in '$SEARCH_DIR' or its subdirectories."
else
    echo "Processing complete. $processed_count file(s) processed."
fi

if [ -n "$GLOBAL_OUTPUT_DIR" ]; then
    echo "Individual reference files generated with suffix '${DEFAULT_REF_SUFFIX}'"
    echo "within directory '$GLOBAL_OUTPUT_DIR', mirroring source structure."
else
    echo "Individual reference files generated with suffix '${DEFAULT_REF_SUFFIX}' next to source files."
fi

exit 0
