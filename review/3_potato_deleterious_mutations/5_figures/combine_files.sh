#!/bin/bash

# Script to combine multiple files with headers
# Usage: ./combine_files.sh output_file input_file1 input_file2 input_file3 ...
# The header from the first file will be kept, and headers from subsequent files will be skipped

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 output_file input_file1 [input_file2 ...]"
    echo "Example: $0 combined.tsv file1.tsv file2.tsv file3.tsv"
    exit 1
fi

OUTPUT_FILE=$1
shift  # Remove first argument, rest are input files

# Check if output file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Warning: $OUTPUT_FILE already exists. It will be overwritten."
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm "$OUTPUT_FILE"
fi

# Process files
FIRST_FILE=true
for FILE in "$@"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: File $FILE not found!"
        exit 1
    fi

    if [ "$FIRST_FILE" = true ]; then
        # For first file, copy everything including header
        cat "$FILE" > "$OUTPUT_FILE"
        echo "Added $FILE with header"
        FIRST_FILE=false
    else
        # For subsequent files, skip the first line (header)
        tail -n +2 "$FILE" >> "$OUTPUT_FILE"
        echo "Added $FILE (skipped header)"
    fi
done

# Count lines
TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "Done! Combined $(($# )) files into $OUTPUT_FILE"
echo "Total lines (including header): $TOTAL_LINES"
