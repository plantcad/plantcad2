#!/bin/bash

# Downsample TSV file: keep all label=1, sample equal number of label=0
# Usage: bash downsample.sh input.tsv output.tsv

if [ "$#" -ne 2 ]; then
    echo "Usage: bash $0 <input.tsv> <output.tsv>"
    echo "Example: bash $0 chr01.tsv chr01_downsampled.tsv"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
TEMP_DIR="./temp_downsample_$$"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Create temporary directory
mkdir -p "$TEMP_DIR"

echo "Processing $INPUT_FILE..."

# Extract header
head -1 "$INPUT_FILE" > "$OUTPUT_FILE"

# Extract all label=1 entries (column 4)
echo "Extracting all label=1 entries..."
awk -F'\t' 'NR>1 && $4==1' "$INPUT_FILE" > "$TEMP_DIR/label1.tsv"

# Count label=1 entries
LABEL1_COUNT=$(wc -l < "$TEMP_DIR/label1.tsv")
echo "Found $LABEL1_COUNT label=1 entries"

# Extract and randomly sample label=0 entries
echo "Sampling $LABEL1_COUNT label=0 entries..."
awk -F'\t' 'NR>1 && $4==0' "$INPUT_FILE" | shuf -n "$LABEL1_COUNT" > "$TEMP_DIR/label0_sampled.tsv"

# Combine and append to output
echo "Combining results..."
cat "$TEMP_DIR/label1.tsv" "$TEMP_DIR/label0_sampled.tsv" >> "$OUTPUT_FILE"

# Clean up
rm -rf "$TEMP_DIR"

echo "Done! Output saved to $OUTPUT_FILE"
echo "Total entries (excluding header): $((LABEL1_COUNT * 2))"
