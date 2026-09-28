#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 PDF_DIR OUTPUT_DIR" >&2
    exit 2
fi
INPUT_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"

for pdf in "$INPUT_DIR"/*.pdf; do
    if [ -f "$pdf" ]; then
        echo "Processing: $pdf"
        mineru -p "$pdf" -o "$OUTPUT_DIR" --source local
    fi
done
