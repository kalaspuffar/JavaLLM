#!/bin/bash
# Downloads the Tiny Shakespeare dataset (~1MB, public domain).
# This file is used by CharGenerationTest for character-level language modeling.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/shakespeare.txt"

if [ -f "$OUTPUT" ]; then
    echo "shakespeare.txt already exists at $OUTPUT"
    exit 0
fi

echo "Downloading Tiny Shakespeare..."
curl -fsSL -o "$OUTPUT" \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if [ $? -eq 0 ]; then
    echo "Downloaded to $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
else
    echo "Download failed. Please download manually from:"
    echo "  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    echo "and save to $OUTPUT"
    exit 1
fi
