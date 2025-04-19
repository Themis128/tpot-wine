#!/bin/bash

REQUIREMENTS_FILE="requirements.txt"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Not inside a virtual environment. Please activate it first!"
    exit 1
fi

# Remove old requirements.txt if it exists
if [[ -f $REQUIREMENTS_FILE ]]; then
    echo "Removing old $REQUIREMENTS_FILE..."
    rm "$REQUIREMENTS_FILE"
fi

# Generate new one
echo "Generating new $REQUIREMENTS_FILE..."
pip freeze > "$REQUIREMENTS_FILE"

echo "✅ Done. Requirements saved to $REQUIREMENTS_FILE"
