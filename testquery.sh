#!/bin/bash

# Configuration
MODEL="fastapi-gen"
URL="http://localhost:11434/api/generate"

# Check if a prompt was provided
if [ -z "$1" ]; then
    echo "Usage: ./query_ollama.sh \"Your request here\""
    exit 1
fi

PROMPT=$1

echo "--- Generating FastAPI Code ---"

# Execute the request
curl -s -X POST "$URL" -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false
}" | jq -r '.response'