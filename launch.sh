#!/bin/bash

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/ubuntu/.local/bin:$PATH"

poetry install

poetry shell 

# Create directory if it doesn't exist
mkdir -p ../models/text

# Set working directory
cd ../models/text

# Define variables
MODEL_NAME="Rombos-LLM-V2.5-Qwen-32b.Q8_0.gguf"
BASE_URL="https://huggingface.co/mradermacher/Rombos-LLM-V2.5-Qwen-32b-GGUF/resolve/main"

# Download model
echo "Downloading ${MODEL_NAME}..."
wget "${BASE_URL}/${MODEL_NAME}" -O "${MODEL_NAME}"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download complete: ${MODEL_NAME}"
else
    echo "Error downloading model"
    exit 1
fi