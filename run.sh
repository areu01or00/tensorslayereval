#!/bin/bash

# Activate virtual environment and run the inference engine

# Change to the root project directory
cd "/media/x/Seagate Expansion Drive/Current projects/28_Refine"

# Activate the virtual environment
source venv/bin/activate

# Change to inference_standalone directory
cd inference_standalone

# Check if model path is provided
if [ -z "$1" ]; then
    # Use default Qwen model if no argument provided
    MODEL_PATH="Qwen_0.6B"
else
    MODEL_PATH="$1"
fi

echo "Starting Inference Engine with model: $MODEL_PATH"
echo "Press Ctrl+Q to quit the application"
echo ""

# Run the inference engine
python3 inference_engine.py "$MODEL_PATH"