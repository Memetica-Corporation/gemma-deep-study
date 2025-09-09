#!/bin/bash

echo "╔══════════════════════════════════════╗"
echo "║  Gemma-3 Architecture Visualizer     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check if model exists
if [ -f "models/gemma-3-4b/config.json" ]; then
    echo "Using actual model data..."
    source venv/bin/activate && python scripts/generate_architecture_viz.py --serve
else
    echo "Model not found. Using sample data..."
    echo "To use real model data, run: make download"
    echo ""
    source venv/bin/activate && python scripts/generate_sample_viz.py
fi