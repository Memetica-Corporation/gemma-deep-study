#!/bin/bash

# Quick activation script for Gemma-3 project

source venv/bin/activate
echo "✅ Virtual environment activated"
echo "📍 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""
echo "Quick commands:"
echo "  run_experiments.py  - Launch experiment menu"
echo "  test_inference.py   - Test model inference"
echo "  metal_benchmark.py  - Run Metal/MLX benchmarks"