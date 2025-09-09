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
    echo "Model not found. To use real model data, run: make download"
    echo "Serving previously generated visualization if available..."
    python - <<'PY'
import webbrowser
import os
p = os.path.abspath('visualizations/architecture_plot.html')
if os.path.exists(p):
    webbrowser.open('file://' + p)
    print('Opened existing visualization at', p)
else:
    print('No visualization found in visualizations/.')
PY
fi