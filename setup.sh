#!/bin/bash

# Setup script for Gemma-3 Deep Architecture Study

echo "╔══════════════════════════════════════╗"
echo "║   Gemma-3 Environment Setup          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p models data logs results visualizations

# Check Python version
echo ""
echo "Python version:"
python --version

# Check if MPS is available
echo ""
echo "Checking Metal Performance Shaders availability..."
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"

# Check MLX availability
echo ""
echo "Checking MLX availability..."
python -c "
try:
    import mlx
    print('MLX Available: Yes')
except ImportError:
    print('MLX Available: No (optional - install with: pip install mlx mlx-lm)')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download Gemma-3 model:"
echo "   python scripts/download_model.py --model gemma-3-4b"
echo ""
echo "2. Test inference:"
echo "   python scripts/test_inference.py"
echo ""
echo "3. Run experiments:"
echo "   python run_experiments.py"