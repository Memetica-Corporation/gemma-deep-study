# Makefile for Gemma-3 Deep Architecture Study

.PHONY: help setup install download visualize viz-gen test benchmark experiments clean

help:
	@echo "Gemma-3 Deep Architecture Study - Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup       - Complete environment setup"
	@echo "  make install     - Install dependencies only"
	@echo "  make download    - Download Gemma-3 4B model"
	@echo ""
	@echo "Visualization:"
	@echo "  make visualize   - Generate and serve architecture visualization"
	@echo "  make viz-gen     - Generate visualization files only"
	@echo ""
	@echo "Testing & Benchmarking:"
	@echo "  make test        - Run inference tests"
	@echo "  make benchmark   - Run Metal/MLX benchmarks"
	@echo ""
	@echo "Experiments:"
	@echo "  make experiments - Launch interactive experiment menu"
	@echo "  make rank1       - Run rank-1 LoRA experiment"
	@echo "  make relora      - Run ReLoRA experiment"
	@echo "  make capacity    - Run capacity analysis"
	@echo "  make attention   - Run attention analysis"
	@echo "  make sbd         - Run Set Block Decoding benchmark"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean cache and temporary files"
	@echo "  make check       - Check environment and dependencies"

setup:
	@echo "Setting up Gemma-3 environment..."
	@./setup.sh

install:
	@echo "Installing dependencies..."
	@source venv/bin/activate && pip install -r requirements.txt

download:
	@echo "Downloading Gemma-3 4B model..."
	@source venv/bin/activate && python scripts/download_model.py --model gemma-3-4b

visualize:
	@echo "Generating and serving architecture visualization..."
	@source venv/bin/activate && python scripts/generate_architecture_viz.py --serve

viz-gen:
	@echo "Generating architecture visualization files..."
	@source venv/bin/activate && python scripts/generate_architecture_viz.py

test:
	@echo "Running inference tests..."
	@source venv/bin/activate && python scripts/test_inference.py --benchmark

benchmark:
	@echo "Running Metal/MLX benchmarks..."
	@source venv/bin/activate && python scripts/metal_benchmark.py

experiments:
	@echo "Launching experiment menu..."
	@source venv/bin/activate && python run_experiments.py

rank1:
	@echo "Running rank-1 LoRA experiment..."
	@source venv/bin/activate && python run_experiments.py --experiment rank1

relora:
	@echo "Running ReLoRA experiment..."
	@source venv/bin/activate && python run_experiments.py --experiment relora

capacity:
	@echo "Running capacity analysis..."
	@source venv/bin/activate && python run_experiments.py --experiment capacity

attention:
	@echo "Running attention analysis..."
	@source venv/bin/activate && python run_experiments.py --experiment attention

sbd:
	@echo "Running Set Block Decoding benchmark..."
	@source venv/bin/activate && python run_experiments.py --experiment sbd

clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete
	@rm -rf .pytest_cache
	@rm -rf .ipynb_checkpoints
	@echo "✅ Cleaned successfully"

check:
	@echo "Checking environment..."
	@source venv/bin/activate && python -c "import torch; print('PyTorch:', torch.__version__)"
	@source venv/bin/activate && python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
	@source venv/bin/activate && python -c "import transformers; print('Transformers:', transformers.__version__)"
	@source venv/bin/activate && python -c "try: import mlx; print('MLX: Available'); except: print('MLX: Not installed (optional)')"
	@echo "✅ Environment check complete"