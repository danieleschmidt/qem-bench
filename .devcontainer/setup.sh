#!/bin/bash

# QEM-Bench Development Environment Setup Script

set -e

echo "🚀 Setting up QEM-Bench development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    graphviz \
    pandoc \
    texlive-latex-base \
    texlive-latex-extra

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
echo "📚 Installing QEM-Bench in development mode..."
pip install -e ".[dev,docs,full]"

# Install additional development tools
echo "🔧 Installing development tools..."
pip install \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pre-commit \
    nbstripout \
    jupytext

# Setup git hooks
echo "🪝 Setting up git hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Setup Jupyter extensions
echo "📊 Setting up Jupyter..."
jupyter lab build

# Create useful directories
echo "📁 Creating project directories..."
mkdir -p {data,results,notebooks,scripts,docs/figures}

# Set up environment variables
echo "🌍 Setting up environment..."
echo 'export PYTHONPATH="/workspaces/qem-bench/src:$PYTHONPATH"' >> ~/.bashrc
echo 'export JAX_PLATFORM_NAME="cpu"' >> ~/.bashrc

# Create useful aliases
echo "⚡ Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# QEM-Bench aliases
alias qem='python -m qem_bench'
alias pytest-cov='pytest --cov=qem_bench --cov-report=html --cov-report=term'
alias pytest-fast='pytest -x -v'
alias lint='ruff check src tests'
alias format='black src tests && isort src tests'
alias typecheck='mypy src'
alias docs='cd docs && make html'
alias clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Useful functions
qem-test() {
    echo "🧪 Running QEM-Bench test suite..."
    pytest --cov=qem_bench --cov-report=html --cov-report=term-missing
}

qem-bench-demo() {
    echo "🎯 Running QEM-Bench demo..."
    python -c "
from qem_bench import ZeroNoiseExtrapolation
from qem_bench.benchmarks import create_benchmark_circuit
from qem_bench.jax import JAXSimulator

# Create demo circuit
circuit = create_benchmark_circuit('quantum_volume', qubits=3, depth=5)
simulator = JAXSimulator(num_qubits=3)

print('✅ QEM-Bench is working correctly!')
print(f'Demo circuit: {circuit.num_qubits} qubits, {circuit.depth} depth')
print(f'Simulator backend: {simulator.backend_info[\"backend\"]}')
"
}

qem-gpu-test() {
    echo "🔥 Testing GPU acceleration..."
    python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')

if 'gpu' in str(jax.devices()).lower():
    print('✅ GPU acceleration available!')
else:
    print('ℹ️  CPU-only mode (GPU not available)')
"
}
EOF

echo ""
echo "✅ QEM-Bench development environment setup complete!"
echo ""
echo "🎯 Quick commands:"
echo "  qem-test          - Run full test suite with coverage"
echo "  qem-bench-demo    - Run a quick demo to verify installation" 
echo "  qem-gpu-test      - Test GPU acceleration availability"
echo "  lint              - Run code linting"
echo "  format            - Format code with black & isort"
echo "  typecheck         - Run type checking with mypy"
echo ""
echo "📚 Get started:"
echo "  1. Run: qem-bench-demo"
echo "  2. Try: jupyter lab (opens on port 8888)"
echo "  3. Explore: cd notebooks"
echo ""
echo "Happy quantum error mitigation! 🚀"