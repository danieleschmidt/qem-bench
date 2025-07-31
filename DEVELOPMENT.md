# Development Setup

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/qem-bench
cd qem-bench

# Create virtual environment  
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Development Workflow

### Code Style

We use automated formatting and linting:

- **Black**: Code formatting (88 character line length)
- **Ruff**: Fast Python linter and import sorter
- **MyPy**: Static type checking

Run locally:
```bash
black src/ tests/
ruff check src/ tests/ --fix
mypy src/
```

### Testing

Testing framework: **pytest** with coverage reporting

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qem_bench --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "not hardware"  # Skip hardware tests
```

Test structure:
- `tests/unit/`: Fast unit tests
- `tests/integration/`: Integration tests  
- `tests/e2e/`: End-to-end benchmarks

### Pre-commit Hooks

Automated checks run on every commit:
- Code formatting (black)
- Linting (ruff)  
- Type checking (mypy)
- Security scanning (bandit)
- Dependency vulnerabilities (safety)

Manually run: `pre-commit run --all-files`

### Documentation

Documentation is built with Sphinx:

```bash
cd docs/
make html  # Build HTML documentation
make clean # Clean build artifacts
```

### Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v0.1.0`
5. Push to trigger CI/CD: `git push --tags`

## Project Structure

```
qem-bench/
├── src/qem_bench/          # Main package
│   ├── mitigation/         # Error mitigation methods
│   ├── benchmarks/         # Benchmark suite
│   ├── noise/              # Noise modeling
│   └── backends/           # Hardware/simulator interfaces
├── tests/                  # Test suite
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # Project overview
```

## Environment Variables

Optional configuration:
- `QEM_BENCH_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `QEM_BENCH_CACHE_DIR`: Override default cache directory

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.