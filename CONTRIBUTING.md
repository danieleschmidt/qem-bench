# Contributing to QEM-Bench

Thank you for your interest in contributing to QEM-Bench! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/qem-bench
   cd qem-bench
   ```
3. **Set up development environment** (see [DEVELOPMENT.md](DEVELOPMENT.md))
4. **Create a feature branch**: `git checkout -b feature/my-new-feature`

## Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the issue template for bug reports
- Include minimal reproducible example
- Specify your environment (Python version, OS, etc.)

### ✨ Feature Requests  
- Describe the motivation and use case
- Provide examples of the desired API
- Consider implementation complexity

### 🔧 Code Contributions
- New error mitigation techniques
- Benchmark circuits and metrics
- Performance improvements
- Documentation improvements

### 📚 Documentation
- Tutorial improvements
- API documentation
- Examples and use cases

## Development Workflow

### Code Standards

- **Style**: Use `black` for formatting (88 character line length)
- **Linting**: Pass `ruff` checks
- **Type Hints**: Include type annotations for new code
- **Tests**: Add tests for new functionality
- **Documentation**: Update relevant documentation

### Commit Guidelines

- Use clear, descriptive commit messages
- Reference issues: `Fixes #123` or `Closes #456`
- Keep commits focused and atomic

Example:
```
Add Richardson extrapolation to ZNE

- Implement polynomial and exponential fitting
- Add confidence interval estimation  
- Include regression tests
- Update documentation with examples

Fixes #45
```

### Pull Request Process

1. **Update your branch** with latest main: `git pull upstream main`
2. **Run pre-commit checks**: `pre-commit run --all-files`
3. **Run tests**: `pytest`
4. **Update documentation** if needed
5. **Create pull request** with:
   - Clear title and description
   - Link to related issues
   - Screenshots/examples if applicable

### Review Process

- All PRs require at least one review
- Address reviewer feedback promptly
- Maintain a professional and collaborative tone
- Be open to suggestions and changes

## Testing Guidelines

### Test Requirements
- New features must include tests
- Maintain or improve test coverage
- Tests should be fast and deterministic

### Test Categories
- **Unit tests**: Fast, isolated component tests
- **Integration tests**: Component interaction tests  
- **Hardware tests**: Mark with `@pytest.mark.hardware`
- **Slow tests**: Mark with `@pytest.mark.slow`

### Running Tests
```bash
# All tests
pytest

# Specific categories
pytest -m "not slow and not hardware"  # Fast tests only
pytest tests/unit/                      # Unit tests only
```

## Adding New Mitigation Methods

To add a new error mitigation technique:

1. **Create module** in `src/qem_bench/mitigation/`
2. **Implement base interface** (see existing methods)
3. **Add comprehensive tests** in `tests/unit/mitigation/`
4. **Update documentation** and examples
5. **Add to benchmarking suite** if applicable

Example structure:
```python
from typing import Protocol
from .base import MitigationProtocol

class MyMitigation(MitigationProtocol):
    def __init__(self, ...):
        ...
    
    def mitigate(self, circuit, backend, **kwargs):
        ...
```

## Documentation Guidelines

- Use clear, concise language
- Include code examples  
- Document all public APIs
- Update relevant tutorials

Build documentation locally:
```bash
cd docs/
make html
```

## Community Guidelines

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: General questions, ideas
- **Email**: security@qem-bench.org for security issues

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions  
- GitHub contributor graph

## Questions?

Feel free to:
- Open a discussion on GitHub
- Reach out to maintainers
- Check existing issues and documentation

Thank you for contributing to QEM-Bench! 🚀