# Contributing to ProofOfThought

Thank you for your interest in contributing to ProofOfThought! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- A GitHub account

### Setting Up the Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/proofofthought.git
   cd proofofthought
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify setup by running tests**
   ```bash
   python run_tests.py
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run the test suite:
   ```bash
   python run_tests.py
   ```

4. Run code quality checks:
   ```bash
   black .
   ruff check .
   mypy z3adapter/
   ```

5. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

6. Push to your fork and create a pull request

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when applicable ("Fix #123: description")

## Code Standards

### Style Guide

- **Formatter**: Black with line length of 100
- **Linter**: Ruff
- **Type hints**: Required for all new code
- **Docstrings**: Google style for public functions and classes

### Example Function

```python
def process_query(question: str, timeout: int = 10000) -> QueryResult:
    """Process a reasoning query using Z3.

    Args:
        question: Natural language question to answer.
        timeout: Z3 solver timeout in milliseconds.

    Returns:
        QueryResult containing the answer and execution details.

    Raises:
        ValueError: If question is empty.
    """
    if not question:
        raise ValueError("Question cannot be empty")
    # Implementation...
```

### Type Hints

All new code must include type hints:

```python
# Good
def calculate_score(results: list[bool]) -> float:
    return sum(results) / len(results)

# Bad
def calculate_score(results):
    return sum(results) / len(results)
```

## Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_sort_manager_creates_enum_sort`
- Each test should test one thing

### Test Structure

```python
import unittest
from z3adapter.dsl.sorts import SortManager

class TestSortManager(unittest.TestCase):
    """Test cases for SortManager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sort_manager = SortManager()

    def test_builtin_sorts_initialized(self) -> None:
        """Test that built-in sorts are initialized."""
        self.assertIn("IntSort", self.sort_manager.sorts)
```

### Running Tests

```bash
# All tests
python run_tests.py

# Specific test file
python -m pytest tests/unit/test_sort_manager.py -v

# With coverage
python -m pytest tests/ --cov=z3adapter --cov-report=html
```

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Run code formatters and linters
   - Update documentation if needed
   - Add tests for new functionality

2. **PR description should include:**
   - Summary of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (for UI changes)

3. **Review process:**
   - PRs require at least one approval
   - Address review comments promptly
   - Keep PRs focused and reasonably sized

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` in the GitHub issue tracker.

### Feature Ideas

- New postprocessing techniques
- Additional benchmark datasets
- Performance optimizations
- Documentation improvements
- New backend implementations

### Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Translate documentation

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person

## Questions?

- Open an issue for questions
- Check existing issues and documentation first
- Tag maintainers if urgent

Thank you for contributing!
