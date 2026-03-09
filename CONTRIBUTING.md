# Contributing to Deep-aPCE

Thank you for your interest in contributing to Deep-aPCE! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (Python version, OS, etc.)

### Suggesting Enhancements

Open an issue with:
- Clear description of the enhancement
- Rationale for the enhancement
- Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

## 🛠️ Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/Xiaohu-Zheng/Deep-aPCE.git
cd Deep-aPCE
pip install -e .[dev]
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## 📝 Code Style

We follow these coding standards:

- **PEP 8**: Python style guide
- **Black**: Code formatter (line length: 100)
- **isort**: Import sorting
- **Type hints**: Encouraged but not required
- **Docstrings**: Google style preferred

### Code Formatting

```bash
# Format code
black src/ tests/ --line-length 100

# Sort imports
isort src/ tests/ --profile black

# Check style
flake8 src/ tests/ --max-line-length 100
```

## ✅ Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain or improve code coverage

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_deep_apce.py -v
```

## 📚 Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update inline comments for complex logic

## 🔍 Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing
- [ ] Local tests pass
- [ ] Pre-commit hooks pass

## 📧 Questions?

Feel free to open an issue for questions or contact:
- Email: zhengxiaohu16@nudt.edu.cn
- GitHub: [@Xiaohu-Zheng](https://github.com/Xiaohu-Zheng)

Thank you for contributing! 🎉
