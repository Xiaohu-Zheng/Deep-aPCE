# Deep-aPCE

[![CI/CD Pipeline](https://github.com/Xiaohu-Zheng/Deep-aPCE/workflows/Deep-aPCE%20CI/CD%20Pipeline/badge.svg)](https://github.com/Xiaohu-Zheng/Deep-aPCE/actions)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Deep Adaptive Arbitrary Polynomial Chaos Expansion (Deep-aPCE)** - A mini-data-driven semi-supervised method for uncertainty quantification

## 📖 Overview

This repository implements the Deep-aPCE algorithm presented in the paper:

> **Deep adaptive arbitrary polynomial chaos expansion: A mini-data-driven semi-supervised method for uncertainty quantification**  
> Yao, Wen; Zheng, Xiaohu; Zhang, Jun; Wang, Ning; Tang, Guijian  
> *Reliability Engineering & System Safety*, 2023, 229: 108813  
> [DOI: 10.1016/j.ress.2022.108813](https://doi.org/10.1016/j.ress.2022.108813)

## 🌟 Key Features

- **Mini-data-driven**: Requires minimal training data
- **Semi-supervised**: Combines labeled and unlabeled data
- **Adaptive**: Automatically adjusts polynomial order
- **Uncertainty Quantification**: Comprehensive UQ framework

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Xiaohu-Zheng/Deep-aPCE.git
cd Deep-aPCE

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
import numpy as np
from src.Deep_PCE import k_center_moment, order_mat_fun

# Generate sample data
x = torch.randn(100, 5)

# Calculate k-center moments
mu = k_center_moment(x, k=3)
print(f"Moment shape: {mu.shape}")

# Generate order matrix
order_mat = order_mat_fun(dim=2, order=3)
print(f"Order matrix shape: {order_mat.shape}")
```

### Example: Cantilever Beam Analysis

```bash
python DPCE_cantilever_beam.py
```

## 📁 Project Structure

```
Deep-aPCE/
├── src/                    # Source code
│   ├── __init__.py
│   ├── apc.py             # Adaptive Polynomial Chaos
│   ├── data_process.py    # Data processing utilities
│   ├── Deep_PCE.py        # Core Deep-aPCE implementation
│   ├── models.py          # Neural network models
│   └── pce_loss.py        # Loss functions
├── tests/                 # Test suite
│   └── test_deep_apce.py
├── data/                  # Data files
├── examples/              # Example scripts
│   ├── DPCE_cantilever_beam.py
│   └── cantilever_fun.py
├── .github/workflows/     # CI/CD configuration
├── requirements.txt       # Dependencies
├── pytest.ini            # Test configuration
└── README.md             # This file
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/ --line-length 100

# Sort imports
isort src/ tests/ --profile black

# Check code quality
flake8 src/ tests/ --max-line-length 100
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📊 Algorithm Description

Deep-aPCE combines deep learning with polynomial chaos expansion for uncertainty quantification:

1. **Adaptive Polynomial Chaos**: Automatically determines optimal polynomial order
2. **Neural Network Enhancement**: Uses DNN to capture complex relationships
3. **Semi-supervised Learning**: Leverages both labeled and unlabeled data
4. **Moment-based Approach**: Calculates statistical moments for UQ

### Mathematical Foundation

The algorithm computes k-center moments:

```
μ_k = E[X^k]
```

And constructs polynomial basis:

```
Ψ_α(x) = ∏ Ψ_αi(xi)
```

where α is a multi-index determined by `order_mat_fun`.

## 🎯 Applications

- **Structural Reliability**: Cantilever beam analysis
- **Aerospace Engineering**: Missile design optimization
- **Uncertainty Quantification**: General UQ problems
- **Sensitivity Analysis**: Parameter sensitivity studies

## 📈 Performance

- **Accuracy**: State-of-the-art results on benchmark problems
- **Efficiency**: Requires less training data than traditional methods
- **Scalability**: Handles high-dimensional problems

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{Zheng2023Deep,
   author = {Yao, Wen and Zheng, Xiaohu and Zhang, Jun and Wang, Ning and Tang, Guijian},
   title = {Deep adaptive arbitrary polynomial chaos expansion: A mini-data-driven semi-supervised method for uncertainty quantification},
   journal = {Reliability Engineering \& System Safety},
   volume = {229},
   pages = {108813},
   ISSN = {09518320},
   DOI = {10.1016/j.ress.2022.108813},
   year = {2023},
   type = {Journal Article}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Xiaohu Zheng
- **Email**: zhengxiaohu16@nudt.edu.cn
- **GitHub**: [@Xiaohu-Zheng](https://github.com/Xiaohu-Zheng)

## 🙏 Acknowledgments

- National University of Defense Technology
- Reliability Engineering & System Safety journal

---

**Star ⭐ this repository if you find it helpful!**
