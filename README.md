# Personal Finance Dashboard

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-0%25-red)

A comprehensive personal finance dashboard for tracking, analyzing, and visualizing your financial data. This tool helps you gain insights into your spending patterns, income sources, and overall financial health through an automated data processing pipeline and interactive visualizations.

## Table of Contents

- [Features](#features)
- [Project Goals](#project-goals)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Import**: Load financial data from multiple sources (CSV, Excel, bank exports)
- **Data Cleaning**: Automated data validation and preprocessing
- **Transaction Categorization**: Intelligent categorization of expenses and income
- **Visualization**: Interactive charts and graphs for financial insights
- **Reports**: Generate comprehensive financial reports and summaries
- **Export**: Export processed data and visualizations

## Project Goals

This project aims to:

1. **Simplify Personal Finance Tracking**: Provide an easy-to-use tool for tracking daily expenses and income
2. **Enable Data-Driven Decisions**: Help users make informed financial decisions through clear visualizations
3. **Automate Analysis**: Reduce manual work by automating data cleaning, categorization, and reporting
4. **Ensure Privacy**: Process all data locally without relying on third-party cloud services
5. **Maintain Flexibility**: Support various data formats and customizable categorization rules

## Quick Start

Get started with a simple example:

```python
# Example: Load and analyze your financial data
from src.data_loader import load_transactions
from src.analyzer import analyze_spending
from src.visualizer import create_spending_chart

# Load your transaction data
transactions = load_transactions('data/sample_transactions.csv')

# Analyze spending patterns
spending_summary = analyze_spending(transactions)
print(spending_summary)

# Create a visualization
create_spending_chart(spending_summary, output_path='output/spending_chart.png')
```

### Sample Data Format

Place your transaction data in the `data/` directory. Expected CSV format:

```csv
Date,Description,Amount,Category
2025-01-15,Grocery Store,-75.50,Food
2025-01-14,Salary Deposit,3000.00,Income
2025-01-13,Electric Bill,-120.00,Utilities
```

## Installation

### Prerequisites

- **Python 3.11 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/andyvn1/personal-finance-dashboard.git
cd personal-finance-dashboard
```

#### 2. Create Virtual Environment

Creating a virtual environment keeps your project dependencies isolated from your system Python installation.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

#### 3. Install Dependencies

**For users (production):**
```bash
pip install -r requirements.txt
```

**For developers (includes testing and linting tools):**
```bash
pip install -r requirements-dev.txt
```

#### 4. Verify Installation

```bash
python -c "import pandas; import matplotlib; print('Installation successful!')"
```

## Usage

### Basic Workflow

1. **Prepare Your Data**: Place your financial data files in the `data/` directory
2. **Run the Pipeline**: Execute the main analysis script
3. **View Results**: Check the `output/` directory for reports and visualizations

### Example Workflow

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the analysis
python scripts/analyze_finances.py --input data/transactions.csv --output output/

# View the generated report
open output/financial_report.html  # macOS
# or
start output/financial_report.html  # Windows
```

## Architecture Overview

The Personal Finance Dashboard follows a modular pipeline architecture:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Data      │────▶│   Data       │────▶│   Analysis  │────▶│ Visualization│
│   Loading   │     │   Cleaning   │     │   Engine    │     │   & Reports  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Pipeline Stages

1. **Data Loading**
   - Import data from various sources (CSV, Excel, JSON)
   - Validate data formats and schemas
   - Handle missing or malformed data

2. **Data Cleaning**
   - Standardize date formats
   - Normalize transaction descriptions
   - Remove duplicates
   - Handle missing values

3. **Analysis Engine**
   - Categorize transactions using rules or ML models
   - Calculate spending patterns and trends
   - Identify anomalies and unusual transactions
   - Generate financial metrics (income, expenses, savings rate)

4. **Visualization & Reports**
   - Create interactive charts (spending by category, trends over time)
   - Generate summary statistics
   - Export reports in multiple formats (HTML, PDF, Excel)

### Core Components

- **`src/data_loader.py`**: Handles data import from various sources
- **`src/cleaner.py`**: Data cleaning and preprocessing logic
- **`src/categorizer.py`**: Transaction categorization engine
- **`src/analyzer.py`**: Financial analysis and metrics calculation
- **`src/visualizer.py`**: Chart generation and visualization
- **`src/reporter.py`**: Report generation and export

## Project Structure

```
personal-finance-dashboard/
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data import modules
│   ├── cleaner.py           # Data cleaning utilities
│   ├── categorizer.py       # Transaction categorization
│   ├── analyzer.py          # Financial analysis engine
│   ├── visualizer.py        # Visualization generation
│   └── reporter.py          # Report generation
│
├── tests/                    # Test files
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_cleaner.py
│   ├── test_categorizer.py
│   └── test_analyzer.py
│
├── data/                     # Data files (gitignored for privacy)
│   ├── sample_transactions.csv
│   └── README.md            # Data format documentation
│
├── docs/                     # Documentation
│   ├── getting_started.md   # Beginner's guide
│   ├── data_formats.md      # Supported data formats
│   ├── customization.md     # Customization guide
│   └── api_reference.md     # API documentation
│
├── scripts/                  # Utility scripts
│   ├── analyze_finances.py  # Main analysis script
│   └── export_report.py     # Report export utility
│
├── output/                   # Generated reports (gitignored)
│
├── venv/                     # Virtual environment (gitignored)
│
├── pyproject.toml           # Project configuration and metadata
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Install pre-commit hooks (if available):
   ```bash
   pre-commit install
   ```

### Code Quality Tools

#### Code Formatting
Format your code with Black:
```bash
black .
```

Check formatting without making changes:
```bash
black --check .
```

#### Linting
Run Ruff linter:
```bash
ruff check .
```

Auto-fix issues:
```bash
ruff check --fix .
```

#### Type Checking
Run mypy for static type checking:
```bash
mypy src/
```

### Development Workflow

1. Create a new branch for your feature/fix
2. Write code following PEP 8 style guidelines
3. Add tests for new functionality
4. Run code quality tools (black, ruff, mypy)
5. Ensure all tests pass
6. Submit a pull request

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run tests with coverage report:
```bash
pytest --cov=src --cov-report=term-missing
```

Generate HTML coverage report:
```bash
pytest --cov=src --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in your browser.

### Running Specific Tests

Run a specific test file:
```bash
pytest tests/test_analyzer.py
```

Run a specific test function:
```bash
pytest tests/test_analyzer.py::test_calculate_spending
```

## Documentation

Detailed documentation is available in the [docs](docs/) directory:

- **[Getting Started Guide](docs/getting_started.md)**: Step-by-step tutorial for new users
- **[Data Formats](docs/data_formats.md)**: Supported input/output formats and schemas
- **[Customization Guide](docs/customization.md)**: How to customize categories and rules
- **[API Reference](docs/api_reference.md)**: Complete API documentation for developers

## Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, or code contributions, your input helps make this project better.

### How to Contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a new branch** for your feature (`git checkout -b feature/amazing-feature`)
4. **Make your changes** and commit them (`git commit -m 'Add amazing feature'`)
5. **Push to your branch** (`git push origin feature/amazing-feature`)
6. **Open a Pull Request** on GitHub

### Contribution Guidelines

- Follow the existing code style and conventions
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting
- Be respectful and constructive in discussions

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [issue tracker](https://github.com/andyvn1/personal-finance-dashboard/issues)
2. If not, create a new issue with a clear description
3. Include steps to reproduce (for bugs) or use cases (for features)
4. Add relevant labels to help categorize the issue

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

You are free to:
- Use this software for any purpose
- Modify the source code
- Distribute the software
- Use it in commercial projects

**Conditions:**
- Include the original copyright notice and license in any copies

---

**Happy Budgeting!** If you have questions or need help, feel free to [open an issue](https://github.com/andyvn1/personal-finance-dashboard/issues) or reach out to the maintainers.
