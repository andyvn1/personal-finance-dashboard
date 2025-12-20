# Personal Finance Dashboard

A comprehensive personal finance dashboard for tracking and analyzing financial data.

## Requirements

- Python 3.11 or higher

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/andyvn1/personal-finance-dashboard.git
cd personal-finance-dashboard
```

### 2. Create and activate virtual environment

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

### 3. Install dependencies

**For development (recommended):**
```bash
pip install -r requirements-dev.txt
```

**For production only:**
```bash
pip install -r requirements.txt
```

## Dependencies

### Main Dependencies
- **pandas** (>=2.0.0) - Data manipulation and analysis
- **matplotlib** (>=3.7.0) - Data visualization
- **openpyxl** (>=3.1.0) - Excel file handling
- **pytest** (>=7.4.0) - Testing framework
- **pytest-cov** (>=4.1.0) - Test coverage

### Development Dependencies
- **black** (>=23.0.0) - Code formatting
- **ruff** (>=0.1.0) - Fast Python linter
- **mypy** (>=1.5.0) - Static type checking
- **pandas-stubs** (>=2.0.0) - Type stubs for pandas

## Development Tools

### Code Formatting
Format your code with Black:
```bash
black .
```

### Linting
Run Ruff linter:
```bash
ruff check .
```

Auto-fix issues:
```bash
ruff check --fix .
```

### Type Checking
Run mypy for type checking:
```bash
mypy src/
```

### Testing
Run tests with pytest:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Project Structure

```
personal-finance-dashboard/
├── src/              # Source code
├── tests/            # Test files
├── data/             # Data files (CSV, Excel, etc.)
├── docs/             # Documentation
├── scripts/          # Utility scripts
├── venv/             # Virtual environment (not tracked in git)
├── pyproject.toml    # Project configuration
├── requirements.txt  # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── .gitignore        # Git ignore rules
├── LICENSE           # MIT License
└── README.md         # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
