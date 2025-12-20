# Getting Started Guide

This guide provides detailed tutorials and examples beyond the basic installation covered in the main README.

## Tutorial: Your First Financial Analysis

### Preparing Sample Data

Download your bank transactions or create a sample CSV file:

```csv
Date,Description,Amount,Category
2025-01-15,Grocery Store,-75.50,Food
2025-01-14,Salary Deposit,3000.00,Income
2025-01-13,Electric Bill,-120.00,Utilities
```

Save this as `data/transactions.csv`.

### Running the Analysis

```python
from src.data_loader import load_transactions
from src.analyzer import analyze_spending

transactions = load_transactions('data/transactions.csv')
summary = analyze_spending(transactions)
print(summary)
```

## Common Use Cases

### Monthly Budget Tracking
### Expense Categorization
### Income Analysis
### Trend Visualization

(More detailed tutorials to be added)

For installation instructions, see the main [README](../README.md).
