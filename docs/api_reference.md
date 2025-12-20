# API Reference

Complete API documentation for the Personal Finance Dashboard modules.

## Data Loader Module

### `load_transactions(file_path, format='auto')`

Load transactions from a file.

**Parameters:**
- `file_path` (str): Path to the transaction file
- `format` (str, optional): File format ('csv', 'excel', 'json', or 'auto')

**Returns:**
- DataFrame: Loaded transactions

**Example:**
```python
from src.data_loader import load_transactions

transactions = load_transactions('data/transactions.csv')
```

## Analyzer Module

### `analyze_spending(transactions, group_by='category')`

Analyze spending patterns in transaction data.

**Parameters:**
- `transactions` (DataFrame): Transaction data
- `group_by` (str, optional): Grouping field ('category', 'merchant', 'month')

**Returns:**
- dict: Spending summary with totals and breakdowns

**Example:**
```python
from src.analyzer import analyze_spending

summary = analyze_spending(transactions)
print(summary['total_expenses'])
```

### `calculate_savings_rate(transactions)`

Calculate savings rate from transaction data.

**Parameters:**
- `transactions` (DataFrame): Transaction data

**Returns:**
- float: Savings rate as a percentage

## Visualizer Module

### `create_spending_chart(data, output_path, chart_type='pie')`

Generate spending visualization charts.

**Parameters:**
- `data` (dict): Spending data from analyzer
- `output_path` (str): Path to save the chart
- `chart_type` (str, optional): Chart type ('pie', 'bar', 'line')

**Returns:**
- None

**Example:**
```python
from src.visualizer import create_spending_chart

create_spending_chart(
    summary,
    output_path='output/spending.png',
    chart_type='pie'
)
```

## Categorizer Module

### `categorize_transaction(description, amount)`

Automatically categorize a transaction.

**Parameters:**
- `description` (str): Transaction description
- `amount` (float): Transaction amount

**Returns:**
- str: Assigned category

### `add_category_rule(pattern, category, use_regex=False)`

Add a custom categorization rule.

**Parameters:**
- `pattern` (str): Text pattern to match
- `category` (str): Category to assign
- `use_regex` (bool, optional): Use regex matching

**Returns:**
- None

## Reporter Module

### `generate_report(transactions, output_path, format='html')`

Generate a comprehensive financial report.

**Parameters:**
- `transactions` (DataFrame): Transaction data
- `output_path` (str): Path to save the report
- `format` (str, optional): Report format ('html', 'pdf', 'excel')

**Returns:**
- str: Path to generated report

(Detailed API documentation to be expanded)
