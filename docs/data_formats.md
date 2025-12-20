# Data Formats

This document describes the supported input and output data formats for the Personal Finance Dashboard.

## Input Formats

### Standard CSV Format

The recommended format for transaction data:

```csv
Date,Description,Amount,Category,Account,Transaction Type
2025-01-15,Grocery Store,-75.50,Food,Checking,debit
2025-01-14,Salary Deposit,3000.00,Income,Checking,credit
```

**Required Fields:**
- `Date`: Transaction date (YYYY-MM-DD format)
- `Description`: Transaction description or memo
- `Amount`: Transaction amount (negative for expenses, positive for income)

**Optional Fields:**
- `Category`: Expense/income category
- `Account`: Account name or identifier
- `Transaction Type`: `debit` or `credit`
- `Merchant`: Merchant or payee name
- `Notes`: Additional notes
- `Tags`: Comma-separated tags

### Bank-Specific Formats

Common bank export formats are supported:
- Chase Bank CSV
- Bank of America CSV
- Wells Fargo CSV
- American Express CSV

The system will attempt to auto-detect the format.

### Excel Files

Excel files (.xlsx, .xls) are supported. The first sheet will be read as transaction data.

## Output Formats

### Processed CSV
Cleaned and categorized transactions saved to `output/processed_transactions.csv`

### Excel Reports
Multi-sheet Excel reports with summaries, charts, and detailed data.

### JSON Export
Machine-readable JSON format for integration with other tools.

(Detailed format specifications to be added)
