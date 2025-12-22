# Transaction Data Schema

This document describes the standardized internal data schema for financial transactions in the Personal Finance Dashboard.

## Table of Contents

- [Overview](#overview)
- [Field Definitions](#field-definitions)
- [Data Types](#data-types)
- [Constraints](#constraints)
- [CSV Input Formats](#csv-input-formats)
- [Examples](#examples)

## Overview

The Personal Finance Dashboard uses a flexible, standardized schema to represent financial transactions. The schema is designed to accommodate data from various sources (bank exports, credit card statements, manual entry) while maintaining data integrity and consistency.

### Design Principles

1. **Flexibility**: Only three fields are strictly required (date, description, amount)
2. **Auto-detection**: Transaction type is automatically inferred from the amount sign
3. **Validation**: Sensible constraints prevent data errors without being overly restrictive
4. **Extensibility**: Optional fields support rich transaction metadata

## Field Definitions

### Required Fields

These fields are the absolute minimum for a valid transaction:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | datetime | Transaction date | `2025-01-15` |
| `description` | str | Transaction description or memo | `"Grocery Store Purchase"` |
| `amount` | Decimal | Transaction amount (negative for expenses) | `-75.50` or `3000.00` |

### Recommended Fields

These fields are highly recommended and will be auto-populated with defaults if missing:

| Field | Type | Default | Description | Example |
|-------|------|---------|-------------|---------|
| `account` | str | `"Unknown"` | Account identifier or name | `"Checking"`, `"Visa ****1234"` |
| `transaction_type` | TransactionType | Auto-detected | `DEBIT` or `CREDIT` | `DEBIT` (for expenses) |

### Optional Fields

These fields provide additional context and organization:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `category` | str | Spending/income category | `"Food & Dining"`, `"Salary"` |
| `merchant` | str | Merchant or payee name | `"Starbucks"`, `"Amazon"` |
| `notes` | str | Additional notes | `"Business lunch with client"` |
| `tags` | list[str] | Custom tags for organization | `["business", "tax-deductible"]` |
| `reference_id` | str | External reference from bank | `"TXN12345678"` |
| `balance` | Decimal | Account balance after transaction | `1234.56` |
| `original_description` | str | Unprocessed original description | Raw bank export text |

## Data Types

### TransactionType Enum

```python
class TransactionType(Enum):
    DEBIT = "debit"    # Money out (expenses, withdrawals)
    CREDIT = "credit"  # Money in (income, deposits)
```

**Auto-Detection Logic**:
- Amount < 0 → `DEBIT`
- Amount > 0 → `CREDIT`

### AccountType Enum

```python
class AccountType(Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    CASH = "cash"
    INVESTMENT = "investment"
    OTHER = "other"
```

### Standard Categories

The system includes 30+ predefined categories:

**Income Categories**:
- Income, Salary, Investment Income

**Expense Categories**:
- Food & Dining (Groceries, Restaurants)
- Transportation (Gas & Fuel, Public Transit)
- Shopping (Clothing, Electronics)
- Utilities (Electricity, Water, Internet, Phone)
- Healthcare
- Entertainment (Movies & TV, Music)
- Insurance (Health, Car, Home)
- Education
- Personal Care
- Travel
- Fees & Charges (Bank Fees, ATM Fees)
- Taxes
- Gifts & Donations
- Uncategorized

## Constraints

### Length Constraints

| Field | Maximum Length |
|-------|----------------|
| Description | 500 characters |
| Account | 100 characters |
| Category | 100 characters |
| Merchant | 200 characters |
| Notes | 1000 characters |
| Reference ID | 100 characters |
| Tag (individual) | 50 characters |
| Tags (total count) | 20 tags |

### Amount Constraints

- **Range**: -999,999,999.99 to 999,999,999.99
- **Precision**: 2 decimal places
- **Type**: Decimal (for precise financial calculations)
- **Validation**: Amount cannot be zero

### Date Constraints

- **Range**: Years 1900-2100
- **Format**: ISO 8601 datetime
- **Type**: Python `datetime` object

## CSV Input Formats

The system supports multiple CSV formats from different banks and sources.

### Standard Format

This is the recommended format for manual data entry:

```csv
Date,Description,Amount,Category,Account,Transaction Type
2025-01-15,Grocery Store,-75.50,Food,Checking,debit
2025-01-14,Salary Deposit,3000.00,Income,Checking,credit
2025-01-13,Electric Bill,-120.00,Utilities,Checking,debit
```

**Required Headers**: `Date`, `Description`, `Amount`
**Optional Headers**: `Category`, `Account`, `Transaction Type`, `Merchant`, `Notes`, `Tags`

### Minimal Format

Minimum viable format (only required fields):

```csv
Date,Description,Amount
2025-01-15,Grocery Store,-75.50
2025-01-14,Salary Deposit,3000.00
2025-01-13,Electric Bill,-120.00
```

### Detailed Format

Extended format with all optional fields:

```csv
Date,Description,Amount,Category,Merchant,Account,Transaction Type,Notes,Tags
2025-01-15,Grocery Store,-75.50,Food,Safeway,Checking,debit,"Weekly shopping","groceries,food"
2025-01-14,Salary Deposit,3000.00,Income,Employer Inc,Checking,credit,"Biweekly paycheck","income,salary"
```

### Bank-Specific Formats

#### Chase Bank

```csv
Transaction Date,Post Date,Description,Category,Type,Amount
01/15/2025,01/15/2025,SAFEWAY #123,Groceries,Sale,-75.50
```

**Field Mappings**:
- Date: `Transaction Date` or `Post Date`
- Description: `Description`
- Amount: `Amount`
- Category: `Category` or `Type`

#### Bank of America

```csv
Date,Description,Amount,Running Bal.
01/15/2025,GROCERY STORE PURCHASE,-75.50,1234.56
```

**Field Mappings**:
- Date: `Date` or `Posted Date`
- Description: `Description` or `Payee`
- Amount: `Amount`
- Balance: `Running Bal.`

#### Wells Fargo

```csv
Date,Description,Amount,Balance
01/15/2025,GROCERY STORE,-75.50,1234.56
```

**Field Mappings**:
- Date: `Date`
- Description: `Description`
- Amount: `Amount`
- Balance: `Balance`

#### American Express

```csv
Date,Description,Amount,Category
01/15/2025,GROCERY STORE,-75.50,Merchandise & Supplies-Groceries
```

**Field Mappings**:
- Date: `Date`
- Description: `Description`
- Amount: `Amount`
- Category: `Category`

## Examples

### Example 1: Expense Transaction

```python
from src.schema import Transaction, TransactionType
from datetime import datetime
from decimal import Decimal

expense = Transaction(
    date=datetime(2025, 1, 15),
    description="Grocery Store",
    amount=Decimal("-75.50"),
    account="Checking",
    category="Food & Dining"
)

# Auto-detected as DEBIT
assert expense.transaction_type == TransactionType.DEBIT
assert expense.is_expense == True
```

### Example 2: Income Transaction

```python
income = Transaction(
    date=datetime(2025, 1, 14),
    description="Salary Deposit",
    amount=Decimal("3000.00"),
    account="Checking",
    category="Salary",
    notes="Biweekly paycheck"
)

# Auto-detected as CREDIT
assert income.transaction_type == TransactionType.CREDIT
assert income.is_income == True
```

### Example 3: Minimal Transaction

```python
# Only required fields
minimal = Transaction(
    date=datetime(2025, 1, 13),
    description="Electric Bill",
    amount=Decimal("-120.00")
)

# Defaults applied
assert minimal.account == "Unknown"
assert minimal.transaction_type == TransactionType.DEBIT
```

### Example 4: Rich Transaction with Metadata

```python
detailed = Transaction(
    date=datetime(2025, 1, 12),
    description="Business Lunch",
    amount=Decimal("-85.00"),
    account="Corporate Card",
    category="Food & Dining",
    merchant="Italian Restaurant",
    notes="Client meeting - Project Alpha",
    tags=["business", "tax-deductible", "client"],
    reference_id="TXN789012"
)
```

## Validation Behavior

The schema uses **flexible validation** that balances data integrity with format compatibility:

### What Gets Validated

✅ Required fields exist and are non-empty
✅ Data types are correct (datetime, Decimal, etc.)
✅ String lengths don't exceed maximums
✅ Amount is non-zero and within range
✅ Date is within reasonable year range

### What Doesn't Fail Validation

✅ Missing optional fields
✅ Missing recommended fields (defaults applied)
✅ Unknown categories (will be marked "Uncategorized")
✅ Extra fields in CSV (ignored gracefully)
✅ Different CSV header names (mapped automatically)

## Usage in Code

### Creating a Transaction

```python
from src.schema import Transaction
from datetime import datetime
from decimal import Decimal

# Create from parsed CSV data
transaction = Transaction(
    date=datetime.fromisoformat("2025-01-15"),
    description="Coffee Shop",
    amount=Decimal("-5.25"),
    account="Checking",
    category="Food & Dining"
)
```

### Converting to Dictionary

```python
# Export transaction as dictionary
data = transaction.to_dict()
# {
#     'date': '2025-01-15T00:00:00',
#     'description': 'Coffee Shop',
#     'amount': '-5.25',
#     'account': 'Checking',
#     'transaction_type': 'debit',
#     'category': 'Food & Dining',
#     ...
# }
```

### Accessing Properties

```python
# Check transaction type
if transaction.is_expense:
    print(f"Spent: ${transaction.absolute_amount}")

# Access auto-detected type
print(transaction.transaction_type)  # TransactionType.DEBIT
```

## See Also

- [Data Formats Guide](data_formats.md) - Detailed CSV format specifications
- [Getting Started](getting_started.md) - Quick start with sample data
- [API Reference](api_reference.md) - Complete API documentation
