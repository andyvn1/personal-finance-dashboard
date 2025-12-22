"""
Transaction data schema definitions and validation.

This module defines the standardized internal data schema for financial transactions,
including field definitions, data types, constraints, and validation logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class TransactionType(Enum):
    """Transaction type enumeration."""

    DEBIT = "debit"  # Money out (expenses, withdrawals)
    CREDIT = "credit"  # Money in (income, deposits)


class AccountType(Enum):
    """Account type enumeration."""

    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    CASH = "cash"
    INVESTMENT = "investment"
    OTHER = "other"


# Standard transaction categories
STANDARD_CATEGORIES = [
    "Income",
    "Salary",
    "Investment Income",
    "Food & Dining",
    "Groceries",
    "Restaurants",
    "Transportation",
    "Gas & Fuel",
    "Public Transit",
    "Shopping",
    "Clothing",
    "Electronics",
    "Home & Garden",
    "Utilities",
    "Electricity",
    "Water",
    "Internet",
    "Phone",
    "Entertainment",
    "Movies & TV",
    "Music",
    "Healthcare",
    "Insurance",
    "Health Insurance",
    "Car Insurance",
    "Home Insurance",
    "Education",
    "Personal Care",
    "Travel",
    "Fees & Charges",
    "Bank Fees",
    "ATM Fees",
    "Taxes",
    "Gifts & Donations",
    "Uncategorized",
]


@dataclass
class Transaction:
    """
    Standardized transaction data model.

    Required Fields (minimum for a valid transaction):
        date: Transaction date (datetime object)
        description: Transaction description or memo
        amount: Transaction amount (Decimal for precise calculations)

    Recommended Fields:
        account: Account identifier or name
        transaction_type: Type of transaction (DEBIT or CREDIT)

    Optional Fields:
        category: Transaction category (for spending analysis)
        merchant: Merchant or payee name
        notes: Additional notes or comments
        tags: List of tags for custom organization
        reference_id: External reference ID (from bank/CSV)
        balance: Account balance after transaction
        original_description: Original unprocessed description
    """

    # Required fields (absolute minimum)
    date: datetime
    description: str
    amount: Decimal

    # Recommended fields (can be inferred if missing)
    account: str = "Unknown"
    transaction_type: Optional[TransactionType] = None

    # Optional fields
    category: Optional[str] = None
    merchant: Optional[str] = None
    notes: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    reference_id: Optional[str] = None
    balance: Optional[Decimal] = None
    original_description: Optional[str] = None

    def __post_init__(self) -> None:
        """Process transaction data after initialization."""
        # Auto-detect transaction type if not provided
        if self.transaction_type is None:
            self.transaction_type = (
                TransactionType.DEBIT if self.amount < 0 else TransactionType.CREDIT
            )

        # Perform validation
        self._validate()

    def _validate(self) -> None:
        """
        Validate transaction fields with flexible rules.

        Raises:
            ValueError: If any field fails validation.
        """
        # Validate date (required)
        if not isinstance(self.date, datetime):
            raise ValueError(f"Invalid date type: {type(self.date)}")

        # Validate description (required)
        if not self.description or not self.description.strip():
            raise ValueError("Description cannot be empty")

        if len(self.description) > SchemaConstraints.MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Description exceeds maximum length of {SchemaConstraints.MAX_DESCRIPTION_LENGTH} characters"
            )

        # Validate amount (required)
        if not isinstance(self.amount, Decimal):
            raise ValueError(f"Invalid amount type: {type(self.amount)}")

        if self.amount == 0:
            raise ValueError("Amount cannot be zero")

        if not SchemaConstraints.validate_amount(self.amount):
            raise ValueError("Amount exceeds acceptable range")

        # Validate account (flexible)
        if self.account and len(self.account) > SchemaConstraints.MAX_ACCOUNT_LENGTH:
            raise ValueError(
                f"Account name exceeds maximum length of {SchemaConstraints.MAX_ACCOUNT_LENGTH} characters"
            )

        # Validate transaction type (should be set by __post_init__)
        if self.transaction_type and not isinstance(
            self.transaction_type, TransactionType
        ):
            raise ValueError(f"Invalid transaction type: {self.transaction_type}")

        # Validate optional fields only if they exist
        if self.category and len(self.category) > SchemaConstraints.MAX_CATEGORY_LENGTH:
            raise ValueError(
                f"Category exceeds maximum length of {SchemaConstraints.MAX_CATEGORY_LENGTH} characters"
            )

        if self.merchant and len(self.merchant) > SchemaConstraints.MAX_MERCHANT_LENGTH:
            raise ValueError(
                f"Merchant name exceeds maximum length of {SchemaConstraints.MAX_MERCHANT_LENGTH} characters"
            )

        if self.notes and len(self.notes) > SchemaConstraints.MAX_NOTES_LENGTH:
            raise ValueError(
                f"Notes exceed maximum length of {SchemaConstraints.MAX_NOTES_LENGTH} characters"
            )

        if (
            self.reference_id
            and len(self.reference_id) > SchemaConstraints.MAX_REFERENCE_ID_LENGTH
        ):
            raise ValueError(
                f"Reference ID exceeds maximum length of {SchemaConstraints.MAX_REFERENCE_ID_LENGTH} characters"
            )

        if len(self.tags) > SchemaConstraints.MAX_TAGS:
            raise ValueError(f"Number of tags exceeds maximum of {SchemaConstraints.MAX_TAGS}")

    def to_dict(self) -> dict:
        """
        Convert transaction to dictionary.

        Returns:
            Dictionary representation of the transaction.
        """
        return {
            "date": self.date.isoformat(),
            "description": self.description,
            "amount": str(self.amount),
            "account": self.account,
            "transaction_type": self.transaction_type.value if self.transaction_type else None,
            "category": self.category,
            "merchant": self.merchant,
            "notes": self.notes,
            "tags": self.tags,
            "reference_id": self.reference_id,
            "balance": str(self.balance) if self.balance else None,
            "original_description": self.original_description,
        }

    @property
    def is_expense(self) -> bool:
        """Check if transaction is an expense (debit)."""
        return self.transaction_type == TransactionType.DEBIT

    @property
    def is_income(self) -> bool:
        """Check if transaction is income (credit)."""
        return self.transaction_type == TransactionType.CREDIT

    @property
    def absolute_amount(self) -> Decimal:
        """Get absolute value of amount."""
        return abs(self.amount)


# Schema constraints
class SchemaConstraints:
    """Transaction schema field constraints."""

    # Field length constraints
    MAX_DESCRIPTION_LENGTH = 500
    MAX_ACCOUNT_LENGTH = 100
    MAX_CATEGORY_LENGTH = 100
    MAX_MERCHANT_LENGTH = 200
    MAX_NOTES_LENGTH = 1000
    MAX_REFERENCE_ID_LENGTH = 100
    MAX_TAGS = 20
    MAX_TAG_LENGTH = 50

    # Amount constraints
    MIN_AMOUNT = Decimal("-999999999.99")
    MAX_AMOUNT = Decimal("999999999.99")

    # Date constraints
    MIN_YEAR = 1900
    MAX_YEAR = 2100

    @classmethod
    def validate_amount(cls, amount: Decimal) -> bool:
        """Validate amount is within acceptable range."""
        return cls.MIN_AMOUNT <= amount <= cls.MAX_AMOUNT

    @classmethod
    def validate_date(cls, date: datetime) -> bool:
        """Validate date is within acceptable range."""
        return cls.MIN_YEAR <= date.year <= cls.MAX_YEAR


# CSV field mappings for common bank formats
CSV_FIELD_MAPPINGS = {
    "generic": {
        "date": ["date", "transaction_date", "posted_date", "Date"],
        "description": ["description", "memo", "details", "Description"],
        "amount": ["amount", "Amount"],
        "category": ["category", "Category"],
        "account": ["account", "Account"],
    },
    "chase": {
        "date": ["Transaction Date", "Post Date"],
        "description": ["Description"],
        "amount": ["Amount"],
        "category": ["Category", "Type"],
        "account": ["Details"],
    },
    "bank_of_america": {
        "date": ["Date", "Posted Date"],
        "description": ["Description", "Payee"],
        "amount": ["Amount"],
        "balance": ["Running Bal."],
        "reference_id": ["Reference Number"],
    },
    "wells_fargo": {
        "date": ["Date"],
        "description": ["Description"],
        "amount": ["Amount"],
        "balance": ["Balance"],
    },
    "amex": {
        "date": ["Date"],
        "description": ["Description"],
        "amount": ["Amount"],
        "category": ["Category"],
        "reference_id": ["Reference"],
    },
}


# Expected CSV headers for different formats
EXPECTED_CSV_FORMATS = {
    "standard": [
        "Date",
        "Description",
        "Amount",
        "Category",
        "Account",
        "Transaction Type",
    ],
    "minimal": ["Date", "Description", "Amount"],
    "detailed": [
        "Date",
        "Description",
        "Amount",
        "Category",
        "Merchant",
        "Account",
        "Transaction Type",
        "Notes",
        "Tags",
    ],
}
