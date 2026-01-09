"""
Sample transaction data generator for testing and demos.

This module generates realistic anonymized transaction data with configurable
parameters, including edge cases and multiple bank format exports.
"""

import random
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for sample data generation."""

    num_transactions: int = 100
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=90))
    end_date: datetime = field(default_factory=datetime.now)
    include_edge_cases: bool = True
    edge_case_probability: float = 0.05  # 5% chance of edge cases
    duplicate_probability: float = 0.02  # 2% chance of duplicates
    missing_field_probability: float = 0.01  # 1% chance of missing fields


# Transaction categories with typical merchants and amount ranges
TRANSACTION_CATEGORIES = {
    "Groceries": {
        "merchants": ["Whole Foods", "Trader Joe's", "Safeway", "Kroger", "Walmart", "Costco"],
        "amount_range": (20.0, 150.0),
        "frequency": 0.20,  # 20% of transactions
    },
    "Restaurants": {
        "merchants": ["Starbucks", "McDonald's", "Chipotle", "Subway", "Pizza Hut", "Olive Garden"],
        "amount_range": (5.0, 75.0),
        "frequency": 0.15,
    },
    "Gas & Fuel": {
        "merchants": ["Shell", "Chevron", "Exxon", "BP", "Mobil"],
        "amount_range": (30.0, 80.0),
        "frequency": 0.08,
    },
    "Utilities": {
        "merchants": ["Electric Company", "Water Utility", "Gas Company", "Internet Provider"],
        "amount_range": (50.0, 200.0),
        "frequency": 0.05,
    },
    "Shopping": {
        "merchants": ["Amazon", "Target", "Best Buy", "Macy's", "Home Depot"],
        "amount_range": (15.0, 300.0),
        "frequency": 0.15,
    },
    "Entertainment": {
        "merchants": ["Netflix", "Spotify", "AMC Theaters", "iTunes", "Steam"],
        "amount_range": (10.0, 50.0),
        "frequency": 0.08,
    },
    "Healthcare": {
        "merchants": ["CVS Pharmacy", "Walgreens", "Dr. Smith's Office", "Dental Clinic"],
        "amount_range": (20.0, 250.0),
        "frequency": 0.05,
    },
    "Transportation": {
        "merchants": ["Uber", "Lyft", "Metro Transit", "Parking Garage"],
        "amount_range": (5.0, 50.0),
        "frequency": 0.08,
    },
    "Insurance": {
        "merchants": ["State Farm", "Geico", "Progressive", "Allstate"],
        "amount_range": (100.0, 500.0),
        "frequency": 0.03,
    },
    "Income": {
        "merchants": ["Employer Payroll", "Freelance Payment", "Investment Dividend"],
        "amount_range": (500.0, 5000.0),
        "frequency": 0.10,
        "is_income": True,
    },
    "Miscellaneous": {
        "merchants": ["ATM Withdrawal", "Bank Fee", "Venmo", "PayPal"],
        "amount_range": (5.0, 100.0),
        "frequency": 0.03,
    },
}


class SampleDataGenerator:
    """
    Generates realistic sample transaction data.

    Creates anonymized transaction data with realistic patterns, amounts,
    and edge cases for testing and demonstration purposes.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize generator with configuration.

        Args:
            config: Generator configuration (uses defaults if None)
        """
        self.config = config or GeneratorConfig()
        random.seed(42)  # For reproducible results
        logger.info(f"Initialized SampleDataGenerator for {self.config.num_transactions} transactions")

    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate sample transactions as a DataFrame.

        Returns:
            DataFrame with transaction data
        """
        logger.info("Generating sample transactions")

        transactions = []
        date_range = (self.config.end_date - self.config.start_date).days

        # Track some transactions for duplicates
        duplicate_pool = []

        for i in range(self.config.num_transactions):
            # Decide if this should be a duplicate
            if (self.config.include_edge_cases and
                duplicate_pool and
                random.random() < self.config.duplicate_probability):
                # Create a duplicate with slight variation
                base_txn = random.choice(duplicate_pool)
                transaction = base_txn.copy()
                # Maybe vary the date slightly
                if random.random() < 0.5:
                    transaction['date'] += timedelta(days=random.randint(0, 2))
            else:
                # Generate new transaction
                transaction = self._generate_single_transaction(date_range)

                # Add to duplicate pool (keep last 20)
                if len(duplicate_pool) < 20:
                    duplicate_pool.append(transaction.copy())

            # Apply edge cases
            if self.config.include_edge_cases:
                transaction = self._apply_edge_cases(transaction)

            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Generated {len(df)} transactions from {df['date'].min()} to {df['date'].max()}")
        return df

    def _generate_single_transaction(self, date_range: int) -> Dict:
        """Generate a single transaction."""
        # Select category based on frequency
        category = self._select_category()
        category_info = TRANSACTION_CATEGORIES[category]

        # Select merchant
        merchant = random.choice(category_info["merchants"])

        # Generate amount
        min_amt, max_amt = category_info["amount_range"]
        amount = round(random.uniform(min_amt, max_amt), 2)

        # Determine if income or expense
        is_income = category_info.get("is_income", False)
        if not is_income:
            amount = -amount  # Expenses are negative

        # Generate date (date only, no time for bank transactions)
        days_offset = random.randint(0, date_range)
        date = self.config.start_date + timedelta(days=days_offset)
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)

        transaction = {
            'date': date,
            'description': merchant,
            'amount': amount,
            'category': category,
            'merchant': merchant,
            'transaction_type': 'credit' if is_income else 'debit',
        }

        return transaction

    def _select_category(self) -> str:
        """Select a category based on frequency weights."""
        categories = list(TRANSACTION_CATEGORIES.keys())
        weights = [TRANSACTION_CATEGORIES[cat]["frequency"] for cat in categories]

        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(categories, weights=weights)[0]

    def _apply_edge_cases(self, transaction: Dict) -> Dict:
        """Apply edge cases to transaction for testing."""
        if random.random() > self.config.edge_case_probability:
            return transaction  # No edge case

        edge_case = random.choice([
            'round_amount',
            'zero_amount',
            'very_large_amount',
            'missing_category',
            'missing_merchant',
            'empty_description',
            'future_date',
        ])

        if edge_case == 'round_amount':
            # Make amount a round number
            transaction['amount'] = float(int(abs(transaction['amount']) / 100) * 100)
            if transaction['transaction_type'] == 'debit':
                transaction['amount'] = -transaction['amount']

        elif edge_case == 'zero_amount':
            transaction['amount'] = 0.0
            transaction['description'] = 'Zero Amount Transaction'

        elif edge_case == 'very_large_amount':
            transaction['amount'] = random.choice([5000.0, 10000.0, -5000.0, -10000.0])

        elif edge_case == 'missing_category':
            transaction['category'] = None

        elif edge_case == 'missing_merchant':
            transaction['merchant'] = None

        elif edge_case == 'empty_description':
            transaction['description'] = '   '  # Whitespace only

        elif edge_case == 'future_date':
            transaction['date'] = datetime.now() + timedelta(days=random.randint(1, 30))

        return transaction

    def export_to_csv(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format_type: str = 'generic'
    ) -> None:
        """
        Export DataFrame to CSV in specified bank format.

        Args:
            df: DataFrame to export
            output_path: Path to output CSV file
            format_type: Bank format ('generic', 'chase', 'bank_of_america', 'wells_fargo')
        """
        logger.info(f"Exporting to {output_path} in {format_type} format")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to appropriate format
        if format_type == 'generic':
            export_df = self._format_generic(df)
        elif format_type == 'chase':
            export_df = self._format_chase(df)
        elif format_type == 'bank_of_america':
            export_df = self._format_bank_of_america(df)
        elif format_type == 'wells_fargo':
            export_df = self._format_wells_fargo(df)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        # Export to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(export_df)} transactions to {output_path}")

    def _format_generic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format as generic CSV."""
        return pd.DataFrame({
            'Date': df['date'].dt.strftime('%Y-%m-%d'),
            'Description': df['description'],
            'Amount': df['amount'],
            'Category': df['category'],
            'Transaction Type': df['transaction_type'],
        })

    def _format_chase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format as Chase bank CSV."""
        return pd.DataFrame({
            'Transaction Date': df['date'].dt.strftime('%m/%d/%Y'),
            'Post Date': df['date'].dt.strftime('%m/%d/%Y'),
            'Description': df['description'],
            'Category': df['category'],
            'Type': df['transaction_type'],
            'Amount': df['amount'],
        })

    def _format_bank_of_america(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format as Bank of America CSV."""
        # Calculate running balance (simplified)
        balance = 1000.0  # Starting balance
        balances = []
        for amount in df['amount']:
            balance += amount
            balances.append(round(balance, 2))

        return pd.DataFrame({
            'Date': df['date'].dt.strftime('%m/%d/%Y'),
            'Description': df['description'],
            'Amount': df['amount'],
            'Running Bal.': balances,
        })

    def _format_wells_fargo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format as Wells Fargo CSV."""
        # Calculate running balance
        balance = 1000.0
        balances = []
        for amount in df['amount']:
            balance += amount
            balances.append(round(balance, 2))

        return pd.DataFrame({
            'Date': df['date'].dt.strftime('%m/%d/%Y'),
            'Amount': df['amount'],
            'Description': df['description'],
            'Balance': balances,
        })


def generate_sample_files(
    output_dir: Path,
    num_transactions: int = 100
) -> List[Path]:
    """
    Generate sample CSV files in multiple formats.

    Args:
        output_dir: Directory to save sample files
        num_transactions: Number of transactions to generate

    Returns:
        List of generated file paths
    """
    logger.info(f"Generating sample files in {output_dir}")

    # Create generator
    config = GeneratorConfig(num_transactions=num_transactions)
    generator = SampleDataGenerator(config)

    # Generate transactions
    df = generator.generate_transactions()

    # Export in different formats
    formats = ['generic', 'chase', 'bank_of_america', 'wells_fargo']
    generated_files = []

    for fmt in formats:
        output_path = output_dir / f"sample_transactions_{fmt}.csv"
        generator.export_to_csv(df, output_path, format_type=fmt)
        generated_files.append(output_path)

    logger.info(f"Generated {len(generated_files)} sample files")
    return generated_files
