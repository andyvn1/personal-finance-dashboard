"""
Data cleaning pipeline for financial transactions.

This module provides functionality to clean and normalize transaction data,
handling missing values, duplicates, and format standardization.
"""

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from .schema import SchemaConstraints, TransactionType

# Configure module logger
logger = logging.getLogger(__name__)


class CleaningError(Exception):
    """Base exception for data cleaning errors."""
    pass


class ValidationError(CleaningError):
    """Raised when data validation fails."""
    pass


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate transactions.

    Duplicates are identified by date + amount + description by default.

    Args:
        df: DataFrame with transactions
        subset: Columns to use for duplicate detection
        keep: Which duplicate to keep ('first', 'last', False)

    Returns:
        DataFrame with duplicates removed
    """
    original_count = len(df)

    if subset is None:
        # Use date, amount, and description for duplicate detection
        subset = []
        if 'date' in df.columns:
            subset.append('date')
        if 'amount' in df.columns:
            subset.append('amount')
        if 'description' in df.columns:
            subset.append('description')

    if not subset:
        logger.warning("No columns available for duplicate detection")
        return df

    # Remove duplicates
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    duplicates_removed = original_count - len(df_clean)

    if duplicates_removed > 0:
        logger.info(
            f"Removed {duplicates_removed} duplicate transactions "
            f"({duplicates_removed/original_count:.1%} of total)"
        )
    else:
        logger.info("No duplicates found")

    return df_clean.reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Handle missing values according to strategy.

    Args:
        df: DataFrame with potential missing values
        strategy: Dictionary mapping column names to strategies
                 Options: 'drop', 'fill:<value>', 'forward_fill', 'backward_fill'

    Returns:
        DataFrame with missing values handled
    """
    if strategy is None:
        # Default strategy
        strategy = {
            'date': 'drop',          # Drop rows without dates
            'description': 'fill:Unknown Transaction',
            'amount': 'drop',        # Drop rows without amounts
            'category': 'fill:Uncategorized',
            'account': 'fill:Unknown',
        }

    original_count = len(df)

    for column, action in strategy.items():
        if column not in df.columns:
            continue

        missing_count = df[column].isna().sum()
        if missing_count == 0:
            continue

        logger.debug(f"Handling {missing_count} missing values in '{column}'")
        current_count = len(df)

        if action == 'drop':
            df = df.dropna(subset=[column])
            rows_dropped = current_count - len(df)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with missing '{column}' values")
        elif action.startswith('fill:'):
            fill_value = action.split(':', 1)[1]
            df.loc[:, column] = df[column].fillna(fill_value)
            logger.debug(f"Filled {missing_count} values in '{column}' with '{fill_value}'")
        elif action == 'forward_fill':
            df.loc[:, column] = df[column].fillna(method='ffill')
            logger.debug(f"Forward filled {missing_count} values in '{column}'")
        elif action == 'backward_fill':
            df.loc[:, column] = df[column].fillna(method='bfill')
            logger.debug(f"Backward filled {missing_count} values in '{column}'")

    total_dropped = original_count - len(df)
    if total_dropped > 0:
        logger.info(
            f"Total rows dropped due to missing values: {total_dropped} "
            f"({total_dropped/original_count:.1%} of original)"
        )

    return df.reset_index(drop=True)


def normalize_amount(amount_str: Any) -> Optional[Decimal]:
    """
    Normalize amount string to Decimal.

    Handles:
    - Currency symbols ($, €, £, etc.)
    - Thousands separators (commas)
    - Parentheses for negative amounts: (100.00) -> -100.00
    - Whitespace

    Args:
        amount_str: Amount as string or number

    Returns:
        Decimal amount or None if invalid
    """
    if pd.isna(amount_str):
        return None

    # Already a number
    if isinstance(amount_str, (int, float, Decimal)):
        try:
            return Decimal(str(amount_str))
        except (InvalidOperation, ValueError):
            return None

    # Convert to string and clean
    amount_str = str(amount_str).strip()

    if not amount_str:
        return None

    # Check for parentheses (negative amount)
    is_negative = amount_str.startswith('(') and amount_str.endswith(')')
    if is_negative:
        amount_str = amount_str[1:-1]  # Remove parentheses

    # Remove currency symbols and whitespace
    amount_str = re.sub(r'[$€£¥₹]', '', amount_str)
    amount_str = amount_str.replace(' ', '')

    # Remove thousands separators (commas)
    amount_str = amount_str.replace(',', '')

    # Try to convert to Decimal
    try:
        amount = Decimal(amount_str)
        if is_negative:
            amount = -amount
        return amount
    except (InvalidOperation, ValueError):
        logger.warning(f"Could not parse amount: {amount_str}")
        return None


def normalize_amount_column(df: pd.DataFrame, column: str = 'amount') -> pd.DataFrame:
    """
    Normalize amount column to Decimal values.

    Args:
        df: DataFrame with amount column
        column: Name of amount column

    Returns:
        DataFrame with normalized amount column
    """
    if column not in df.columns:
        logger.warning(f"Amount column '{column}' not found")
        return df

    logger.debug(f"Normalizing amount column: {column}")

    # Apply normalization
    df[column] = df[column].apply(normalize_amount)

    # Count invalid amounts
    invalid_count = df[column].isna().sum()
    if invalid_count > 0:
        logger.warning(
            f"Found {invalid_count} invalid amounts that could not be parsed"
        )

    return df


def strip_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from text columns.

    Args:
        df: DataFrame with text columns
        columns: Specific columns to clean (all string columns if None)

    Returns:
        DataFrame with cleaned text
    """
    if columns is None:
        # Auto-detect string columns
        columns = df.select_dtypes(include=['object']).columns.tolist()

    total_changed = 0
    for column in columns:
        if column not in df.columns:
            continue

        if df[column].dtype == 'object':
            original = df[column].copy()
            df[column] = df[column].str.strip()

            # Count changes
            changed = (original != df[column]).sum()
            total_changed += changed

    if total_changed > 0:
        logger.info(f"Stripped whitespace from {total_changed} total values across all columns")

    return df


def normalize_description(description: str) -> str:
    """
    Normalize transaction description.

    - Strips whitespace
    - Removes extra spaces
    - Capitalizes consistently

    Args:
        description: Raw description

    Returns:
        Normalized description
    """
    if pd.isna(description):
        return "Unknown Transaction"

    # Convert to string and strip
    desc = str(description).strip()

    # Remove special characters that don't add value (replace with space)
    desc = re.sub(r'[#*]+', ' ', desc)

    # Remove extra whitespace (including spaces from removed special chars)
    desc = re.sub(r'\s+', ' ', desc).strip()

    # Capitalize first letter of each word for consistency
    # But preserve all-caps acronyms
    words = desc.split()
    normalized_words = []
    for word in words:
        if word.isupper() and len(word) > 1:
            # Preserve acronyms
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())

    return ' '.join(normalized_words)


def normalize_description_column(
    df: pd.DataFrame,
    column: str = 'description'
) -> pd.DataFrame:
    """
    Normalize description column.

    Args:
        df: DataFrame with description column
        column: Name of description column

    Returns:
        DataFrame with normalized descriptions
    """
    if column not in df.columns:
        logger.warning(f"Description column '{column}' not found")
        return df

    logger.debug(f"Normalizing description column: {column}")

    df[column] = df[column].apply(normalize_description)

    return df


def infer_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer transaction type from amount if not present.

    Negative amounts = DEBIT (expenses)
    Positive amounts = CREDIT (income)

    Args:
        df: DataFrame with amount column

    Returns:
        DataFrame with transaction_type column
    """
    if 'amount' not in df.columns:
        return df

    if 'transaction_type' in df.columns:
        # Already has transaction type
        missing = df['transaction_type'].isna().sum()
        if missing == 0:
            return df
        logger.debug(f"Inferring {missing} missing transaction types")
    else:
        logger.debug("Adding transaction_type column based on amount")
        df['transaction_type'] = None

    # Infer from amount sign
    def infer_type(row):
        if pd.notna(row.get('transaction_type')):
            return row['transaction_type']

        amount = row.get('amount')
        if pd.isna(amount):
            return None

        return TransactionType.DEBIT.value if amount < 0 else TransactionType.CREDIT.value

    df['transaction_type'] = df.apply(infer_type, axis=1)

    return df


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert data types according to schema.

    Args:
        df: DataFrame to validate

    Returns:
        DataFrame with corrected data types

    Raises:
        ValidationError: If critical validation fails
    """
    logger.debug("Validating data types")

    # Date column should be datetime
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
                logger.debug("Converted date column to datetime")
            except Exception as e:
                raise ValidationError(f"Failed to convert date column: {e}")

    # Amount should be numeric (Decimal converted to float for DataFrame)
    if 'amount' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['amount']):
            try:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                logger.debug("Converted amount column to numeric")
            except Exception as e:
                raise ValidationError(f"Failed to convert amount column: {e}")

        # Validate amount range
        if SchemaConstraints.MIN_AMOUNT is not None:
            out_of_range = (
                (df['amount'] < float(SchemaConstraints.MIN_AMOUNT)) |
                (df['amount'] > float(SchemaConstraints.MAX_AMOUNT))
            ).sum()
            if out_of_range > 0:
                logger.warning(f"{out_of_range} amounts are outside acceptable range")

    return df


def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates_flag: bool = True,
    handle_missing_flag: bool = True,
    normalize_amounts_flag: bool = True,
    normalize_text_flag: bool = True,
    validate_types_flag: bool = True
) -> pd.DataFrame:
    """
    Apply complete cleaning pipeline to DataFrame.

    This is the main cleaning function that orchestrates all cleaning steps.

    Args:
        df: Raw DataFrame from CSV loader
        remove_duplicates_flag: Whether to remove duplicates
        handle_missing_flag: Whether to handle missing values
        normalize_amounts_flag: Whether to normalize amount column
        normalize_text_flag: Whether to normalize text fields
        validate_types_flag: Whether to validate data types

    Returns:
        Cleaned DataFrame ready for analysis

    Raises:
        ValidationError: If critical validation fails
    """
    logger.info(f"Starting data cleaning pipeline. Input: {len(df)} rows")
    original_count = len(df)

    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Step 1: Strip whitespace from all text columns
    if normalize_text_flag:
        logger.debug("Step 1: Stripping whitespace")
        df_clean = strip_whitespace(df_clean)

    # Step 2: Normalize amounts
    if normalize_amounts_flag and 'amount' in df_clean.columns:
        logger.debug("Step 2: Normalizing amounts")
        df_clean = normalize_amount_column(df_clean)

    # Step 3: Normalize descriptions
    if normalize_text_flag and 'description' in df_clean.columns:
        logger.debug("Step 3: Normalizing descriptions")
        df_clean = normalize_description_column(df_clean)

    # Step 4: Handle missing values
    if handle_missing_flag:
        logger.debug("Step 4: Handling missing values")
        df_clean = handle_missing_values(df_clean)

    # Step 5: Remove duplicates
    if remove_duplicates_flag:
        logger.debug("Step 5: Removing duplicates")
        df_clean = remove_duplicates(df_clean)

    # Step 6: Infer transaction types
    logger.debug("Step 6: Inferring transaction types")
    df_clean = infer_transaction_type(df_clean)

    # Step 7: Validate data types
    if validate_types_flag:
        logger.debug("Step 7: Validating data types")
        df_clean = validate_data_types(df_clean)

    # Log summary
    final_count = len(df_clean)
    rows_removed = original_count - final_count
    logger.info(
        f"Data cleaning complete. Output: {final_count} rows "
        f"({rows_removed} rows removed, {rows_removed/original_count:.1%})"
    )

    # Log column info
    logger.debug(f"Columns: {list(df_clean.columns)}")
    logger.debug(f"Data types: {df_clean.dtypes.to_dict()}")

    return df_clean


# Convenience alias
clean_transactions = clean_dataframe
