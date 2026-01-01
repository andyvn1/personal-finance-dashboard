"""
Tests for data_cleaner module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal

from src.data_cleaner import (
    remove_duplicates,
    handle_missing_values,
    normalize_amount,
    normalize_amount_column,
    strip_whitespace,
    normalize_description,
    normalize_description_column,
    infer_transaction_type,
    validate_data_types,
    clean_dataframe,
    CleaningError,
    ValidationError
)


class TestRemoveDuplicates:
    """Tests for duplicate removal."""

    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate transactions."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-15', '2025-01-14']),
            'description': ['Coffee Shop', 'Coffee Shop', 'Lunch'],
            'amount': [-5.25, -5.25, -12.00]
        })
        result = remove_duplicates(df)
        assert len(result) == 2  # One duplicate removed

    def test_remove_duplicates_keep_last(self):
        """Test keeping last duplicate instead of first."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-15']),
            'description': ['Coffee', 'Coffee'],
            'amount': [-5.00, -5.00]
        })
        result = remove_duplicates(df, keep='last')
        assert len(result) == 1

    def test_no_duplicates(self):
        """Test when there are no duplicates."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', '2025-01-14']),
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.00, -12.00]
        })
        result = remove_duplicates(df)
        assert len(result) == 2  # No change


class TestNormalizeAmount:
    """Tests for amount normalization."""

    def test_normalize_plain_number(self):
        """Test normalizing plain number string."""
        assert normalize_amount("100.50") == Decimal("100.50")
        assert normalize_amount("-50.25") == Decimal("-50.25")

    def test_normalize_with_currency_symbol(self):
        """Test normalizing amount with currency symbols."""
        assert normalize_amount("$100.50") == Decimal("100.50")
        assert normalize_amount("€50.25") == Decimal("50.25")
        assert normalize_amount("£75.00") == Decimal("75.00")

    def test_normalize_with_thousands_separator(self):
        """Test normalizing amount with commas."""
        assert normalize_amount("1,234.56") == Decimal("1234.56")
        assert normalize_amount("$10,000.00") == Decimal("10000.00")

    def test_normalize_parentheses_negative(self):
        """Test normalizing negative amount in parentheses."""
        assert normalize_amount("(100.00)") == Decimal("-100.00")
        assert normalize_amount("($50.25)") == Decimal("-50.25")

    def test_normalize_already_number(self):
        """Test normalizing already numeric values."""
        assert normalize_amount(100.50) == Decimal("100.50")
        assert normalize_amount(Decimal("50.25")) == Decimal("50.25")

    def test_normalize_invalid_amount(self):
        """Test handling invalid amounts."""
        assert normalize_amount("invalid") is None
        assert normalize_amount("") is None
        assert normalize_amount(None) is None


class TestNormalizeAmountColumn:
    """Tests for amount column normalization."""

    def test_normalize_amount_column_mixed_formats(self):
        """Test normalizing column with mixed formats."""
        df = pd.DataFrame({
            'amount': ['$100.50', '(50.00)', '1,234.56', '75']
        })
        result = normalize_amount_column(df)
        assert result['amount'][0] == Decimal("100.50")
        assert result['amount'][1] == Decimal("-50.00")
        assert result['amount'][2] == Decimal("1234.56")
        assert result['amount'][3] == Decimal("75")


class TestStripWhitespace:
    """Tests for whitespace stripping."""

    def test_strip_whitespace_from_strings(self):
        """Test stripping whitespace from text columns."""
        df = pd.DataFrame({
            'description': ['  Coffee  ', 'Lunch', '  Dinner  '],
            'amount': [-5.00, -12.00, -25.00]
        })
        result = strip_whitespace(df)
        assert result['description'][0] == 'Coffee'
        assert result['description'][2] == 'Dinner'

    def test_strip_whitespace_specific_columns(self):
        """Test stripping whitespace from specific columns only."""
        df = pd.DataFrame({
            'col1': ['  a  ', '  b  '],
            'col2': ['  c  ', '  d  ']
        })
        result = strip_whitespace(df, columns=['col1'])
        assert result['col1'][0] == 'a'
        assert result['col2'][0] == '  c  '  # Not stripped


class TestNormalizeDescription:
    """Tests for description normalization."""

    def test_normalize_description_basic(self):
        """Test basic description normalization."""
        assert normalize_description("  coffee shop  ") == "Coffee Shop"
        assert normalize_description("GROCERY STORE") == "GROCERY STORE"  # Preserve acronym-like

    def test_normalize_description_extra_spaces(self):
        """Test removing extra spaces."""
        assert normalize_description("Coffee    Shop") == "Coffee Shop"

    def test_normalize_description_special_chars(self):
        """Test removing special characters."""
        assert normalize_description("Coffee #Shop") == "Coffee Shop"
        assert normalize_description("Store***Name") == "Store Name"

    def test_normalize_description_none(self):
        """Test handling None/NaN descriptions."""
        assert normalize_description(None) == "Unknown Transaction"
        assert normalize_description(np.nan) == "Unknown Transaction"


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_handle_missing_with_drop(self):
        """Test dropping rows with missing values."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', None, '2025-01-13']),
            'amount': [-5.00, -10.00, -15.00]
        })
        strategy = {'date': 'drop'}
        result = handle_missing_values(df, strategy)
        assert len(result) == 2  # One row dropped

    def test_handle_missing_with_fill(self):
        """Test filling missing values."""
        df = pd.DataFrame({
            'description': ['Coffee', None, 'Lunch'],
            'amount': [-5.00, -10.00, -15.00]
        })
        strategy = {'description': 'fill:Unknown'}
        result = handle_missing_values(df, strategy)
        assert result['description'][1] == 'Unknown'
        assert len(result) == 3  # No rows dropped

    def test_handle_missing_default_strategy(self):
        """Test default missing value strategy."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', None]),
            'description': ['Coffee', None],
            'amount': [-5.00, None],
            'category': ['Food', None]
        })
        result = handle_missing_values(df)
        # Should drop rows with missing date or amount
        assert len(result) == 1


class TestInferTransactionType:
    """Tests for transaction type inference."""

    def test_infer_from_negative_amount(self):
        """Test inferring DEBIT from negative amount."""
        df = pd.DataFrame({
            'amount': [-50.00, -25.00]
        })
        result = infer_transaction_type(df)
        assert result['transaction_type'][0] == 'debit'
        assert result['transaction_type'][1] == 'debit'

    def test_infer_from_positive_amount(self):
        """Test inferring CREDIT from positive amount."""
        df = pd.DataFrame({
            'amount': [1000.00, 500.00]
        })
        result = infer_transaction_type(df)
        assert result['transaction_type'][0] == 'credit'
        assert result['transaction_type'][1] == 'credit'

    def test_preserve_existing_transaction_type(self):
        """Test that existing transaction types are preserved."""
        df = pd.DataFrame({
            'amount': [-50.00, 100.00],
            'transaction_type': ['debit', 'credit']
        })
        result = infer_transaction_type(df)
        assert result['transaction_type'][0] == 'debit'
        assert result['transaction_type'][1] == 'credit'


class TestValidateDataTypes:
    """Tests for data type validation."""

    def test_validate_date_column(self):
        """Test date column validation and conversion."""
        df = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-14']
        })
        result = validate_data_types(df)
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_validate_amount_column(self):
        """Test amount column validation."""
        df = pd.DataFrame({
            'amount': ['100.50', '50.25']
        })
        result = validate_data_types(df)
        assert pd.api.types.is_numeric_dtype(result['amount'])

    def test_validate_amount_already_numeric(self):
        """Test validation of already numeric amount."""
        df = pd.DataFrame({
            'amount': [100.50, 50.25]
        })
        result = validate_data_types(df)
        assert pd.api.types.is_numeric_dtype(result['amount'])


class TestCleanDataFrame:
    """Tests for complete cleaning pipeline."""

    def test_clean_dataframe_basic(self):
        """Test basic cleaning pipeline."""
        df = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-14', '2025-01-15'],
            'description': ['  Coffee  ', 'Lunch', '  Coffee  '],
            'amount': ['$5.50', '-12.00', '$5.50']
        })
        result = clean_dataframe(df)

        # Should have duplicates removed
        assert len(result) == 2

        # Descriptions should be cleaned
        assert result['description'][0] == 'Coffee'

        # Dates should be datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_clean_dataframe_with_missing_values(self):
        """Test cleaning with missing values."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-15', None, '2025-01-13']),
            'description': ['Coffee', 'Lunch', 'Dinner'],
            'amount': [-5.00, -10.00, -15.00]
        })
        result = clean_dataframe(df)

        # Row with missing date should be dropped
        assert len(result) == 2

    def test_clean_dataframe_preserve_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'date': ['2025-01-15'],
            'description': ['  Coffee  '],
            'amount': ['$5.50']
        })
        original_desc = df['description'][0]
        clean_dataframe(df)

        # Original should be unchanged
        assert df['description'][0] == original_desc

    def test_clean_dataframe_all_flags_false(self):
        """Test cleaning with all flags disabled."""
        df = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-15'],
            'description': ['Coffee', 'Coffee'],
            'amount': [-5.00, -5.00]
        })
        result = clean_dataframe(
            df,
            remove_duplicates_flag=False,
            handle_missing_flag=False,
            normalize_amounts_flag=False,
            normalize_text_flag=False,
            validate_types_flag=False
        )

        # No cleaning should occur
        assert len(result) == 2  # Duplicates not removed

    def test_clean_dataframe_complex_scenario(self):
        """Test cleaning with complex, realistic data."""
        df = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-15', '2025-01-14', None],
            'description': ['  STARBUCKS  ', '  STARBUCKS  ', 'whole foods', 'Unknown'],
            'amount': ['($5.50)', '($5.50)', '$75.25', '-10.00'],
            'category': [None, None, 'Groceries', 'Food']
        })
        result = clean_dataframe(df)

        # Should remove duplicate and row with missing date
        assert len(result) == 2

        # Amounts should be normalized
        assert float(result['amount'].iloc[0]) == -5.50
        assert float(result['amount'].iloc[1]) == 75.25

        # Descriptions should be normalized (STARBUCKS preserved as all-caps)
        assert result['description'].iloc[0] == 'STARBUCKS'

        # Categories should be filled
        assert result['category'].iloc[0] == 'Uncategorized'
