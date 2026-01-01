"""
Tests for data_loader module.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import os

from src.data_loader import (
    load_csv,
    load_multiple_csv,
    load_transactions,
    detect_encoding,
    parse_date_column,
    map_columns,
    DataLoaderError,
    FileFormatError,
    EncodingError,
    DateParsingError,
    ColumnMappingError
)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_content = """Date,Description,Amount,Category
2025-01-15,Grocery Store,-75.50,Food
2025-01-14,Salary Deposit,3000.00,Income
2025-01-13,Electric Bill,-120.00,Utilities
"""
    file_path = tmp_path / "test_transactions.csv"
    file_path.write_text(csv_content)
    return file_path


@pytest.fixture
def sample_csv_different_format(tmp_path):
    """Create a CSV file with different date format."""
    csv_content = """Transaction Date,Memo,Amt
01/15/2025,Coffee Shop,-5.25
01/14/2025,Lunch,-12.00
"""
    file_path = tmp_path / "test_different.csv"
    file_path.write_text(csv_content)
    return file_path


class TestDetectEncoding:
    """Tests for encoding detection."""

    def test_detect_utf8_encoding(self, sample_csv_file):
        """Test detection of UTF-8 encoding."""
        encoding = detect_encoding(sample_csv_file)
        assert encoding.lower() in ['utf-8', 'ascii']  # ASCII is a subset of UTF-8

    def test_detect_encoding_nonexistent_file(self):
        """Test encoding detection with non-existent file."""
        with pytest.raises(EncodingError):
            detect_encoding("nonexistent_file.csv")


class TestParseDateColumn:
    """Tests for date parsing."""

    def test_parse_date_column_iso_format(self):
        """Test parsing ISO format dates."""
        df = pd.DataFrame({
            'date': ['2025-01-15', '2025-01-14', '2025-01-13']
        })
        result = parse_date_column(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'][0] == pd.Timestamp('2025-01-15')

    def test_parse_date_column_us_format(self):
        """Test parsing US date format (MM/DD/YYYY)."""
        df = pd.DataFrame({
            'date': ['01/15/2025', '01/14/2025', '01/13/2025']
        })
        result = parse_date_column(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_parse_date_column_missing(self):
        """Test error when date column is missing."""
        df = pd.DataFrame({'other': [1, 2, 3]})
        with pytest.raises(ColumnMappingError):
            parse_date_column(df, 'date')


class TestMapColumns:
    """Tests for column mapping."""

    def test_map_columns_custom_mapping(self):
        """Test custom column mapping."""
        df = pd.DataFrame({
            'Transaction Date': ['2025-01-15'],
            'Memo': ['Test'],
            'Amt': [100]
        })
        mapping = {
            'Transaction Date': 'date',
            'Memo': 'description',
            'Amt': 'amount'
        }
        result = map_columns(df, column_mapping=mapping)
        assert 'date' in result.columns
        assert 'description' in result.columns
        assert 'amount' in result.columns

    def test_map_columns_missing_required(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({'other': [1, 2, 3]})
        with pytest.raises(ColumnMappingError):
            map_columns(df)


class TestLoadCSV:
    """Tests for loading single CSV file."""

    def test_load_csv_basic(self, sample_csv_file):
        """Test basic CSV loading."""
        df = load_csv(sample_csv_file)
        assert len(df) == 3
        assert 'date' in df.columns
        assert 'description' in df.columns
        assert 'amount' in df.columns

    def test_load_csv_dates_parsed(self, sample_csv_file):
        """Test that dates are parsed correctly."""
        df = load_csv(sample_csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df['date'])

    def test_load_csv_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")

    def test_load_csv_empty_file(self, tmp_path):
        """Test loading empty CSV file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        with pytest.raises(FileFormatError):
            load_csv(empty_file)

    def test_load_csv_with_custom_mapping(self, sample_csv_different_format):
        """Test loading with custom column mapping."""
        mapping = {
            'Transaction Date': 'date',
            'Memo': 'description',
            'Amt': 'amount'
        }
        df = load_csv(sample_csv_different_format, column_mapping=mapping)
        assert 'date' in df.columns
        assert 'description' in df.columns
        assert 'amount' in df.columns
        assert len(df) == 2


class TestLoadMultipleCSV:
    """Tests for loading multiple CSV files."""

    def test_load_multiple_csv(self, sample_csv_file, tmp_path):
        """Test loading multiple CSV files."""
        # Create a second file
        csv_content2 = """Date,Description,Amount
2025-01-10,Restaurant,-25.00
2025-01-09,Taxi,-15.00
"""
        file2 = tmp_path / "test2.csv"
        file2.write_text(csv_content2)

        df = load_multiple_csv([sample_csv_file, file2])
        assert len(df) == 5  # 3 from first file + 2 from second
        assert 'date' in df.columns

    def test_load_multiple_csv_sorted_by_date(self, sample_csv_file, tmp_path):
        """Test that combined data is sorted by date."""
        csv_content2 = """Date,Description,Amount
2025-01-20,Future Transaction,-10.00
"""
        file2 = tmp_path / "test2.csv"
        file2.write_text(csv_content2)

        df = load_multiple_csv([sample_csv_file, file2])
        # Should be sorted by date
        assert df['date'].is_monotonic_increasing

    def test_load_multiple_csv_empty_list(self):
        """Test error when no files provided."""
        with pytest.raises(ValueError):
            load_multiple_csv([])


class TestLoadTransactions:
    """Tests for main load_transactions function."""

    def test_load_transactions_single_file(self, sample_csv_file):
        """Test loading a single file."""
        df = load_transactions(sample_csv_file)
        assert len(df) == 3
        assert 'date' in df.columns

    def test_load_transactions_multiple_files(self, sample_csv_file, tmp_path):
        """Test loading multiple files."""
        csv_content2 = """Date,Description,Amount
2025-01-10,Test,-10.00
"""
        file2 = tmp_path / "test2.csv"
        file2.write_text(csv_content2)

        df = load_transactions([sample_csv_file, file2])
        assert len(df) == 4

    def test_load_transactions_with_format_type(self, sample_csv_file):
        """Test loading with specific format type."""
        df = load_transactions(sample_csv_file, format_type='generic')
        assert 'date' in df.columns

    def test_load_transactions_data_sample(self):
        """Test loading the actual sample data file."""
        # This tests with the real sample file if it exists
        sample_path = Path('data/sample_transactions.csv')
        if sample_path.exists():
            df = load_transactions(sample_path)
            assert len(df) > 0
            assert 'date' in df.columns
            assert 'description' in df.columns
            assert 'amount' in df.columns
