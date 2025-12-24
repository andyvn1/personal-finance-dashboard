"""
CSV data loader for financial transactions.

This module provides functionality to load transaction data from CSV files
with support for multiple bank formats, automatic encoding detection,
and flexible date parsing.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Union
from decimal import Decimal
from datetime import datetime

import pandas as pd
import chardet
from dateutil import parser as date_parser

from .schema import CSV_FIELD_MAPPINGS, EXPECTED_CSV_FORMATS

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass


class FileFormatError(DataLoaderError):
    """Raised when file format is invalid."""
    pass


class EncodingError(DataLoaderError):
    """Raised when encoding detection fails."""
    pass


class DateParsingError(DataLoaderError):
    """Raised when date parsing fails."""
    pass


class ColumnMappingError(DataLoaderError):
    """Raised when required columns are missing."""
    pass


def detect_encoding(file_path: Union[str, Path], sample_size: int = 10000) -> str:
    """
    Detect file encoding automatically.

    Args:
        file_path: Path to the file
        sample_size: Number of bytes to sample for detection

    Returns:
        Detected encoding name (e.g., 'utf-8', 'latin-1')

    Raises:
        EncodingError: If encoding detection fails
    """
    file_path = Path(file_path)
    logger.debug(f"Detecting encoding for: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)

            encoding = result['encoding']
            confidence = result['confidence']

            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")

            if encoding is None:
                raise EncodingError(f"Could not detect encoding for file: {file_path}")

            return encoding

    except IOError as e:
        raise EncodingError(f"Error reading file for encoding detection: {e}")


def parse_date_column(
    df: pd.DataFrame,
    column_name: str,
    date_formats: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Parse date column with multiple format support.

    Args:
        df: DataFrame containing the date column
        column_name: Name of the date column
        date_formats: List of date formats to try (optional)

    Returns:
        DataFrame with parsed date column

    Raises:
        DateParsingError: If date parsing fails
    """
    if column_name not in df.columns:
        raise ColumnMappingError(f"Date column '{column_name}' not found in DataFrame")

    logger.debug(f"Parsing date column: {column_name}")

    # Common date formats to try
    if date_formats is None:
        date_formats = [
            '%Y-%m-%d',          # 2025-01-15
            '%m/%d/%Y',          # 01/15/2025
            '%d/%m/%Y',          # 15/01/2025
            '%Y/%m/%d',          # 2025/01/15
            '%m-%d-%Y',          # 01-15-2025
            '%d-%m-%Y',          # 15-01-2025
            '%m/%d/%y',          # 01/15/25
            '%d/%m/%y',          # 15/01/25
        ]

    # Try pandas built-in parsing first (fastest)
    try:
        df[column_name] = pd.to_datetime(df[column_name], infer_datetime_format=True)
        logger.info(f"Successfully parsed dates using pandas auto-detection")
        return df
    except Exception:
        logger.debug("Pandas auto-detection failed, trying explicit formats")

    # Try each format explicitly
    for fmt in date_formats:
        try:
            df[column_name] = pd.to_datetime(df[column_name], format=fmt)
            logger.info(f"Successfully parsed dates using format: {fmt}")
            return df
        except Exception:
            continue

    # Last resort: use dateutil parser (slower but very flexible)
    try:
        df[column_name] = df[column_name].apply(
            lambda x: date_parser.parse(str(x)) if pd.notna(x) else pd.NaT
        )
        logger.info("Successfully parsed dates using dateutil parser")
        return df
    except Exception as e:
        raise DateParsingError(
            f"Failed to parse date column '{column_name}'. "
            f"Please ensure dates are in a recognizable format. Error: {e}"
        )


def map_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    format_type: str = 'generic'
) -> pd.DataFrame:
    """
    Map CSV columns to standardized names.

    Args:
        df: DataFrame with original column names
        column_mapping: Custom column mapping dictionary
        format_type: Predefined format type ('generic', 'chase', 'bank_of_america', etc.)

    Returns:
        DataFrame with standardized column names

    Raises:
        ColumnMappingError: If required columns cannot be mapped
    """
    logger.debug(f"Mapping columns using format: {format_type}")

    # Use custom mapping if provided, otherwise use predefined mapping
    if column_mapping:
        mapping = column_mapping
    elif format_type in CSV_FIELD_MAPPINGS:
        # Build mapping from predefined format
        mapping = {}
        format_config = CSV_FIELD_MAPPINGS[format_type]

        for standard_name, possible_names in format_config.items():
            for col_name in df.columns:
                if col_name in possible_names:
                    mapping[col_name] = standard_name
                    break
    else:
        logger.warning(f"Unknown format type: {format_type}, using generic mapping")
        mapping = {}

    # Apply the mapping
    if mapping:
        df = df.rename(columns=mapping)
        logger.info(f"Mapped columns: {mapping}")

    # Verify required columns exist
    required_columns = ['date', 'description', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        available_cols = list(df.columns)
        raise ColumnMappingError(
            f"Required columns missing: {missing_columns}. "
            f"Available columns: {available_cols}. "
            f"Please provide a custom column_mapping or check the format_type."
        )

    return df


def load_csv(
    file_path: Union[str, Path],
    encoding: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    format_type: str = 'generic',
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load a single CSV file with automatic encoding detection and date parsing.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding (auto-detected if None)
        column_mapping: Custom column name mapping
        format_type: Predefined format type for column mapping
        parse_dates: Whether to automatically parse date columns

    Returns:
        DataFrame with standardized column names and parsed dates

    Raises:
        FileNotFoundError: If file doesn't exist
        FileFormatError: If file format is invalid
        EncodingError: If encoding detection fails
        ColumnMappingError: If required columns are missing
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading CSV file: {file_path}")

    # Detect encoding if not provided
    if encoding is None:
        encoding = detect_encoding(file_path)

    # Read CSV file
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Loaded {len(df)} rows from {file_path.name}")
    except pd.errors.EmptyDataError:
        raise FileFormatError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise FileFormatError(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise DataLoaderError(f"Unexpected error loading CSV: {e}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Map columns to standard names
    df = map_columns(df, column_mapping, format_type)

    # Parse date column if present
    if parse_dates and 'date' in df.columns:
        df = parse_date_column(df, 'date')

    return df


def load_multiple_csv(
    file_paths: List[Union[str, Path]],
    encoding: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    format_type: str = 'generic',
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.

    Args:
        file_paths: List of paths to CSV files
        encoding: File encoding (auto-detected if None)
        column_mapping: Custom column name mapping
        format_type: Predefined format type for column mapping
        parse_dates: Whether to automatically parse date columns

    Returns:
        Combined DataFrame with all transactions

    Raises:
        ValueError: If file_paths is empty
        DataLoaderError: If any file fails to load
    """
    if not file_paths:
        raise ValueError("No files provided to load")

    logger.info(f"Loading {len(file_paths)} CSV files")

    dataframes = []
    errors = []

    for file_path in file_paths:
        try:
            df = load_csv(
                file_path=file_path,
                encoding=encoding,
                column_mapping=column_mapping,
                format_type=format_type,
                parse_dates=parse_dates
            )
            dataframes.append(df)
        except Exception as e:
            error_msg = f"Error loading {file_path}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    if not dataframes:
        raise DataLoaderError(
            f"Failed to load any files. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            f"Successfully loaded {len(dataframes)} out of {len(file_paths)} files. "
            f"Failed: {len(errors)}"
        )

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total rows from {len(dataframes)} files")

    # Sort by date if date column exists
    if 'date' in combined_df.columns:
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        logger.debug("Sorted combined data by date")

    return combined_df


def load_transactions(
    file_path: Union[str, Path, List[Union[str, Path]]],
    encoding: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    format_type: str = 'generic'
) -> pd.DataFrame:
    """
    Convenient wrapper to load transaction data from CSV file(s).

    This is the main public API for loading transactions. It handles both
    single files and multiple files automatically.

    Args:
        file_path: Path to CSV file or list of paths
        encoding: File encoding (auto-detected if None)
        column_mapping: Custom column name mapping
        format_type: Predefined format type ('generic', 'chase', 'bank_of_america', etc.)

    Returns:
        DataFrame with standardized columns and parsed dates

    Raises:
        FileNotFoundError: If file(s) don't exist
        FileFormatError: If file format is invalid
        DataLoaderError: If loading fails

    Example:
        >>> # Load a single file
        >>> df = load_transactions('data/transactions.csv')

        >>> # Load multiple files
        >>> df = load_transactions(['data/jan.csv', 'data/feb.csv'])

        >>> # Load with custom mapping
        >>> mapping = {'Date': 'date', 'Description': 'description', 'Amount': 'amount'}
        >>> df = load_transactions('data/custom.csv', column_mapping=mapping)

        >>> # Load Chase bank format
        >>> df = load_transactions('data/chase.csv', format_type='chase')
    """
    # Handle both single file and multiple files
    if isinstance(file_path, (list, tuple)):
        return load_multiple_csv(
            file_paths=file_path,
            encoding=encoding,
            column_mapping=column_mapping,
            format_type=format_type
        )
    else:
        return load_csv(
            file_path=file_path,
            encoding=encoding,
            column_mapping=column_mapping,
            format_type=format_type
        )
