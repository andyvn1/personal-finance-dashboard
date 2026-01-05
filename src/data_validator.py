"""
Data validation module for checking transaction data quality.

This module provides comprehensive validation functionality to check data quality,
flag potential issues, and generate detailed validation reports.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Any
from enum import Enum

import pandas as pd

from .schema import SchemaConstraints, TransactionType

# Configure module logger
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    ERROR = "error"      # Critical issues that must be fixed
    WARNING = "warning"  # Issues that should be reviewed
    INFO = "info"        # Informational messages


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: ValidationSeverity
    field: str
    message: str
    row_indices: List[int] = field(default_factory=list)
    count: Optional[int] = None

    def __post_init__(self):
        """Set count based on row_indices if not explicitly provided."""
        if self.count is None:
            self.count = len(self.row_indices)


@dataclass
class ValidationResult:
    """
    Comprehensive validation results with warnings and errors.

    Provides detailed information about data quality issues found during validation.
    """

    total_rows: int
    valid_rows: int
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_issue(
        self,
        severity: ValidationSeverity,
        field: str,
        message: str,
        row_indices: Optional[List[int]] = None,
        count: Optional[int] = None
    ) -> None:
        """Add a validation issue to the results."""
        if row_indices is None:
            row_indices = []

        issue = ValidationIssue(
            severity=severity,
            field=field,
            message=message,
            row_indices=row_indices,
            count=count  # Will be set to len(row_indices) if None
        )
        self.issues.append(issue)
        logger.debug(f"[{severity.value.upper()}] {field}: {message} (count: {issue.count})")

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of info-level issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.INFO)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return self.error_count == 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warning_count > 0

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_field(self, field: str) -> List[ValidationIssue]:
        """Get all issues for a specific field."""
        return [issue for issue in self.issues if issue.field == field]

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "is_valid": self.is_valid,
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
            },
            "issues": [
                {
                    "severity": issue.severity.value,
                    "field": issue.field,
                    "message": issue.message,
                    "count": issue.count,
                    "sample_rows": issue.row_indices[:5] if issue.row_indices else [],
                }
                for issue in self.issues
            ],
        }

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Rows: {self.total_rows}")
        lines.append(f"Valid Rows: {self.valid_rows}")
        lines.append(f"Overall Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}")
        lines.append("")

        lines.append("SUMMARY:")
        lines.append(f"  Errors:   {self.error_count}")
        lines.append(f"  Warnings: {self.warning_count}")
        lines.append(f"  Info:     {self.info_count}")
        lines.append("")

        if self.issues:
            # Group by severity
            for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                issues = self.get_issues_by_severity(severity)
                if issues:
                    lines.append(f"{severity.value.upper()}S:")
                    for issue in issues:
                        lines.append(f"  [{issue.field}] {issue.message}")
                        if issue.count and issue.count > 0:
                            lines.append(f"    Affected rows: {issue.count}")
                        if issue.row_indices:
                            sample = issue.row_indices[:5]
                            lines.append(f"    Sample indices: {sample}")
                    lines.append("")
        else:
            lines.append("No issues found!")

        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class ValidationConfig:
    """Configuration for validation rules."""

    # Required fields validation
    required_fields: List[str] = field(default_factory=lambda: ["date", "description", "amount"])

    # Date validation
    allow_future_dates: bool = False
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None

    # Amount validation
    allow_zero_amounts: bool = False
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None

    # Duplicate detection
    check_duplicates: bool = True
    duplicate_threshold: int = 3  # Flag if more than N exact duplicates
    duplicate_window_days: Optional[int] = None  # Check duplicates within N days

    # Data type validation
    strict_types: bool = True

    # Custom validators
    custom_validators: Dict[str, callable] = field(default_factory=dict)


class DataValidator:
    """
    Comprehensive data validator for transaction data.

    Performs multiple validation checks and generates detailed reports.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()
        logger.info("Initialized DataValidator")

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive validation on DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with all issues found
        """
        logger.info(f"Starting validation of {len(df)} rows")

        result = ValidationResult(
            total_rows=len(df),
            valid_rows=len(df)  # Will be adjusted based on errors
        )

        # Run all validation checks
        self._check_required_fields(df, result)
        self._check_data_types(df, result)
        self._check_date_range(df, result)
        self._check_amount_values(df, result)
        self._check_duplicates(df, result)
        self._check_field_lengths(df, result)
        self._check_transaction_types(df, result)
        self._run_custom_validators(df, result)

        # Calculate valid rows (rows without errors)
        error_rows = set()
        for issue in result.get_issues_by_severity(ValidationSeverity.ERROR):
            error_rows.update(issue.row_indices)
        result.valid_rows = len(df) - len(error_rows)

        logger.info(
            f"Validation complete: {result.valid_rows}/{result.total_rows} valid rows, "
            f"{result.error_count} errors, {result.warning_count} warnings"
        )

        return result

    def _check_required_fields(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that all required fields are present."""
        missing_fields = [field for field in self.config.required_fields if field not in df.columns]

        if missing_fields:
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="schema",
                message=f"Missing required fields: {missing_fields}",
                count=len(missing_fields)
            )
        else:
            logger.debug("All required fields present")

    def _check_data_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate data type consistency."""
        # Check date column
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                result.add_issue(
                    severity=ValidationSeverity.ERROR,
                    field="date",
                    message="Date column is not datetime type",
                    count=1
                )
            else:
                # Check for NaT (Not a Time) values
                nat_mask = pd.isna(df['date'])
                if nat_mask.any():
                    nat_indices = df[nat_mask].index.tolist()
                    result.add_issue(
                        severity=ValidationSeverity.ERROR,
                        field="date",
                        message="Found invalid/missing dates (NaT)",
                        row_indices=nat_indices
                    )

        # Check amount column
        if 'amount' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['amount']):
                result.add_issue(
                    severity=ValidationSeverity.ERROR,
                    field="amount",
                    message="Amount column is not numeric type",
                    count=1
                )
            else:
                # Check for NaN values
                nan_mask = pd.isna(df['amount'])
                if nan_mask.any():
                    nan_indices = df[nan_mask].index.tolist()
                    result.add_issue(
                        severity=ValidationSeverity.ERROR,
                        field="amount",
                        message="Found NaN amount values",
                        row_indices=nan_indices
                    )

                # Check for zero amounts if not allowed
                if not self.config.allow_zero_amounts:
                    zero_mask = df['amount'] == 0
                    if zero_mask.any():
                        zero_indices = df[zero_mask].index.tolist()
                        result.add_issue(
                            severity=ValidationSeverity.WARNING,
                            field="amount",
                            message="Found zero amount transactions",
                            row_indices=zero_indices
                        )

        # Check description column
        if 'description' in df.columns:
            if df['description'].dtype != 'object':
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="description",
                    message="Description column is not string type",
                    count=1
                )
            else:
                # Check for empty descriptions
                empty_mask = df['description'].isna() | (df['description'].str.strip() == '')
                if empty_mask.any():
                    empty_indices = df[empty_mask].index.tolist()
                    result.add_issue(
                        severity=ValidationSeverity.ERROR,
                        field="description",
                        message="Found empty or missing descriptions",
                        row_indices=empty_indices
                    )

    def _check_date_range(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate date ranges."""
        if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            return

        now = datetime.now()

        # Check for future dates
        if not self.config.allow_future_dates:
            future_mask = df['date'] > pd.Timestamp(now)
            if future_mask.any():
                future_indices = df[future_mask].index.tolist()
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="date",
                    message=f"Found {len(future_indices)} future dates",
                    row_indices=future_indices
                )

        # Check minimum date
        if self.config.min_date:
            old_mask = df['date'] < pd.Timestamp(self.config.min_date)
            if old_mask.any():
                old_indices = df[old_mask].index.tolist()
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="date",
                    message=f"Found {len(old_indices)} dates before {self.config.min_date}",
                    row_indices=old_indices
                )

        # Check maximum date
        if self.config.max_date:
            new_mask = df['date'] > pd.Timestamp(self.config.max_date)
            if new_mask.any():
                new_indices = df[new_mask].index.tolist()
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="date",
                    message=f"Found {len(new_indices)} dates after {self.config.max_date}",
                    row_indices=new_indices
                )

        # Check for unrealistic years
        old_year_mask = df['date'].dt.year < SchemaConstraints.MIN_YEAR
        if old_year_mask.any():
            old_year_indices = df[old_year_mask].index.tolist()
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="date",
                message=f"Found dates with year < {SchemaConstraints.MIN_YEAR}",
                row_indices=old_year_indices
            )

        new_year_mask = df['date'].dt.year > SchemaConstraints.MAX_YEAR
        if new_year_mask.any():
            new_year_indices = df[new_year_mask].index.tolist()
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="date",
                message=f"Found dates with year > {SchemaConstraints.MAX_YEAR}",
                row_indices=new_year_indices
            )

    def _check_amount_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate amount values are reasonable."""
        if 'amount' not in df.columns or not pd.api.types.is_numeric_dtype(df['amount']):
            return

        # Check minimum amount
        min_amount = self.config.min_amount or SchemaConstraints.MIN_AMOUNT
        low_mask = df['amount'] < float(min_amount)
        if low_mask.any():
            low_indices = df[low_mask].index.tolist()
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="amount",
                message=f"Found {len(low_indices)} amounts below minimum ({min_amount})",
                row_indices=low_indices
            )

        # Check maximum amount
        max_amount = self.config.max_amount or SchemaConstraints.MAX_AMOUNT
        high_mask = df['amount'] > float(max_amount)
        if high_mask.any():
            high_indices = df[high_mask].index.tolist()
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="amount",
                message=f"Found {len(high_indices)} amounts above maximum ({max_amount})",
                row_indices=high_indices
            )

        # Check for suspiciously round numbers (might indicate data issues)
        if len(df) > 10:
            round_mask = (df['amount'] % 100 == 0) & (df['amount'].abs() >= 100)
            round_pct = round_mask.sum() / len(df)
            if round_pct > 0.5:  # More than 50% are round hundreds
                result.add_issue(
                    severity=ValidationSeverity.INFO,
                    field="amount",
                    message=f"{round_pct:.1%} of amounts are round hundreds (might indicate data quality issues)",
                    count=round_mask.sum()
                )

    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for suspicious duplicate transactions."""
        if not self.config.check_duplicates:
            return

        # Check for required columns
        duplicate_cols = []
        for col in ['date', 'description', 'amount']:
            if col in df.columns:
                duplicate_cols.append(col)

        if len(duplicate_cols) < 2:
            return

        # Find exact duplicates
        duplicated_mask = df.duplicated(subset=duplicate_cols, keep=False)
        if duplicated_mask.any():
            dup_count = duplicated_mask.sum()
            dup_indices = df[duplicated_mask].index.tolist()

            # Check if duplicate count exceeds threshold
            severity = (
                ValidationSeverity.WARNING
                if dup_count > self.config.duplicate_threshold
                else ValidationSeverity.INFO
            )

            result.add_issue(
                severity=severity,
                field="duplicates",
                message=f"Found {dup_count} potential duplicate transactions",
                row_indices=dup_indices
            )

        # Check for near-duplicates (same date/description, different amounts)
        if 'date' in df.columns and 'description' in df.columns and 'amount' in df.columns:
            df_temp = df.copy()
            df_temp['dup_key'] = df_temp['date'].astype(str) + '|' + df_temp['description'].astype(str)
            dup_keys = df_temp[df_temp['dup_key'].duplicated(keep=False)]

            # Group by dup_key and check if amounts differ
            near_dup_indices = []
            for key, group in dup_keys.groupby('dup_key'):
                if len(group) > 1 and group['amount'].nunique() > 1:
                    near_dup_indices.extend(group.index.tolist())

            if near_dup_indices:
                result.add_issue(
                    severity=ValidationSeverity.INFO,
                    field="duplicates",
                    message=f"Found {len(near_dup_indices)} near-duplicate transactions (same date/description, different amounts)",
                    row_indices=near_dup_indices
                )

    def _check_field_lengths(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that text fields don't exceed maximum lengths."""
        length_checks = {
            'description': SchemaConstraints.MAX_DESCRIPTION_LENGTH,
            'account': SchemaConstraints.MAX_ACCOUNT_LENGTH,
            'category': SchemaConstraints.MAX_CATEGORY_LENGTH,
            'merchant': SchemaConstraints.MAX_MERCHANT_LENGTH,
            'notes': SchemaConstraints.MAX_NOTES_LENGTH,
            'reference_id': SchemaConstraints.MAX_REFERENCE_ID_LENGTH,
        }

        for field, max_length in length_checks.items():
            if field in df.columns and df[field].dtype == 'object':
                # Get length of non-null values
                lengths = df[field].dropna().str.len()
                if len(lengths) > 0:
                    too_long_mask = lengths > max_length
                    if too_long_mask.any():
                        long_indices = lengths[too_long_mask].index.tolist()
                        result.add_issue(
                            severity=ValidationSeverity.WARNING,
                            field=field,
                            message=f"Found {len(long_indices)} values exceeding maximum length ({max_length})",
                            row_indices=long_indices
                        )

    def _check_transaction_types(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate transaction type consistency."""
        if 'transaction_type' not in df.columns:
            return

        # Check for invalid transaction types
        valid_types = {t.value for t in TransactionType}
        invalid_mask = ~df['transaction_type'].isin(valid_types) & df['transaction_type'].notna()
        if invalid_mask.any():
            invalid_indices = df[invalid_mask].index.tolist()
            invalid_values = df[invalid_mask]['transaction_type'].unique().tolist()
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                field="transaction_type",
                message=f"Found invalid transaction types: {invalid_values}",
                row_indices=invalid_indices
            )

        # Check consistency between amount sign and transaction type
        if 'amount' in df.columns:
            # Debits should be negative
            debit_positive_mask = (
                (df['transaction_type'] == TransactionType.DEBIT.value) &
                (df['amount'] > 0)
            )
            if debit_positive_mask.any():
                inconsistent_indices = df[debit_positive_mask].index.tolist()
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="transaction_type",
                    message=f"Found {len(inconsistent_indices)} DEBIT transactions with positive amounts",
                    row_indices=inconsistent_indices
                )

            # Credits should be positive
            credit_negative_mask = (
                (df['transaction_type'] == TransactionType.CREDIT.value) &
                (df['amount'] < 0)
            )
            if credit_negative_mask.any():
                inconsistent_indices = df[credit_negative_mask].index.tolist()
                result.add_issue(
                    severity=ValidationSeverity.WARNING,
                    field="transaction_type",
                    message=f"Found {len(inconsistent_indices)} CREDIT transactions with negative amounts",
                    row_indices=inconsistent_indices
                )

    def _run_custom_validators(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Run custom validation functions."""
        for field, validator_func in self.config.custom_validators.items():
            try:
                validator_func(df, result)
                logger.debug(f"Custom validator for '{field}' completed")
            except Exception as e:
                logger.error(f"Custom validator for '{field}' failed: {e}")
                result.add_issue(
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Custom validation failed: {str(e)}",
                    count=1
                )


def validate_dataframe(
    df: pd.DataFrame,
    config: Optional[ValidationConfig] = None
) -> ValidationResult:
    """
    Convenience function to validate a DataFrame.

    Args:
        df: DataFrame to validate
        config: Validation configuration (uses defaults if None)

    Returns:
        ValidationResult with all issues found

    Example:
        >>> result = validate_dataframe(df)
        >>> if not result.is_valid:
        ...     print(result.generate_report())
        >>> print(f"Valid rows: {result.valid_rows}/{result.total_rows}")
    """
    validator = DataValidator(config)
    return validator.validate(df)
