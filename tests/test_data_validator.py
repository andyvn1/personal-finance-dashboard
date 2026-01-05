"""
Tests for the data validation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.data_validator import (
    DataValidator,
    ValidationConfig,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_dataframe,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_count_from_row_indices(self):
        """Test that count is calculated from row_indices if not provided."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="test",
            message="Test message",
            row_indices=[1, 2, 3, 4, 5]
        )
        assert issue.count == 5

    def test_explicit_count(self):
        """Test that explicit count is preserved."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="test",
            message="Test message",
            row_indices=[1, 2, 3],
            count=10
        )
        assert issue.count == 10

    def test_zero_count_explicit(self):
        """Test that explicit zero count is preserved."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field="test",
            message="Test message",
            row_indices=[],
            count=0
        )
        assert issue.count == 0


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result(self):
        """Test ValidationResult with no issues."""
        result = ValidationResult(total_rows=100, valid_rows=100)
        assert result.is_valid
        assert not result.has_warnings
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.info_count == 0

    def test_add_issue(self):
        """Test adding issues to result."""
        result = ValidationResult(total_rows=100, valid_rows=100)
        result.add_issue(
            severity=ValidationSeverity.ERROR,
            field="amount",
            message="Invalid amount",
            row_indices=[1, 2, 3]
        )
        assert result.error_count == 1
        assert not result.is_valid

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        result = ValidationResult(total_rows=100, valid_rows=100)
        result.add_issue(ValidationSeverity.ERROR, "field1", "Error 1")
        result.add_issue(ValidationSeverity.WARNING, "field2", "Warning 1")
        result.add_issue(ValidationSeverity.INFO, "field3", "Info 1")

        errors = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(errors) == 1
        assert errors[0].message == "Error 1"

    def test_get_issues_by_field(self):
        """Test filtering issues by field."""
        result = ValidationResult(total_rows=100, valid_rows=100)
        result.add_issue(ValidationSeverity.ERROR, "amount", "Error 1")
        result.add_issue(ValidationSeverity.WARNING, "amount", "Warning 1")
        result.add_issue(ValidationSeverity.INFO, "date", "Info 1")

        amount_issues = result.get_issues_by_field("amount")
        assert len(amount_issues) == 2

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(total_rows=100, valid_rows=95)
        result.add_issue(
            ValidationSeverity.ERROR,
            "amount",
            "Invalid amount",
            row_indices=[1, 2, 3]
        )

        result_dict = result.to_dict()
        assert result_dict["total_rows"] == 100
        assert result_dict["valid_rows"] == 95
        assert result_dict["summary"]["errors"] == 1
        assert len(result_dict["issues"]) == 1

    def test_generate_report(self):
        """Test generating human-readable report."""
        result = ValidationResult(total_rows=100, valid_rows=95)
        result.add_issue(
            ValidationSeverity.ERROR,
            "amount",
            "Invalid amounts found",
            row_indices=[1, 2, 3]
        )

        report = result.generate_report()
        assert "DATA VALIDATION REPORT" in report
        assert "Total Rows: 100" in report
        assert "Valid Rows: 95" in report
        assert "ERRORS:" in report
        assert "Invalid amounts found" in report


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ValidationConfig()
        assert config.required_fields == ["date", "description", "amount"]
        assert not config.allow_future_dates
        assert not config.allow_zero_amounts
        assert config.check_duplicates
        assert config.duplicate_threshold == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidationConfig(
            required_fields=["date", "amount"],
            allow_future_dates=True,
            min_amount=Decimal("-1000"),
            max_amount=Decimal("1000")
        )
        assert config.required_fields == ["date", "amount"]
        assert config.allow_future_dates
        assert config.min_amount == Decimal("-1000")


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator()
        assert validator.config is not None

        custom_config = ValidationConfig(allow_future_dates=True)
        validator = DataValidator(custom_config)
        assert validator.config.allow_future_dates

    def test_valid_dataframe(self):
        """Test validation of a valid DataFrame."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert result.is_valid
        assert result.error_count == 0
        assert result.total_rows == 2
        assert result.valid_rows == 2

    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        df = pd.DataFrame({
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        assert result.error_count == 1
        issues = result.get_issues_by_field("schema")
        assert len(issues) == 1
        assert "date" in issues[0].message

    def test_invalid_date_type(self):
        """Test detection of invalid date type."""
        df = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],  # String instead of datetime
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        date_issues = result.get_issues_by_field("date")
        assert any("not datetime type" in issue.message for issue in date_issues)

    def test_missing_dates(self):
        """Test detection of missing dates (NaT)."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', None, '2025-01-03']),
            'description': ['Coffee', 'Lunch', 'Dinner'],
            'amount': [-5.50, -12.00, -20.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        date_issues = result.get_issues_by_field("date")
        assert any("NaT" in issue.message for issue in date_issues)
        assert date_issues[0].row_indices == [1]

    def test_future_dates(self):
        """Test detection of future dates."""
        future_date = datetime.now() + timedelta(days=30)
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', future_date]),
            'description': ['Coffee', 'Future purchase'],
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert result.is_valid  # Warnings don't make it invalid
        assert result.has_warnings
        date_issues = result.get_issues_by_field("date")
        assert any("future dates" in issue.message for issue in date_issues)

    def test_allow_future_dates(self):
        """Test allowing future dates with configuration."""
        future_date = datetime.now() + timedelta(days=30)
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', future_date]),
            'description': ['Coffee', 'Future purchase'],
            'amount': [-5.50, -12.00]
        })

        config = ValidationConfig(allow_future_dates=True)
        validator = DataValidator(config)
        result = validator.validate(df)

        # Should not have future date warnings
        date_issues = result.get_issues_by_field("date")
        assert not any("future dates" in issue.message for issue in date_issues)

    def test_unrealistic_years(self):
        """Test detection of unrealistic years."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['1850-01-01', '2200-01-01']),
            'description': ['Old', 'Far future'],
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        date_issues = result.get_issues_by_field("date")
        # Should have at least 2 error-level issues (old year and future year)
        error_issues = [issue for issue in date_issues if issue.severity == ValidationSeverity.ERROR]
        assert len(error_issues) == 2

    def test_nan_amounts(self):
        """Test detection of NaN amounts."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, np.nan]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        amount_issues = result.get_issues_by_field("amount")
        assert any("NaN" in issue.message for issue in amount_issues)

    def test_zero_amounts(self):
        """Test detection of zero amounts."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Free item'],
            'amount': [-5.50, 0.0]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert result.is_valid  # Zero amounts are warnings
        assert result.has_warnings
        amount_issues = result.get_issues_by_field("amount")
        assert any("zero amount" in issue.message.lower() for issue in amount_issues)

    def test_allow_zero_amounts(self):
        """Test allowing zero amounts with configuration."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Free item'],
            'amount': [-5.50, 0.0]
        })

        config = ValidationConfig(allow_zero_amounts=True)
        validator = DataValidator(config)
        result = validator.validate(df)

        amount_issues = result.get_issues_by_field("amount")
        assert not any("zero amount" in issue.message.lower() for issue in amount_issues)

    def test_amount_range_validation(self):
        """Test amount range validation."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Huge expense', 'Normal'],
            'amount': [-1000000000000.00, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        amount_issues = result.get_issues_by_field("amount")
        assert any("below minimum" in issue.message for issue in amount_issues)

    def test_round_number_detection(self):
        """Test detection of suspiciously round numbers."""
        # Create dataset with >50% round hundreds
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01'] * 20),
            'description': ['Transaction'] * 20,
            'amount': [-100.0] * 15 + [-5.50] * 5  # 75% are round hundreds
        })

        validator = DataValidator()
        result = validator.validate(df)

        amount_issues = result.get_issues_by_field("amount")
        assert any("round hundreds" in issue.message for issue in amount_issues)

    def test_empty_descriptions(self):
        """Test detection of empty descriptions."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', '   '],  # Empty after strip
            'amount': [-5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        desc_issues = result.get_issues_by_field("description")
        assert any("empty" in issue.message.lower() for issue in desc_issues)

    def test_exact_duplicates(self):
        """Test detection of exact duplicate transactions."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Coffee', 'Lunch'],
            'amount': [-5.50, -5.50, -12.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        dup_issues = result.get_issues_by_field("duplicates")
        assert len(dup_issues) > 0
        assert any("duplicate" in issue.message.lower() for issue in dup_issues)

    def test_duplicate_threshold(self):
        """Test duplicate detection threshold."""
        # Only 2 duplicates (below threshold of 3)
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'description': ['Coffee', 'Coffee'],
            'amount': [-5.50, -5.50]
        })

        validator = DataValidator()
        result = validator.validate(df)

        dup_issues = result.get_issues_by_field("duplicates")
        # Should be INFO level, not WARNING
        if dup_issues:
            assert all(issue.severity == ValidationSeverity.INFO for issue in dup_issues)

    def test_near_duplicates(self):
        """Test detection of near-duplicate transactions."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Coffee', 'Lunch'],
            'amount': [-5.50, -6.00, -12.00]  # Same date/desc, different amount
        })

        validator = DataValidator()
        result = validator.validate(df)

        dup_issues = result.get_issues_by_field("duplicates")
        assert any("near-duplicate" in issue.message.lower() for issue in dup_issues)

    def test_field_length_validation(self):
        """Test field length validation."""
        long_description = 'X' * 600  # Exceeds MAX_DESCRIPTION_LENGTH (500)
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01']),
            'description': [long_description],
            'amount': [-5.50]
        })

        validator = DataValidator()
        result = validator.validate(df)

        desc_issues = result.get_issues_by_field("description")
        assert any("maximum length" in issue.message for issue in desc_issues)

    def test_invalid_transaction_types(self):
        """Test detection of invalid transaction types."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, -12.00],
            'transaction_type': ['debit', 'invalid_type']
        })

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        type_issues = result.get_issues_by_field("transaction_type")
        assert any("invalid" in issue.message.lower() for issue in type_issues)

    def test_transaction_type_amount_consistency(self):
        """Test consistency between transaction type and amount sign."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Deposit'],
            'amount': [5.50, -100.00],  # Positive debit, negative credit
            'transaction_type': ['debit', 'credit']
        })

        validator = DataValidator()
        result = validator.validate(df)

        type_issues = result.get_issues_by_field("transaction_type")
        assert len(type_issues) == 2  # Both are inconsistent

    def test_custom_validator(self):
        """Test custom validation function."""
        def check_minimum_transaction(df, result):
            """Custom validator to check minimum transaction amount."""
            small_amounts = df[df['amount'].abs() < 1.0]
            if len(small_amounts) > 0:
                result.add_issue(
                    severity=ValidationSeverity.INFO,
                    field="amount",
                    message="Found transactions less than $1",
                    row_indices=small_amounts.index.tolist()
                )

        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Small item', 'Large item'],
            'amount': [-0.50, -50.00]
        })

        config = ValidationConfig(
            custom_validators={'minimum_transaction': check_minimum_transaction}
        )
        validator = DataValidator(config)
        result = validator.validate(df)

        assert result.info_count > 0
        amount_issues = result.get_issues_by_field("amount")
        assert any("less than $1" in issue.message for issue in amount_issues)

    def test_valid_rows_calculation(self):
        """Test that valid_rows is calculated correctly."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', None, '2025-01-03']),
            'description': ['Coffee', 'Lunch', '   '],  # Row 2 has empty desc
            'amount': [-5.50, -12.00, -20.00]
        })

        validator = DataValidator()
        result = validator.validate(df)

        # Rows 1 and 2 have errors
        assert result.total_rows == 3
        assert result.valid_rows == 1  # Only row 0 is valid


class TestValidateDataframeFunction:
    """Tests for the convenience validate_dataframe function."""

    def test_validate_dataframe_basic(self):
        """Test basic usage of validate_dataframe function."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Lunch'],
            'amount': [-5.50, -12.00]
        })

        result = validate_dataframe(df)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_validate_dataframe_with_config(self):
        """Test validate_dataframe with custom configuration."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'description': ['Coffee', 'Free item'],
            'amount': [-5.50, 0.0]
        })

        config = ValidationConfig(allow_zero_amounts=True)
        result = validate_dataframe(df, config)

        # Should not have zero amount warnings
        amount_issues = result.get_issues_by_field("amount")
        assert not any("zero amount" in issue.message.lower() for issue in amount_issues)
