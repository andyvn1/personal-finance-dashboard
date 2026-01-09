"""
Tests for the sample data generator module.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

from src.sample_data_generator import (
    SampleDataGenerator,
    GeneratorConfig,
    generate_sample_files,
    TRANSACTION_CATEGORIES,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.num_transactions == 100
        assert config.include_edge_cases is True
        assert config.edge_case_probability == 0.05
        assert config.duplicate_probability == 0.02
        assert config.missing_field_probability == 0.01

    def test_custom_config(self):
        """Test custom configuration values."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = GeneratorConfig(
            num_transactions=500,
            start_date=start,
            end_date=end,
            include_edge_cases=False
        )
        assert config.num_transactions == 500
        assert config.start_date == start
        assert config.end_date == end
        assert config.include_edge_cases is False


class TestSampleDataGenerator:
    """Tests for SampleDataGenerator class."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SampleDataGenerator()
        assert generator.config is not None
        assert generator.config.num_transactions == 100

    def test_generator_with_custom_config(self):
        """Test generator with custom configuration."""
        config = GeneratorConfig(num_transactions=50)
        generator = SampleDataGenerator(config)
        assert generator.config.num_transactions == 50

    def test_generate_transactions_count(self):
        """Test that correct number of transactions is generated."""
        config = GeneratorConfig(num_transactions=50, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()
        assert len(df) == 50

    def test_generate_transactions_columns(self):
        """Test that generated DataFrame has required columns."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        required_columns = ['date', 'description', 'amount', 'category', 'transaction_type']
        for col in required_columns:
            assert col in df.columns

    def test_generate_transactions_date_range(self):
        """Test that transactions are within specified date range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)
        config = GeneratorConfig(
            num_transactions=100,
            start_date=start,
            end_date=end,
            include_edge_cases=False  # No future dates
        )
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        assert df['date'].min() >= pd.Timestamp(start)
        assert df['date'].max() <= pd.Timestamp(end)

    def test_generate_transactions_sorted_by_date(self):
        """Test that transactions are sorted by date."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        # Check if sorted
        assert df['date'].is_monotonic_increasing

    def test_transaction_types_present(self):
        """Test that both credit and debit transactions are generated."""
        config = GeneratorConfig(num_transactions=100, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        transaction_types = df['transaction_type'].unique()
        assert 'debit' in transaction_types
        assert 'credit' in transaction_types

    def test_income_transactions_positive(self):
        """Test that income transactions have positive amounts."""
        config = GeneratorConfig(num_transactions=200, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        income_df = df[df['transaction_type'] == 'credit']
        if len(income_df) > 0:
            assert (income_df['amount'] > 0).all()

    def test_expense_transactions_negative(self):
        """Test that expense transactions have negative amounts."""
        config = GeneratorConfig(num_transactions=200, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        expense_df = df[df['transaction_type'] == 'debit']
        if len(expense_df) > 0:
            # Allow for zero amounts in edge cases
            assert (expense_df['amount'] <= 0).all()

    def test_categories_from_defined_list(self):
        """Test that all categories are from the defined list."""
        config = GeneratorConfig(num_transactions=100, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        valid_categories = set(TRANSACTION_CATEGORIES.keys())
        generated_categories = set(df['category'].dropna().unique())

        assert generated_categories.issubset(valid_categories)

    def test_merchants_realistic(self):
        """Test that merchants are from defined lists."""
        config = GeneratorConfig(num_transactions=100, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        # Collect all valid merchants
        valid_merchants = set()
        for cat_info in TRANSACTION_CATEGORIES.values():
            valid_merchants.update(cat_info["merchants"])

        generated_merchants = set(df['merchant'].dropna().unique())
        assert generated_merchants.issubset(valid_merchants)

    def test_amount_ranges_realistic(self):
        """Test that amounts are within realistic ranges."""
        config = GeneratorConfig(num_transactions=200, include_edge_cases=False)
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        # Check that amounts are reasonable (not including edge cases)
        abs_amounts = df['amount'].abs()
        assert abs_amounts.min() >= 0
        assert abs_amounts.max() <= 10000  # Reasonable max

    def test_edge_cases_included(self):
        """Test that edge cases are generated when enabled."""
        config = GeneratorConfig(
            num_transactions=500,
            include_edge_cases=True,
            edge_case_probability=0.2  # Higher probability for testing
        )
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        # Should have some edge cases
        # Check for round amounts (multiples of 100)
        round_amounts = df[df['amount'].abs() % 100 == 0]
        assert len(round_amounts) > 0

        # Check for missing values
        assert df['category'].isna().any() or df['merchant'].isna().any()

    def test_duplicates_generated(self):
        """Test that duplicates can be generated."""
        config = GeneratorConfig(
            num_transactions=200,
            include_edge_cases=True,
            duplicate_probability=0.1  # Higher probability
        )
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        # Check for duplicate descriptions on same date
        duplicates = df.duplicated(subset=['date', 'description', 'amount'], keep=False)
        # Should have at least some duplicates
        assert duplicates.sum() > 0

    def test_no_edge_cases_when_disabled(self):
        """Test that edge cases are not generated when disabled."""
        config = GeneratorConfig(
            num_transactions=100,
            include_edge_cases=False
        )
        generator = SampleDataGenerator(config)
        df = generator.generate_transactions()

        # Should not have missing categories or merchants
        assert not df['category'].isna().any()
        assert not df['merchant'].isna().any()

    def test_export_generic_format(self, tmp_path):
        """Test export to generic CSV format."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        output_path = tmp_path / "test_generic.csv"
        generator.export_to_csv(df, output_path, format_type='generic')

        assert output_path.exists()

        # Read and verify
        exported_df = pd.read_csv(output_path)
        assert 'Date' in exported_df.columns
        assert 'Description' in exported_df.columns
        assert 'Amount' in exported_df.columns
        assert 'Category' in exported_df.columns
        assert len(exported_df) == len(df)

    def test_export_chase_format(self, tmp_path):
        """Test export to Chase CSV format."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        output_path = tmp_path / "test_chase.csv"
        generator.export_to_csv(df, output_path, format_type='chase')

        assert output_path.exists()

        # Read and verify Chase-specific columns
        exported_df = pd.read_csv(output_path)
        assert 'Transaction Date' in exported_df.columns
        assert 'Post Date' in exported_df.columns
        assert 'Description' in exported_df.columns
        assert 'Amount' in exported_df.columns

    def test_export_bank_of_america_format(self, tmp_path):
        """Test export to Bank of America CSV format."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        output_path = tmp_path / "test_boa.csv"
        generator.export_to_csv(df, output_path, format_type='bank_of_america')

        assert output_path.exists()

        # Read and verify BoA-specific columns
        exported_df = pd.read_csv(output_path)
        assert 'Date' in exported_df.columns
        assert 'Description' in exported_df.columns
        assert 'Amount' in exported_df.columns
        assert 'Running Bal.' in exported_df.columns

    def test_export_wells_fargo_format(self, tmp_path):
        """Test export to Wells Fargo CSV format."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        output_path = tmp_path / "test_wf.csv"
        generator.export_to_csv(df, output_path, format_type='wells_fargo')

        assert output_path.exists()

        # Read and verify Wells Fargo-specific columns
        exported_df = pd.read_csv(output_path)
        assert 'Date' in exported_df.columns
        assert 'Amount' in exported_df.columns
        assert 'Description' in exported_df.columns
        assert 'Balance' in exported_df.columns

    def test_export_invalid_format(self, tmp_path):
        """Test that invalid format raises error."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        output_path = tmp_path / "test_invalid.csv"

        with pytest.raises(ValueError, match="Unknown format type"):
            generator.export_to_csv(df, output_path, format_type='invalid_format')

    def test_export_creates_directory(self, tmp_path):
        """Test that export creates output directory if needed."""
        generator = SampleDataGenerator()
        df = generator.generate_transactions()

        nested_path = tmp_path / "nested" / "dir" / "test.csv"
        generator.export_to_csv(df, nested_path, format_type='generic')

        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestGenerateSampleFiles:
    """Tests for the generate_sample_files convenience function."""

    def test_generate_sample_files_count(self, tmp_path):
        """Test that 4 files are generated (one per format)."""
        files = generate_sample_files(tmp_path, num_transactions=50)

        assert len(files) == 4
        for f in files:
            assert f.exists()

    def test_generate_sample_files_formats(self, tmp_path):
        """Test that all expected formats are generated."""
        files = generate_sample_files(tmp_path, num_transactions=50)

        filenames = [f.name for f in files]
        assert any('generic' in name for name in filenames)
        assert any('chase' in name for name in filenames)
        assert any('bank_of_america' in name for name in filenames)
        assert any('wells_fargo' in name for name in filenames)

    def test_generate_sample_files_content(self, tmp_path):
        """Test that generated files have content."""
        files = generate_sample_files(tmp_path, num_transactions=50)

        for f in files:
            df = pd.read_csv(f)
            assert len(df) == 50  # Should have requested number of transactions

    def test_generate_sample_files_creates_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        nested_dir = tmp_path / "nested" / "sample"
        files = generate_sample_files(nested_dir, num_transactions=10)

        assert nested_dir.exists()
        assert len(files) == 4


class TestTransactionCategories:
    """Tests for TRANSACTION_CATEGORIES configuration."""

    def test_all_categories_have_merchants(self):
        """Test that all categories have merchant lists."""
        for category, info in TRANSACTION_CATEGORIES.items():
            assert 'merchants' in info
            assert len(info['merchants']) > 0

    def test_all_categories_have_amount_range(self):
        """Test that all categories have amount ranges."""
        for category, info in TRANSACTION_CATEGORIES.items():
            assert 'amount_range' in info
            min_amt, max_amt = info['amount_range']
            assert min_amt > 0
            assert max_amt > min_amt

    def test_all_categories_have_frequency(self):
        """Test that all categories have frequency weights."""
        for category, info in TRANSACTION_CATEGORIES.items():
            assert 'frequency' in info
            assert 0 < info['frequency'] <= 1

    def test_frequencies_sum_reasonable(self):
        """Test that frequencies sum to approximately 1."""
        total_freq = sum(info['frequency'] for info in TRANSACTION_CATEGORIES.values())
        assert 0.95 <= total_freq <= 1.05  # Allow small variance

    def test_income_category_marked(self):
        """Test that income category is properly marked."""
        income_info = TRANSACTION_CATEGORIES.get('Income')
        assert income_info is not None
        assert income_info.get('is_income') is True


class TestReproducibility:
    """Tests for reproducible data generation."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        config1 = GeneratorConfig(num_transactions=50, include_edge_cases=False)
        generator1 = SampleDataGenerator(config1)
        df1 = generator1.generate_transactions()

        config2 = GeneratorConfig(num_transactions=50, include_edge_cases=False)
        generator2 = SampleDataGenerator(config2)
        df2 = generator2.generate_transactions()

        # Should generate identical data (due to fixed seed)
        pd.testing.assert_frame_equal(df1, df2)
