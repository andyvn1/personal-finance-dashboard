# Customization Guide

Learn how to customize the Personal Finance Dashboard to match your specific needs.

## Custom Categories

### Default Categories

The system includes standard categories:
- Income (Salary, Investment Income)
- Food & Dining (Groceries, Restaurants)
- Transportation (Gas, Public Transit)
- Shopping (Clothing, Electronics)
- Utilities (Electricity, Water, Internet)
- Healthcare
- Entertainment
- And more...

### Adding Custom Categories

Create your own categories to match your spending habits:

```python
from src.categorizer import add_category

# Add a custom category
add_category("Pet Expenses", parent="Personal Care")
```

### Category Rules

Define rules to automatically categorize transactions:

```python
from src.categorizer import add_category_rule

# Categorize all transactions containing "Petco" as "Pet Expenses"
add_category_rule(pattern="Petco", category="Pet Expenses")

# Use regex for advanced matching
add_category_rule(pattern=r"Starbucks|Coffee Bean", category="Coffee", use_regex=True)
```

## Custom Visualizations

### Chart Customization

Modify chart appearance, colors, and styles to match your preferences.

```python
from src.visualizer import create_custom_chart

# Create a custom spending chart
create_custom_chart(
    data=spending_summary,
    chart_type="pie",
    colors=["#FF6384", "#36A2EB", "#FFCE56"],
    title="My Spending Breakdown"
)
```

## Budget Settings

Set monthly budgets for each category and track your progress.

```python
from src.budget import set_budget

set_budget("Food & Dining", 500.00)
set_budget("Transportation", 200.00)
```

## Report Templates

Customize report layouts and content to focus on metrics that matter to you.

(Detailed customization options to be added)
