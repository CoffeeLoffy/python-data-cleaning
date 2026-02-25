import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')

# Initial exploration
print(df.shape)
print(df.info())
print(df.head())

# Check missing values
print("\n=== Missing Values ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Check duplicates
print(f"\nDuplicates: {df.duplicated().sum()}")

# Check data types
print("\n=== Data Types ===")
print(df.dtypes)

# Basic statistics
print("\n=== Cost Columns Stats ===")
cost_cols = ['Order Item Product Price', 'Order Item Total', 'Sales']
print(df[cost_cols].describe())

# 1. Handle Missing Values
print("\n=== STEP 1: Handle Missing Values ===")

df_clean = df.copy()

# Check which columns have nulls
null_cols = df_clean.columns[df_clean.isnull().any()].tolist()
print(f"Columns with nulls: {null_cols}")

# Drop if critical cost columns are null
critical_cols = ['Order Item Product Price', 'Order Item Total', 'Sales']
df_clean = df_clean.dropna(subset=critical_cols)

# Fill missing customer/product info with 'Unknown'
text_cols = ['Customer City', 'Customer State', 'Product Description']
for col in text_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('Unknown')

print(f"Rows before: {len(df)}, after: {len(df_clean)}")


# 2. Remove Duplicates
print("\n=== STEP 2: Remove Duplicates ===")
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
print(f"Removed {before - after} duplicate rows")


# 3. Clean Cost Columns
print("\n=== STEP 3: Clean Cost Data ===")

# Remove negative costs (data errors)
df_clean = df_clean[df_clean['Order Item Product Price'] > 0]
df_clean = df_clean[df_clean['Order Item Total'] > 0]
df_clean = df_clean[df_clean['Sales'] > 0]

# Check for outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

for col in cost_cols:
    outliers, lower, upper = detect_outliers(df_clean, col)
    print(f"{col}: {len(outliers)} outliers (range: {lower:.2f} - {upper:.2f})")


# 4. Standardize Text Fields
print("\n=== STEP 4: Standardize Categories ===")

# Clean shipping mode names
df_clean['Shipping Mode'] = df_clean['Shipping Mode'].str.strip().str.title()

# Clean market/region
df_clean['Market'] = df_clean['Market'].str.strip().str.upper()
df_clean['Order Region'] = df_clean['Order Region'].str.strip().str.title()

# Clean category names
df_clean['Category Name'] = df_clean['Category Name'].str.strip()

# Clean department names
df_clean['Department Name'] = df_clean['Department Name'].str.strip()

print("Shipping Modes:", df_clean['Shipping Mode'].value_counts())
print("\nMarkets:", df_clean['Market'].value_counts())


# 5. Convert Dates
print("\n=== STEP 5: Convert Dates ===")

date_cols = ['order date (DateOrders)', 'shipping date (DateOrders)']
for col in date_cols:
    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

# Create time-based features
df_clean['Order Year'] = df_clean['order date (DateOrders)'].dt.year
df_clean['Order Month'] = df_clean['order date (DateOrders)'].dt.month
df_clean['Order Quarter'] = df_clean['order date (DateOrders)'].dt.quarter
df_clean['Order Day of Week'] = df_clean['order date (DateOrders)'].dt.day_name()

# Calculate delivery time
df_clean['Actual Delivery Days'] = df_clean['Days for shipping (real)']
df_clean['Scheduled Delivery Days'] = df_clean['Days for shipment (scheduled)']
df_clean['Delivery Delay'] = df_clean['Actual Delivery Days'] - df_clean['Scheduled Delivery Days']

print(df_clean[['Order Year', 'Order Month', 'Actual Delivery Days', 'Delivery Delay']].head())

print("\n=== STEP 6: Create Cost Metrics ===")

# 1. Cost per Unit
df_clean['Cost Per Unit'] = (
    df_clean['Order Item Total'] / df_clean['Order Item Quantity']
)

# 2. Profit (already exists as 'Benefit per order' and 'Order Profit Per Order')
# Let's use the existing profit column
df_clean['Profit'] = df_clean['Order Profit Per Order']

# 3. Profit Margin %
df_clean['Profit Margin %'] = (
    df_clean['Profit'] / df_clean['Sales'] * 100
)

# 4. Revenue per Customer
df_clean['Revenue Per Customer'] = df_clean['Sales per customer']

# 5. Discount Impact
df_clean['Discount Amount'] = (
    df_clean['Order Item Product Price'] * 
    df_clean['Order Item Quantity'] * 
    df_clean['Order Item Discount Rate']
)

# 6. Cost Category (Low/Medium/High)
df_clean['Cost Category'] = pd.cut(
    df_clean['Order Item Total'],
    bins=[0, 100, 500, np.inf],
    labels=['Low', 'Medium', 'High']
)

# 7. Late Delivery Flag (already exists as 'Late_delivery_risk')
df_clean['Is Late'] = df_clean['Late_delivery_risk']

# 8. Order Size Category
df_clean['Order Size'] = pd.cut(
    df_clean['Order Item Quantity'],
    bins=[0, 1, 3, 10, np.inf],
    labels=['Single', 'Small', 'Medium', 'Bulk']
)

print(df_clean[['Cost Per Unit', 'Profit Margin %', 'Discount Amount', 
                'Cost Category', 'Order Size']].describe())

print("\n=== STEP 7: Handle Outliers ===")

# Option 1: Cap outliers at 99th percentile
def cap_outliers(df, column, percentile=0.99):
    cap = df[column].quantile(percentile)
    df[column] = df[column].clip(upper=cap)
    return df

# Cap extreme values
df_clean = cap_outliers(df_clean, 'Order Item Total', 0.99)
df_clean = cap_outliers(df_clean, 'Sales', 0.99)

# Option 2: Flag outliers for investigation
df_clean['High Cost Flag'] = np.where(
    df_clean['Order Item Total'] > df_clean['Order Item Total'].quantile(0.95),
    'High',
    'Normal'
)

df_clean['High Discount Flag'] = np.where(
    df_clean['Order Item Discount Rate'] > 0.2,
    'High Discount',
    'Normal'
)

print("High Cost Orders:", df_clean['High Cost Flag'].value_counts())
print("\nHigh Discount Orders:", df_clean['High Discount Flag'].value_counts())

print("\n=== STEP 8: Final Validation ===")

# Check no nulls in key columns
assert df_clean[cost_cols].isnull().sum().sum() == 0, "Nulls found in cost columns!"

# Check no negative costs
assert (df_clean[cost_cols] >= 0).all().all(), "Negative costs found!"

# Check data integrity
assert (df_clean['Sales'] >= df_clean['Order Item Total']).all(), \
    "Sales less than cost - data error!"

assert (df_clean['Order Item Total'] >= 0).all(), "Negative order totals!"

print("✓ All validations passed!")
print(f"\nFinal dataset shape: {df_clean.shape}")
print(f"Date range: {df_clean['Order Year'].min()} - {df_clean['Order Year'].max()}")
print(f"Total orders: {len(df_clean):,}")
print(f"Total revenue: ${df_clean['Sales'].sum():,.2f}")
print(f"Total profit: ${df_clean['Profit'].sum():,.2f}")

# Save cleaned data
df_clean.to_csv('supply_chain_cleaned.csv', index=False)

# Create summary report
summary = pd.DataFrame({
    'Metric': [
        'Total Orders',
        'Total Revenue',
        'Total Cost',
        'Total Profit',
        'Avg Order Value',
        'Avg Cost per Order',
        'Avg Profit Margin %',
        'Late Delivery Rate %'
    ],
    'Value': [
        f"{len(df_clean):,}",
        f"${df_clean['Sales'].sum():,.2f}",
        f"${df_clean['Order Item Total'].sum():,.2f}",
        f"${df_clean['Profit'].sum():,.2f}",
        f"${df_clean['Sales'].mean():,.2f}",
        f"${df_clean['Order Item Total'].mean():,.2f}",
        f"{df_clean['Profit Margin %'].mean():.2f}%",
        f"{(df_clean['Late_delivery_risk'].sum() / len(df_clean) * 100):.2f}%"
    ]
})

print("\n=== Summary Report ===")
print(summary.to_string(index=False))

summary.to_csv('cleaning_summary.csv', index=False)

print("\n✓ Files exported: supply_chain_cleaned.csv, cleaning_summary.csv")

# Set style
sns.set_style("whitegrid")

# 1. Cost Distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df_clean['Order Item Total'].hist(bins=50, edgecolor='black', color='steelblue')
plt.title('Order Cost Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Cost ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
df_clean['Sales'].hist(bins=50, edgecolor='black', color='orange')
plt.title('Sales Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
df_clean['Profit Margin %'].hist(bins=50, edgecolor='black', color='green')
plt.title('Profit Margin Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Margin (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('cost_distributions.png', dpi=300, bbox_inches='tight')
plt.show()


# 2. Cost by Category & Department
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
category_cost = df_clean.groupby('Category Name')['Order Item Total'].sum().sort_values(ascending=False)
category_cost.plot(kind='barh', color='steelblue')
plt.title('Total Cost by Category', fontsize=12, fontweight='bold')
plt.xlabel('Total Cost ($)')
plt.ylabel('')

plt.subplot(1, 2, 2)
dept_cost = df_clean.groupby('Department Name')['Order Item Total'].sum().sort_values(ascending=False)
dept_cost.plot(kind='barh', color='coral')
plt.title('Total Cost by Department', fontsize=12, fontweight='bold')
plt.xlabel('Total Cost ($)')
plt.ylabel('')

plt.tight_layout()
plt.savefig('cost_by_category.png', dpi=300, bbox_inches='tight')
plt.show()


# 3. Delivery Performance
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
shipping_cost = df_clean.groupby('Shipping Mode').agg({
    'Order Item Total': 'mean',
    'Actual Delivery Days': 'mean'
}).sort_values('Order Item Total')

x = range(len(shipping_cost))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

bars1 = ax1.bar([i - width/2 for i in x], shipping_cost['Order Item Total'], 
                width, label='Avg Cost', color='steelblue')
bars2 = ax2.bar([i + width/2 for i in x], shipping_cost['Actual Delivery Days'], 
                width, label='Avg Days', color='orange')

ax1.set_xlabel('Shipping Mode')
ax1.set_ylabel('Average Cost ($)', color='steelblue')
ax2.set_ylabel('Average Delivery Days', color='orange')
ax1.set_xticks(x)
ax1.set_xticklabels(shipping_cost.index, rotation=45, ha='right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Cost vs Delivery Time by Shipping Mode', fontweight='bold')
plt.tight_layout()
plt.savefig('shipping_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# 4. Monthly Trend
monthly_cost = df_clean.groupby(['Order Year', 'Order Month']).agg({
    'Order Item Total': 'sum',
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

monthly_cost['YearMonth'] = (
    monthly_cost['Order Year'].astype(str) + '-' + 
    monthly_cost['Order Month'].astype(str).str.zfill(2)
)

plt.figure(figsize=(14, 6))
plt.plot(range(len(monthly_cost)), monthly_cost['Order Item Total'], 
         marker='o', label='Total Cost', linewidth=2, color='steelblue')
plt.plot(range(len(monthly_cost)), monthly_cost['Sales'], 
         marker='s', label='Sales', linewidth=2, color='orange')
plt.plot(range(len(monthly_cost)), monthly_cost['Profit'], 
         marker='^', label='Profit', linewidth=2, color='green')
plt.xticks(range(len(monthly_cost))[::3], monthly_cost['YearMonth'][::3], rotation=45)
plt.title('Monthly Cost, Sales & Profit Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_trend.png', dpi=300, bbox_inches='tight')
plt.show()


# 5. Market Performance
market_perf = df_clean.groupby('Market').agg({
    'Sales': 'sum',
    'Order Item Total': 'sum',
    'Profit': 'sum',
    'Late_delivery_risk': 'mean'
}).round(2)

print("\n=== Market Performance ===")
print(market_perf)

market_perf[['Sales', 'Order Item Total', 'Profit']].plot(kind='bar', figsize=(10, 6))
plt.title('Performance by Market', fontsize=14, fontweight='bold')
plt.xlabel('Market')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.legend(['Sales', 'Total Cost', 'Profit'])
plt.tight_layout()
plt.savefig('market_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# 1. Cost Summary by Category
category_summary = df_clean.groupby('Category Name').agg({
    'Order Item Total': ['sum', 'mean', 'count'],
    'Sales': ['sum', 'mean'],
    'Profit': 'sum',
    'Profit Margin %': 'mean',
    'Late_delivery_risk': 'mean'
}).round(2)
category_summary.columns = ['_'.join(col) if col[1] else col[0] for col in category_summary.columns]
category_summary.to_csv('category_cost_summary.csv')


# 2. Cost by Region & Market
region_summary = df_clean.groupby(['Market', 'Order Region']).agg({
    'Order Item Total': 'sum',
    'Sales': 'sum',
    'Profit': 'sum',
    'Order Item Quantity': 'sum',
    'Late_delivery_risk': 'mean'
}).reset_index()
region_summary['Profit Margin %'] = (region_summary['Profit'] / region_summary['Sales'] * 100).round(2)
region_summary.to_csv('region_cost_summary.csv', index=False)


# 3. Shipping Mode Analysis
shipping_summary = df_clean.groupby('Shipping Mode').agg({
    'Order Item Total': ['count', 'sum', 'mean'],
    'Actual Delivery Days': 'mean',
    'Delivery Delay': 'mean',
    'Late_delivery_risk': 'mean',
    'Profit Margin %': 'mean'
}).round(2)
shipping_summary.to_csv('shipping_analysis.csv')


# 4. Monthly KPIs
monthly_kpi = df_clean.groupby(['Order Year', 'Order Month']).agg({
    'Order Item Total': 'sum',
    'Sales': 'sum',
    'Profit': 'sum',
    'Order Item Quantity': 'sum',
    'Late_delivery_risk': 'mean'
}).reset_index()
monthly_kpi['Profit Margin %'] = (monthly_kpi['Profit'] / monthly_kpi['Sales'] * 100).round(2)
monthly_kpi.to_csv('monthly_kpi.csv', index=False)


# 5. Department Performance
dept_summary = df_clean.groupby('Department Name').agg({
    'Order Item Total': 'sum',
    'Sales': 'sum',
    'Profit': 'sum',
    'Order Item Quantity': 'sum'
}).reset_index()
dept_summary['Profit Margin %'] = (dept_summary['Profit'] / dept_summary['Sales'] * 100).round(2)
dept_summary.to_csv('department_summary.csv', index=False)


print("\n✓ All dashboard files exported!")
print("Files created:")
print("- category_cost_summary.csv")
print("- region_cost_summary.csv")
print("- shipping_analysis.csv")
print("- monthly_kpi.csv")
print("- department_summary.csv")