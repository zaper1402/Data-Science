
### ✅ Data Handling & Cleaning (Detailed Guide)

#### 🧹 What is Data Cleaning?
Data cleaning is the process of identifying and correcting errors, inconsistencies, and inaccuracies in datasets to improve data quality for analysis and modeling.

#### 🔍 Types of Data Quality Issues

**1. Missing Data**
- **Completely Missing Values**: `NaN`, `None`, empty strings
- **Implicit Missing**: Zeros that should be NaN, placeholder values (-999, 'Unknown')
- **Partial Missing**: Incomplete records with some fields missing

**2. Duplicate Data**
- **Exact Duplicates**: Identical rows across all columns
- **Near Duplicates**: Similar records with minor differences (typos, formatting)
- **Logical Duplicates**: Same entity with different representations

**3. Inconsistent Data**
- **Format Inconsistencies**: Date formats (DD/MM/YYYY vs MM-DD-YYYY)
- **Case Inconsistencies**: 'New York' vs 'new york' vs 'NEW YORK'
- **Unit Inconsistencies**: Mixing currencies, measurement units

**4. Invalid/Incorrect Data**
- **Out of Range Values**: Ages > 150, negative prices
- **Invalid Formats**: Malformed emails, phone numbers
- **Logical Inconsistencies**: End date before start date

**5. Outliers**
- **Statistical Outliers**: Values beyond 3 standard deviations
- **Domain Outliers**: Values that don't make business sense
- **Data Entry Errors**: Typos causing extreme values

#### 🛠️ Data Cleaning Techniques

**1. Handling Missing Data**

```python
import pandas as pd
import numpy as np

# Identify missing data
df.isnull().sum()
df.info()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Strategies:
# a) Remove missing data
df.dropna()  # Remove rows with any missing values
df.dropna(axis=1)  # Remove columns with any missing values
df.dropna(thresh=5)  # Keep rows with at least 5 non-null values

# b) Fill missing data
df.fillna(0)  # Fill with constant
df.fillna(df.mean())  # Fill with mean (numerical)
df.fillna(df.mode().iloc[0])  # Fill with mode (categorical)
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill

# c) Advanced imputation
from sklearn.impute import SimpleImputer, KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

**2. Handling Duplicates**

```python
# Identify duplicates
df.duplicated().sum()
df[df.duplicated()]

# Remove duplicates
df.drop_duplicates()  # Remove exact duplicates
df.drop_duplicates(subset=['column1', 'column2'])  # Based on specific columns
df.drop_duplicates(keep='first')  # Keep first occurrence

# Handle near duplicates
from fuzzywuzzy import fuzz
# Custom function to identify similar strings
```

**3. Data Type Conversion**

```python
# Convert data types
df['column'] = df['column'].astype('int64')
df['date_column'] = pd.to_datetime(df['date_column'])
df['category'] = df['category'].astype('category')

# Handle mixed types
df['mixed_column'] = pd.to_numeric(df['mixed_column'], errors='coerce')
```

**4. Standardizing Text Data**

```python
# String cleaning
df['text_column'] = df['text_column'].str.lower()  # Lowercase
df['text_column'] = df['text_column'].str.strip()  # Remove whitespace
df['text_column'] = df['text_column'].str.replace('[^a-zA-Z0-9]', '', regex=True)  # Remove special chars

# Standardize categories
df['category'] = df['category'].replace({
    'NY': 'New York',
    'NYC': 'New York',
    'California': 'CA'
})
```

**5. Outlier Detection & Treatment**

```python
# Statistical methods
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Cap outliers
df['column'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Z-score method
from scipy.stats import zscore
df['z_score'] = zscore(df['column'])
df_clean = df[abs(df['z_score']) < 3]
```

#### 🚨 Common Data Cleaning Pitfalls

**What to Avoid:**
- Removing too much data without understanding impact
- Blindly filling missing values with mean/median
- Ignoring the business context of data
- Not documenting cleaning decisions
- Cleaning data before understanding it
- Assuming correlation implies data quality issues

**Best Practices:**
- Always backup original data
- Document all cleaning steps
- Validate cleaning results
- Consider domain expertise
- Test impact on downstream analysis
- Use version control for cleaning scripts

#### 🔧 Advanced Cleaning Techniques

**1. Data Profiling**
```python
# Using pandas-profiling for comprehensive data assessment
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Data Quality Report")
profile.to_file("data_quality_report.html")
```

**2. Automated Data Quality Checks**
```python
# Custom data quality functions
def check_data_quality(df):
    report = {
        'missing_values': df.isnull().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes,
        'unique_values': df.nunique(),
        'memory_usage': df.memory_usage(deep=True)
    }
    return report
```

**3. Pipeline Approach**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create cleaning pipeline
cleaning_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
```