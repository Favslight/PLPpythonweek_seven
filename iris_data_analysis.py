
# Iris Data Analysis Script
# Tasks: Load, Explore, Analyze, Visualize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style
sns.set(style='whitegrid')

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Iris dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check structure and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (no missing values in Iris, but shown as example)
df_cleaned = df.dropna()

# Task 2: Basic Data Analysis
print("\nBasic Statistical Summary:")
print(df_cleaned.describe())

grouped_means = df_cleaned.groupby('species')['petal length (cm)'].mean()
print("\nAverage Petal Length by Species:")
print(grouped_means)

median_petal = df_cleaned['petal length (cm)'].median()
std_petal = df_cleaned['petal length (cm)'].std()
print(f"\nMedian Petal Length: {median_petal}")
print(f"Standard Deviation of Petal Length: {std_petal}")

print("""
Observations:
- Petal length varies significantly across species.
- Virginica has the highest average petal length.
- Low standard deviation suggests consistent measurements within species.
""")

# Task 3: Visualizations

# Line Chart - Trend over sorted petal length
df_sorted = df.sort_values(by='petal length (cm)').reset_index()
plt.figure(figsize=(10, 5))
plt.plot(df_sorted.index, df_sorted['petal length (cm)'], color='green', marker='o', linestyle='-')
plt.title('Line Chart: Petal Length Trend Across Samples')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart - Average Petal Length by Species
plt.figure(figsize=(8, 5))
grouped_means.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Histogram - Sepal Width Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='mediumpurple')
plt.title('Histogram: Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set1')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

print("""
Final Observations:
- Virginica species stands out with highest petal and sepal lengths.
- Species are well-separated in the scatter plot, suggesting strong feature-based classification potential.
""")
