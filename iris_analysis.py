# 📊 Analyzing the Iris Dataset with Pandas and Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.datasets import load_iris

    # Load the iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].apply(lambda x: iris.target_names[x])

    # Display first few rows
    print("First few rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean missing values (if any)
    df = df.dropna()

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Group by species and compute mean of features
    group_means = df.groupby('species').mean()
    print("\nMean of each feature by species:")
    print(group_means)

    # Set seaborn style
    sns.set(style="whitegrid")

    # Line Chart - Petal Length over index
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
    plt.title('Petal Length Trend')
    plt.xlabel('Index')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Bar Chart - Average petal length by species
    plt.figure(figsize=(8, 6))
    sns.barplot(x=group_means.index, y=group_means['petal length (cm)'], palette='viridis')
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.tight_layout()
    plt.show()

    # Histogram - Sepal Width
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Scatter Plot - Sepal Length vs Petal Length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.tight_layout()
    plt.show()

    # Observations:
    print("\nObservations and Insights:")
    print("- Setosa species shows distinctly smaller petal lengths and widths.")
    print("- Versicolor and Virginica are more similar but still distinguishable via petal dimensions.")
    print("- Scatter plot shows a clear relationship between sepal and petal length.")
    print("- Histogram of sepal width is roughly normal, slightly right-skewed.")
    print("- Bar chart shows increasing average petal length from Setosa to Virginica.")

except ImportError as e:
    print("Import failed:", e)

except Exception as e:
    print("An error occurred during data processing or plotting:", e)
