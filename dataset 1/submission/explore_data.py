import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create folder if it doesn't exist
os.makedirs("datasetPictures", exist_ok=True)

# Load training data
df = pd.read_csv("dataset/train.csv")

# Basic info
print("🔍 Dataset Overview:")
print(df.info())
print("\n📊 Descriptive Statistics:")
print(df.describe())

# Missing values
print("\n❗ Missing Values:")
print(df.isnull().sum())

# Target variable distribution
print("\n🎯 Target Variable Distribution (smoking):")
print(df['smoking'].value_counts(normalize=True))

# Correlation matrix (only numeric)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("🔗 Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("datasetPictures/correlation_matrix.png")
plt.close()

# Boxplots for outlier detection
numerical_cols = df.select_dtypes(include='number').columns.drop(['id', 'smoking'])
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='smoking', y=col)
    plt.title(f'Boxplot of {col} by Smoking Status')
    plt.tight_layout()
    plt.savefig(f"datasetPictures/boxplot_{col}.png")
    plt.close()

# Distribution of key features
important_cols = ['age', 'waist(cm)', 'HDL', 'LDL', 'Gtp']
for col in important_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True, hue='smoking', multiple='stack')
    plt.title(f'Distribution of {col} by Smoking Status')
    plt.tight_layout()
    plt.savefig(f"datasetPictures/dist_{col}.png")
    plt.close()

print("\n✅ Visuals saved: correlation matrix, boxplots, distributions.")

# Optional: save class balance to file
df['smoking'].value_counts(normalize=True).to_csv("datasetPictures/class_balance.csv")

print("\n📁 Check output files in 'datasetPictures' folder.")
