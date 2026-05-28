# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("train.csv")

# -----------------------------
# DISPLAY BASIC INFORMATION
# -----------------------------

print("First 5 Rows")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Fill missing Age values with mean
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill missing Embarked values with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because many values are missing
data.drop(columns='Cabin', inplace=True)

print("\nMissing Values After Cleaning")
print(data.isnull().sum())

# -----------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------

# Survival Count
plt.figure(figsize=(6,5))
sns.countplot(x='Survived', data=data)

plt.title("Survival Count")
plt.show()

# Gender Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='Sex', data=data)

plt.title("Gender Distribution")
plt.show()

# Passenger Class Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='Pclass', data=data)

plt.title("Passenger Class Distribution")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(data['Age'], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# Survival based on Gender
plt.figure(figsize=(6,5))
sns.countplot(x='Sex', hue='Survived', data=data)

plt.title("Survival Based on Gender")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(data.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()