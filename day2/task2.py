# -------------------------------------------------------------
# TITANIC DATASET - EXPLORATORY DATA ANALYSIS (EDA)
# Author: [Your Name]
# Experience: 15+ Years in Data Analytics & ML
# -------------------------------------------------------------

# Importing core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Setting global plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# -------------------------------------------------------------
# 1. LOAD THE DATA
# -------------------------------------------------------------
df = pd.read_csv(r"C:\Users\hp\Desktop\dataset1\Titanic-Dataset2.csv")

print("🔍 Dataset Preview:")
print(df.head())
print("\nShape of dataset:", df.shape)

# -------------------------------------------------------------
# 2. BASIC INFORMATION & DATA TYPES
# -------------------------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# -------------------------------------------------------------
print("\nStatistical Summary (Numeric Features):")
print(df.describe().T)

print("\nStatistical Summary (Categorical Features):")
print(df.describe(include="object").T)


missing_percent = df.isnull().mean() * 100
print("\nMissing Value Percentage:\n", missing_percent)



# Target variable: Survived
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Distribution of Survival")
plt.show()

# Passenger Class
sns.countplot(x="Pclass", data=df, palette="viridis")
plt.title("Passenger Class Distribution")
plt.show()

# Gender
sns.countplot(x="Sex", data=df, palette="pastel")
plt.title("Gender Distribution")
plt.show()

# Age Distribution
sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="blue")
plt.title("Age Distribution of Passengers")
plt.show()

# Fare Distribution (log scale due to skewness)
sns.histplot(df["Fare"], bins=40, kde=True, color="green")
plt.xscale("log")
plt.title("Fare Distribution (Log Scaled)")
plt.show()



# Survival by Gender
sns.countplot(x="Survived", hue="Sex", data=df, palette="coolwarm")
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
sns.countplot(x="Survived", hue="Pclass", data=df, palette="Set1")
plt.title("Survival by Passenger Class")
plt.show()

# Boxplot: Age vs Survival
sns.boxplot(x="Survived", y="Age", data=df, palette="muted")
plt.title("Age Distribution by Survival")
plt.show()

# Boxplot: Fare vs Survival
sns.boxplot(x="Survived", y="Fare", data=df, palette="Blues")
plt.title("Fare Distribution by Survival")
plt.show()

numeric_features = df.select_dtypes(include=[np.number])

corr = numeric_features.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot for selected features
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue="Survived", palette="husl")
plt.show()


# Gender-based survival rates
survival_by_gender = df.groupby("Sex")["Survived"].mean()
print("\nSurvival Rate by Gender:\n", survival_by_gender)

# Class-based survival rates
survival_by_class = df.groupby("Pclass")["Survived"].mean()
print("\nSurvival Rate by Passenger Class:\n", survival_by_class)

# Age groups
df["AgeGroup"] = pd.cut(df["Age"], bins=[0,12,18,35,60,80], 
                        labels=["Child","Teen","Young Adult","Adult","Senior"])
agegroup_survival = df.groupby("AgeGroup")["Survived"].mean()
print("\nSurvival Rate by Age Group:\n", agegroup_survival)



print("\n✅  Completed Successfully!")
