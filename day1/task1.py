# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler

# # Step 1: Load dataset
# df = pd.read_csv(r"C:\Users\hp\Desktop\dataset1\Titanic-Dataset.csv")

# print(df.head())
# print(df.info())

# # Step 2: Handle missing values
# df = df.fillna(df.mean(numeric_only=True))   # numeric -> mean
# if not df.mode().empty:
#     df = df.fillna(df.mode().iloc[0])        # categorical -> mode

# # Step 3: Encode categorical data
# df = pd.get_dummies(df, drop_first=True)

# # Step 4: Normalize features
# scaler = MinMaxScaler()
# df[df.columns] = scaler.fit_transform(df)

# # Step 5: Detect & remove outliers
# for col in df.select_dtypes(include=['float', 'int']).columns:
#     sns.boxplot(x=df[col])
#     plt.show()
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# print("✅ Data clean & ready for ML")


# =======================
# Step 1: Import Libraries
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
titanic =pd.read_csv(r"C:\Users\hp\Desktop\dataset1\Titanic-Dataset.csv")


# Basic Info
print(titanic.info())
print("\nMissing Values:\n", titanic.isnull().sum())
print("\nFirst 5 Rows:\n", titanic.head())


# =======================
# Step 2: Handle Missing Values
# =======================

# Age: Fill with Median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Cabin: Too many missing → Drop column
titanic.drop(columns=['Cabin'], inplace=True)

# Embarked: Fill with Mode
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Confirm missing values
print("\nMissing after cleaning:\n", titanic.isnull().sum())


# =======================
# Step 3: Encode Categorical Features
# =======================

# Sex: Label Encoding (male=0, female=1)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

# Embarked: One-Hot Encoding (drop_first to avoid dummy trap)
titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns (not useful for ML models)
titanic.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("\nAfter Encoding:\n", titanic.head())


# =======================
# Step 4: Feature Scaling
# =======================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Columns to scale
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']

titanic[num_cols] = scaler.fit_transform(titanic[num_cols])

print("\nAfter Scaling:\n", titanic.head())


# =======================
# Step 5: Outlier Detection & Removal
# =======================

# Visualize outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=titanic[['Age','Fare','SibSp','Parch']])
plt.title("Outlier Detection (Boxplots)")
plt.show()

# Remove outliers using IQR method (for demonstration on Fare)
Q1 = titanic['Fare'].quantile(0.25)
Q3 = titanic['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
titanic = titanic[~((titanic['Fare'] < (Q1 - 1.5 * IQR)) | (titanic['Fare'] > (Q3 + 1.5 * IQR)))]

print("\nShape after removing outliers:", titanic.shape)


titanic.to_csv("Titanic-Cleaned.csv", index=False)

print("done")