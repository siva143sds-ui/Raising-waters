import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/flood_data.csv")

# Display first rows
print(data.head())

# Dataset information
print(data.info())

# Statistical summary
print(data.describe())

# Check missing values
print(data.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# Flood distribution
sns.countplot(x="flood", data=data)
plt.title("Flood Prediction Distribution")
plt.show()
