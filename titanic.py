import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('train.csv')

print("Before cleaning:")
print(df.isnull().sum())

# Data Cleaning
# 1. Fill missing Age with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2. Fill missing Embarked with the most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Drop Cabin column because too many values are missing (687 out of 891!)
df.drop(columns=['Cabin'], inplace=True)

print("\nAfter cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

# Visualization 1 - Survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count (0=Died, 1=Survived)')
plt.tight_layout()
plt.show()

# Visualization 2 - Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.tight_layout()
plt.show()

# Visualization 3 - Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Passenger Class')
plt.tight_layout()
plt.show()

# Visualization 4 - Age distribution of survivors vs non survivors
plt.figure(figsize=(8, 4))
df[df['Survived']==1]['Age'].plot(kind='hist', alpha=0.5, label='Survived', color='green', bins=20)
df[df['Survived']==0]['Age'].plot(kind='hist', alpha=0.5, label='Died', color='red', bins=20)
plt.title('Age Distribution - Survived vs Died')
plt.xlabel('Age')
plt.legend()
plt.tight_layout()
plt.show()