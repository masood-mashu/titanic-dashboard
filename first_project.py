import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Name': ['Mashu', 'Rahul', 'Priya', 'Arjun', 'Sneha', 
             'Kiran', 'Divya', 'Rohan', 'Ananya', 'Vikram'],
    'Age': [20, 21, 20, 22, 21, 20, 21, 22, 20, 21],
    'Maths': [85, 72, 90, 65, 88, 76, 95, 60, 82, 78],
    'Science': [78, 85, 92, 70, 75, 88, 91, 65, 79, 83],
    'English': [88, 65, 80, 75, 90, 70, 85, 72, 91, 68]
}

df = pd.DataFrame(data)

# Plot 1 - Bar chart of average marks
plt.figure(figsize=(8, 5))
df[['Maths', 'Science', 'English']].mean().plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Average Marks Per Subject')
plt.ylabel('Marks')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot 2 - Heatmap of marks
plt.figure(figsize=(8, 5))
sns.heatmap(df[['Maths', 'Science', 'English']], annot=True, cmap='coolwarm')
plt.title('Student Marks Heatmap')
plt.tight_layout()
plt.show()

# Plot 3 - Histogram of Maths marks
plt.figure(figsize=(8, 5))
plt.hist(df['Maths'], bins=5, color='purple', edgecolor='black')
plt.title('Distribution of Maths Marks')
plt.xlabel('Marks')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Save the DataFrame to a CSV file
df.to_csv('students.csv', index=False)
print("Data saved to students.csv!")

# Read it back
df2 = pd.read_csv('students.csv')
print("\nData read back from CSV:")
print(df2)