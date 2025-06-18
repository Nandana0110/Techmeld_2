import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"c:\Users\AL SharQ\Downloads\intern\Titanic-Dataset.csv")
df_cleaned = df.copy()
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)
df_cleaned.drop(columns=['Cabin'], inplace=True)

age_survived = df_cleaned[df_cleaned['Survived'] == 1]['Age']
age_not_survived = df_cleaned[df_cleaned['Survived'] == 0]['Age']
t_stat, p_val_ttest = ttest_ind(age_survived, age_not_survived)

print("T-Test: Mean Age of Survivors vs. Non-Survivors")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val_ttest:.4f}")
print("Interpretation:", "Significant" if p_val_ttest < 0.05 else "Not significant")
print()
plt.figure(figsize=(7, 5))
sns.boxplot(x='Survived', y='Age', data=df_cleaned, palette='coolwarm')
plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

contingency_table = pd.crosstab(df_cleaned['Survived'], df_cleaned['Pclass'])
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Test: Survival vs. Passenger Class")
print(f"Chi-square statistic: {chi2_stat:.2f}")
print(f"P-value: {p_val_chi2:.4e}")
print("Interpretation:", "Associated" if p_val_chi2 < 0.05 else "Not associated")

plt.figure(figsize=(7, 5))
sns.countplot(x='Pclass', hue='Survived', data=df_cleaned, palette='Set2')
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()
