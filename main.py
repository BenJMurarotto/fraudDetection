import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as pplt


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("bank-transaction-dataset-for-fraud-detection/versions/4/bank_transactions_data_2.csv")

print("Shape of dataset", df.shape)
display(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
display(df.describe().T)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(numerical_columns)
print(non_numerical_columns)

def plot_histograms(df, numerical_columns):
    for column in numerical_columns:
        pplt.figure(figsize=(8, 5))
        sns.histplot(df[column], bins = 35, kde=True, color="orange")
        pplt.title(f'Distribution of {column}')
        pplt.xlabel(column)
        pplt.ylabel('Frequency')
        pplt.show()

def correlation_matrix(df, numerical_columns):
    matrix = df[numerical_columns].corr()
    pplt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    pplt.title('Correlation Matrix of Numerical Variables')
    pplt.show()



correlation_matrix(df, numerical_columns)
