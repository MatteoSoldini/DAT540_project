import numpy as np
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")

# Since the dataset file is saved in the Strict Open XML Spreadsheet format, we
# have to use `calamine` to read it
df = pd.read_excel("dataset/dataset-uci.xlsx", engine="calamine")

# Shape of dataset; data types; feature overview; missing values; duplicates; etc.
print(f"Dataset shape: {df.shape}")
print(f"Dataset datatypes:\n{df.dtypes}")

print(f"# null values for each feature:\n{df.isnull().sum()}")
print(f"# null values in total: {df.isnull().sum().sum()}")

print(f"# duplicated values in total: {df.duplicated().sum()}")