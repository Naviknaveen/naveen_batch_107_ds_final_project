import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("final_project_dataset.csv")
print(df.head())
print(df.columns)

columns_to_keep = ['Service Type','Creation Date','Vehicle Sales Date','Vehicle Model',
                'Mileage', 'Engine Hours','Bill to','Part Amount',
       'Labour Amount','Concern Code' ]
print(df[columns_to_keep].info())

# Drop columns not in the list of columns to keep
df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])
print(df.head())

filter_cat_col =df.select_dtypes(include="object")
print("cat_col=",filter_cat_col.columns)

filter_num_col =df.select_dtypes(exclude="object")
print("num_col=",filter_cat_col.columns)

cat_col = ['Service Type', 'Vehicle Model','Bill to', 'Concern Code']

num_col = ['Mileage', 'Engine Hours', 'Part Amount', 'Labour Amount']

date_col = ["Creation Date","Vehicle Sales Date"]

print(df.isnull().sum())

filter_nan_col = df.columns[df.isnull().any()].tolist()
print(filter_nan_col)

nan_col = ['Service Type', 'Vehicle Sales Date', 'Engine Hours', 'Concern Code']

df_cleaned = df.dropna(subset=['Service Type', 'Vehicle Sales Date', 'Engine Hours', 'Concern Code'],inplace=True)
print(df_cleaned)

print(df.isnull().sum())
