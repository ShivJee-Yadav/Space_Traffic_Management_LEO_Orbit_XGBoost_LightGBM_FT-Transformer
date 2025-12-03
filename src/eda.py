import os
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',200)
pd.set_option('display.width',120)

Data_Path = os.path.join('data','CZ_6A_Events_2024-08-06.xlsx')
Plot_Path = os.path.join('outputs','Plots')
os.makedirs(Plot_Path , exist_ok=True)
print(Data_Path)
print(Plot_Path)
df = pd.read_excel(Data_Path)
print("Shape of Dataframe :",  df.shape)
print("-" * 60)
print(df.info())
print("-" * 60)

duplicate_rows = df.duplicated().sum()
print("Duplicate Rows" , duplicate_rows)
print("# # " * 20)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
bool_cols = df.select_dtypes(include=[np.bool]).columns.tolist()
datetime64_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
Object_cols = df.select_dtypes(include=[np.object_]).columns.tolist()
print("Numerical COls = " , num_cols)
print("-" * 60)
print("Bool COls = " , bool_cols)
print("-" * 60)
print("datetime64 COls = " , datetime64_cols)
print("-" * 60)
print("Object COls = " , Object_cols)
print("-" * 60)


# Assure All columns are present in each set

total_Col_split = len(num_cols) + len(bool_cols) + len(datetime64_cols) + len(Object_cols)
if total_Col_split == df.shape[1]:
    print("All colums Are splitted successfully")
else:
    print("Missing Columns Split")


## -- Missing Values Analysis -- ##  

missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).sort_values(ascending = False)
print("Missing values count:\n" , missing)
print("Missing values percent:\n", missing_pct)


# # - - Time Feature Checks  - - # # 

df['creationTsOfCDM'] = pd.to_datetime(df['creationTsOfCDM'], errors='coerce', utc=True)
df['cdmTca'] = pd.to_datetime(df['cdmTca'], errors='coerce', utc=True)
df['hours_to_tca'] = (df['cdmTca'] - df['creationTsOfCDM']).dt.total_seconds() / 3600.0
print("\nhours_to_tca summary:")
print(df['hours_to_tca'].describe())



################################################################

import matplotlib.pyplot as plt
import seaborn as sns

###############################################################

# # - - Target Creation and Class Imblance - - # # 

df['HighRisk'] = ((df['cdmPc'] > 1e-6) & (df['cdmMissDistance'] < 2000)).astype(int)
print("\n HighRisk Distribution (counts) : ")
print(df['HighRisk'].value_counts())
plt.figure(figsize=(6,4))
sns.histplot(df['hours_to_tca'].dropna(), bins=50, kde=True)
plt.title('Hours to TCA Distribution')
plt.savefig(os.path.join(Plot_Path,'TCA Distribution.png'))
plt.show()


plt.figure(figsize=(4,3))
sns.countplot(x='HighRisk', data=df)
plt.title('HighRisk Class Distribution')
plt.savefig(os.path.join(Plot_Path,'Class_Distribution.png'))
plt.show()
# plt.close()


# # - - Correlations - - # # 
print(df.info())
corr = df[num_cols].corr(method='spearman')
print("\n SPearman Correlation :")
corrs = (
    corr.abs().unstack().dropna().sort_values(ascending=False)
)

plt.figure(figsize=(10,8))
sns.heatmap(corr , cmap='coolwarm' , center = 0, annot = True)
plt.title('Spearman Correlation Heatmap')
plt.savefig(os.path.join(Plot_Path,'Correlation Heatmap.png'))
plt.show()
