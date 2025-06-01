#!/usr/bin/env python
# coding: utf-8

# # Preprocess the DISPATCHINTERCONNECTOR DATA

# In[82]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
import zipfile
import shutil

def extract_nested_zip(zip_path, extract_to):
    """Recursively extract ZIP files until no ZIP found inside."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Check if there are any nested ZIP files inside extract_to
    nested_zips = []
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith('.zip'):
                nested_zips.append(os.path.join(root, f))

    # If nested ZIPs found, extract them recursively
    for nested_zip in nested_zips:
        nested_extract_dir = nested_zip.replace('.zip', '')
        os.makedirs(nested_extract_dir, exist_ok=True)
        extract_nested_zip(nested_zip, nested_extract_dir)
        os.remove(nested_zip)  # Remove nested zip after extraction

def find_csv_file(search_dir):
    """Find the first CSV file in the directory tree."""
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                return os.path.join(root, file)
    return None

# Step 1: Unzip top-level ZIP
top_zip = '/content/drive/MyDrive/DISPATCHINTERCONNECTORRES.zip'
top_extract_dir = 'DISPATCHINTERCONNECTORRES'

if not os.path.exists(top_extract_dir):
    print(f"📦 Extracting top-level ZIP: {top_zip}")
    with zipfile.ZipFile(top_zip, 'r') as zip_ref:
        zip_ref.extractall(top_extract_dir)

# Step 2: Handle nested folder
nested_dir = os.path.join(top_extract_dir, 'DISPATCHINTERCONNECTORRES')

# Step 3: Setup output folder
output_dir = '/content/dispatch connection'
os.makedirs(output_dir, exist_ok=True)

# Step 4: Month mapping
month_map = {
    'january': 'jan', 'february': 'feb', 'march': 'mar', 'april': 'apr',
    'may': 'may', 'june': 'jun', 'july': 'jul', 'august': 'aug',
    'september': 'sep', 'october': 'oct', 'november': 'nov', 'december': 'dec'
}

# Step 5: Walk through each year and month
for year in ['2022', '2023', '2024']:
    year_path = os.path.join(nested_dir, year)

    for month_full, month_short in month_map.items():
        zip_name = f"{month_full}.zip"
        zip_path = os.path.join(year_path, zip_name)

        if os.path.exists(zip_path):
            print(f"🔍 Extracting {zip_path}")
            temp_dir = 'temp_extract'
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            extract_nested_zip(zip_path, temp_dir)

            csv_file = find_csv_file(temp_dir)
            if csv_file:
                new_name = f"{month_short}_{year}.csv"
                dest = os.path.join(output_dir, new_name)
                shutil.move(csv_file, dest)
                print(f"✅ Saved: {new_name}")
            else:
                print(f"⚠️ No CSV found in {zip_name} ({year})")

            shutil.rmtree(temp_dir)

        else:
            print(f"❌ Missing ZIP file: {zip_path}")

print("\n🎉 DONE: Check the '/content/dispatch connection' folder for CSVs.")


# In[ ]:


import os
import pandas as pd

# Folder containing all CSVs
csv_folder = '/content/dispatch connection'

# Process each CSV file
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        print(f"🔄 Processing: {filename}")

        # Read CSV skipping the first row (index 0)
        df = pd.read_csv(file_path, skiprows=1)

        # Save back in-place
        df.to_csv(file_path, index=False)
        print(f"✅ Overwritten: {filename}")

print("\n🎉 All CSV files processed (first row skipped, saved in-place).")


# In[ ]:


import os
import pandas as pd

# Paths
input_folder = '/content/dispatch connection'
output_folder = '/content/interconnection'
os.makedirs(output_folder, exist_ok=True)

# Process each year
for year in ['2022', '2023', '2024']:
    df_list = []

    # Find all matching CSV files for the current year
    for file in os.listdir(input_folder):
        if file.endswith(f'{year}.csv'):
            file_path = os.path.join(input_folder, file)
            print(f"📄 Adding: {file}")
            df = pd.read_csv(file_path)
            df_list.append(df)

    # Concatenate and save
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        output_path = os.path.join(output_folder, f'interconnection_{year}.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
    else:
        print(f"⚠️ No CSVs found for {year}")

print("\n🎉 All done! Check the /content/interconnection folder.")


# In[ ]:


df2022= pd.read_csv(r'/content/interconnection/interconnection_2022.csv')


# In[ ]:


df2023 = pd.read_csv(r'/content/interconnection/interconnection_2023.csv')


# In[ ]:


df2024 = pd.read_csv(r'/content/interconnection/interconnection_2024.csv')


# In[ ]:


def missing_percentage(df, name):
    print(f"\n📊 Missing Percentage for {name}:")
    percent_missing = df.isnull().mean() * 100
    percent_missing = percent_missing[percent_missing > 0].sort_values(ascending=False)
    if percent_missing.empty:
        print("✅ No missing values.")
    else:
        print(percent_missing.round(2).astype(str) + '%')

# Show missing percentage
missing_percentage(df2022, "2022")



# In[ ]:


missing_percentage(df2023, "2023")


# In[ ]:


missing_percentage(df2024, "2024")


# In[ ]:


# Function to drop columns with >= 50% missing values
def drop_high_missing(df, name):
    threshold = 0.5  # 50%
    missing_fraction = df.isnull().mean()
    to_drop = missing_fraction[missing_fraction >= threshold].index
    print(f"🧹 Dropping {len(to_drop)} columns from {name} with >= 50% missing:")
    print(list(to_drop))
    df_cleaned = df.drop(columns=to_drop)
    return df_cleaned

# Apply to all dataframes
df2022_cleaned = drop_high_missing(df2022, "2022")
df2023_cleaned = drop_high_missing(df2023, "2023")
df2024_cleaned = drop_high_missing(df2024, "2024")


# In[ ]:


def fill_missing_values(df, name):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            mean_val = df[column].mean()
            df[column].fillna(mean_val, inplace=True)
        else:
            mode_val = df[column].mode(dropna=True)
            if not mode_val.empty:
                df[column].fillna(mode_val[0], inplace=True)
            else:
                df[column].fillna('Unknown', inplace=True)
    print(f"✅ Missing values filled for {name}")
    return df

# Apply to each cleaned DataFrame
df2022_filled = fill_missing_values(df2022_cleaned, "2022")
df2023_filled = fill_missing_values(df2023_cleaned, "2023")
df2024_filled = fill_missing_values(df2024_cleaned, "2024")


# # Check again

# In[ ]:


missing_percentage(df2022_filled, "2022")
missing_percentage(df2023_filled, "2023")
missing_percentage(df2024_filled, "2024")


# In[ ]:


# Find intersection of columns in all three DataFrames
common_columns = set(df2022_filled.columns) & set(df2023_filled.columns) & set(df2024_filled.columns)

# Convert to sorted list for better readability
common_columns = sorted(list(common_columns))

# Print the common columns
print("✅ Common columns in all three DataFrames (2022, 2023, 2024):")
for col in common_columns:
    print(f"- {col}")


# In[ ]:


# Define the common columns explicitly
common_columns = [
    '3', 'DISPATCH', 'DISPATCHINTERVAL', 'EXPORTGENCONID', 'EXPORTLIMIT',
    'FCASEXPORTLIMIT', 'FCASIMPORTLIMIT', 'I', 'IMPORTGENCONID', 'IMPORTLIMIT',
    'INTERCONNECTORID', 'INTERCONNECTORRES', 'INTERVENTION', 'LASTCHANGED',
    'MARGINALLOSS', 'MARGINALVALUE', 'METEREDMWFLOW', 'MWFLOW', 'MWLOSSES',
    'RUNNO', 'SETTLEMENTDATE', 'VIOLATIONDEGREE'
]

# Subset each DataFrame to the common columns
df2022_common = df2022_filled[common_columns]
df2023_common = df2023_filled[common_columns]
df2024_common = df2024_filled[common_columns]

# Concatenate all three DataFrames
merged_interconnection = pd.concat([df2022_common, df2023_common, df2024_common], ignore_index=True)

# Confirm the shape and preview
print(f"✅ merged_interconnection shape: {merged_interconnection.shape}")



# In[ ]:


merged_interconnection.head()


# In[ ]:


missing_percentage(merged_interconnection, "2022-2024")


# In[ ]:


merged_interconnection.shape


# # Now preprocess the DISPATCHREGIONS data
# 

# In[ ]:


import os
import zipfile
import shutil

def extract_nested_zip(zip_path, extract_to):
   """Recursively extract ZIP files until no ZIP found inside."""
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall(extract_to)

   # Check if there are any nested ZIP files inside extract_to
   nested_zips = []
   for root, dirs, files in os.walk(extract_to):
       for f in files:
           if f.lower().endswith('.zip'):
               nested_zips.append(os.path.join(root, f))

   # If nested ZIPs found, extract them recursively
   for nested_zip in nested_zips:
       nested_extract_dir = nested_zip.replace('.zip', '')
       os.makedirs(nested_extract_dir, exist_ok=True)
       extract_nested_zip(nested_zip, nested_extract_dir)
       os.remove(nested_zip)  # Remove nested zip after extraction

def find_csv_file(search_dir):
   """Find the first CSV file in the directory tree."""
   for root, dirs, files in os.walk(search_dir):
       for file in files:
           if file.lower().endswith('.csv'):
               return os.path.join(root, file)
   return None

# Step 1: Unzip top-level ZIP
top_zip = '/content/drive/MyDrive/DISPATCHREGIONS.zip'
top_extract_dir = 'DISPATCHREGIONS'

if not os.path.exists(top_extract_dir):
   print(f"📦 Extracting top-level ZIP: {top_zip}")
   with zipfile.ZipFile(top_zip, 'r') as zip_ref:
       zip_ref.extractall(top_extract_dir)

# Step 2: Handle nested folder
nested_dir = os.path.join(top_extract_dir, 'DISPATCHREGIONS')

# Step 3: Setup output folder
output_dir = '/content/dispatch region'
os.makedirs(output_dir, exist_ok=True)

# Step 4: Month mapping
month_map = {
   'january': 'jan', 'february': 'feb', 'march': 'mar', 'april': 'apr',
   'may': 'may', 'june': 'jun', 'july': 'jul', 'august': 'aug',
   'september': 'sep', 'october': 'oct', 'november': 'nov', 'december': 'dec'
}

# Step 5: Walk through each year and month
for year in ['2022', '2023', '2024']:
   year_path = os.path.join(nested_dir, year)

   for month_full, month_short in month_map.items():
       zip_name = f"{month_full}.zip"
       zip_path = os.path.join(year_path, zip_name)

       if os.path.exists(zip_path):
           print(f"🔍 Extracting {zip_path}")
           temp_dir = 'temp_extract'
           if os.path.exists(temp_dir):
               shutil.rmtree(temp_dir)
           os.makedirs(temp_dir, exist_ok=True)

           extract_nested_zip(zip_path, temp_dir)

           csv_file = find_csv_file(temp_dir)
           if csv_file:
               new_name = f"{month_short}_{year}.csv"
               dest = os.path.join(output_dir, new_name)
               shutil.move(csv_file, dest)
               print(f"✅ Saved: {new_name}")
           else:
               print(f"⚠️ No CSV found in {zip_name} ({year})")

           shutil.rmtree(temp_dir)

       else:
           print(f"❌ Missing ZIP file: {zip_path}")

print("\n🎉 DONE: Check the '/content/dispatch region' folder for CSVs.")


# In[ ]:


import os
import pandas as pd

# Folder containing all CSVs
csv_folder = '/content/dispatch region'

# Process each CSV file
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        print(f"🔄 Processing: {filename}")

        # Read CSV skipping the first row (index 0)
        df = pd.read_csv(file_path, skiprows=1)

        # Save back in-place
        df.to_csv(file_path, index=False)
        print(f"✅ Overwritten: {filename}")

print("\n🎉 All CSV files processed (first row skipped, saved in-place).")


# In[ ]:


import os
import pandas as pd

# Paths
input_folder = '/content/dispatch region'
output_folder = '/content/region'
os.makedirs(output_folder, exist_ok=True)

# Process each year
for year in ['2022', '2023', '2024']:
    df_list = []

    # Find all matching CSV files for the current year
    for file in os.listdir(input_folder):
        if file.endswith(f'{year}.csv'):
            file_path = os.path.join(input_folder, file)
            print(f"📄 Adding: {file}")
            df = pd.read_csv(file_path)
            df_list.append(df)

    # Concatenate and save
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        output_path = os.path.join(output_folder, f'region_{year}.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
    else:
        print(f"⚠️ No CSVs found for {year}")

print("\n🎉 All done! Check the /content/ region folder.")


# In[ ]:


reg2022= pd.read_csv(r'/content/region/region_2022.csv')


# In[ ]:


reg2023= pd.read_csv(r'/content/region/region_2023.csv')


# In[ ]:


reg2024= pd.read_csv(r'/content/region/region_2024.csv')


# ##### Show missing percentage

# In[ ]:


missing_percentage(reg2022, "2022")


# In[ ]:


missing_percentage(reg2023, "2023")


# In[ ]:


missing_percentage(reg2024, "2024")


# ##### Drop columns

# In[ ]:


reg2022_cleaned = drop_high_missing(reg2022, "2022")
reg2023_cleaned = drop_high_missing(reg2023, "2023")
reg2024_cleaned = drop_high_missing(reg2024, "2024")


# In[ ]:


##### Fill values


# In[ ]:


reg2022_filled = fill_missing_values(reg2022_cleaned, "2022")
reg2023_filled = fill_missing_values(reg2023_cleaned, "2023")
reg2024_filled = fill_missing_values(reg2024_cleaned, "2024")


# ##### Check Again missing percentage

# In[ ]:


missing_percentage(reg2022_filled, "2022")


# In[ ]:


missing_percentage(reg2023_filled, "2023")


# In[ ]:


missing_percentage(reg2024_filled, "2024")


# In[ ]:


reg2022_filled.shape


# In[ ]:


reg2023_filled.shape


# In[ ]:


reg2024_filled.shape


# In[ ]:


# Find intersection of columns in all three DataFrames
common_columns_reg = set(reg2022_filled.columns) & set(reg2023_filled.columns) & set(reg2024_filled.columns)

# Convert to sorted list for better readability
common_columns_reg = sorted(list(common_columns_reg))

# Print the common columns
print("✅ Common columns in all three Region DataFrames (2022, 2023, 2024):")
for col in common_columns_reg:
    print(f"- {col}")


# In[ ]:


# Subset each DataFrame to the common columns
reg2022_common = reg2022_filled[common_columns_reg]
reg2023_common = reg2023_filled[common_columns_reg]
reg2024_common = reg2024_filled[common_columns_reg]

# Concatenate all three DataFrames
merged_region = pd.concat([reg2022_common, reg2023_common, reg2024_common], ignore_index=True)

# Confirm the shape and preview
print(f"✅ merged_region shape: {merged_region.shape}")


# In[ ]:


from google.colab import files
# Save the DataFrame to a CSV file
merged_region.to_csv('/content/merged_region.csv', index=False)

merged_interconnection.to_csv('/content/merged_interconnection.csv', index=False)

files.download('/content/merged_region.csv')
files.download('/content/merged_interconnection.csv')


# ## Read the both csv files

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


region= pd.read_csv(r'/content/drive/MyDrive/merged_region.csv')


# In[3]:


interconnection= pd.read_csv(r'/content/drive/MyDrive/merged_interconnection.csv')


# In[4]:


region.head()


# In[5]:


region["REGIONID"].value_counts()


# In[6]:


interconnection['INTERCONNECTORID'].value_counts()


# In[7]:


region.columns


# In[8]:


interconnection.columns


# In[9]:


# === Step 1: Filter `region` DataFrame for QLD1, NSW1, VIC1 ===
target_regions = ['QLD1', 'NSW1', 'VIC1']
filtered_region = region[region['REGIONID'].isin(target_regions)]

# === Step 2: Filter `interconnection` DataFrame for NSW1-QLD1 and VIC1-NSW1 ===
target_interconnectors = ['NSW1-QLD1', 'VIC1-NSW1']
filtered_interconnection = interconnection[interconnection['INTERCONNECTORID'].isin(target_interconnectors)]


# In[10]:


filtered_region.shape


# In[11]:


filtered_interconnection.shape


# # Preprocess Dates

# In[16]:


# Convert settlement date columns to datetime
filtered_region['SETTLEMENTDATE'] = pd.to_datetime(filtered_region['SETTLEMENTDATE'])
filtered_interconnection['SETTLEMENTDATE'] = pd.to_datetime(filtered_interconnection['SETTLEMENTDATE'])

# Extract year for filtering
filtered_region['YEAR'] = filtered_region['SETTLEMENTDATE'].dt.year
filtered_interconnection['YEAR'] = filtered_interconnection['SETTLEMENTDATE'].dt.year


# ### Merge the data

# In[17]:


merged_df = pd.merge(
    filtered_region,
    filtered_interconnection,
    on='SETTLEMENTDATE',
    how='inner',
    suffixes=('_region', '_interconnect')
)


# In[18]:


merged_df


# In[19]:


merged_df.columns


# In[20]:


merged_df.isnull().sum()


# In[21]:


merged_df.describe()


# In[22]:


merged_df.columns


# # Now create a new column for supply demand imbalance

# In[23]:


merged_df['SUPPLY_DEMAND_IMBALANCE'] = merged_df['CLEAREDSUPPLY'] - merged_df['TOTALDEMAND']


# In[24]:


merged_df.columns


# In[56]:


# Randomly sample 90% of the dataset
half_df = merged_df.sample(frac=0.9, random_state=42)


# In[57]:


half_df.shape


# In[27]:


# Now take the most correlated columns


# In[28]:


numeric_df = merged_df.select_dtypes(include=['float64', 'int64'])


# In[29]:


numeric_df


# In[30]:


corr_matrix = numeric_df.corr()


# In[31]:


imbalance_corr = corr_matrix['SUPPLY_DEMAND_IMBALANCE'].drop('SUPPLY_DEMAND_IMBALANCE')


# In[32]:


top_corr_features = imbalance_corr.abs().sort_values(ascending=False)
print(top_corr_features.head(20))  # Top 20 most correlated features


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

top_features = top_corr_features.head(10).index.tolist() + ['SUPPLY_DEMAND_IMBALANCE']
sns.heatmap(numeric_df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top Correlations with Supply-Demand Imbalance")
plt.show()


# In[34]:


# Define your top 20 important columns
top_20_columns = [
    'AVAILABLEGENERATION', 'CLEAREDSUPPLY', 'TOTALDEMAND', 'DEMANDFORECAST',
    'EXCESSGENERATION', 'DISPATCHABLEGENERATION', 'DISPATCHABLELOAD',
    'INITIALSUPPLY', 'AVAILABLELOAD', 'AGGREGATEDISPATCHERROR',
    'SEMISCHEDULE_CLEAREDMW', 'SEMISCHEDULE_COMPLIANCEMW', 'UIGF',
    'SS_SOLAR_CLEAREDMW', 'SS_SOLAR_COMPLIANCEMW', 'SS_SOLAR_UIGF',
    'SS_WIND_CLEAREDMW', 'SS_WIND_COMPLIANCEMW', 'SS_WIND_UIGF',
    'SUPPLY_DEMAND_IMBALANCE'
]

# Step 1: Identify all float64 and int64 columns in merged_df
numeric_cols = half_df.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Identify which numeric columns to drop (not in top 20 list)
cols_to_drop = [col for col in numeric_cols if col not in top_20_columns]

# Step 3: Drop the unwanted numeric columns
merged_df_cleaned = half_df.drop(columns=cols_to_drop)


# In[35]:


merged_df_cleaned


# In[36]:


merged_df_cleaned


# In[37]:


merged_df_cleaned.info()


# # Data Visualisation

# In[38]:


import plotly.express as px

# Filter data for 2022-2024 and regions Queensland (QLD1), Victoria (VIC1), New South Wales (NSW1)
filtered_df = merged_df_cleaned[
    (merged_df_cleaned['SETTLEMENTDATE'].dt.year >= 2022) &
    (merged_df_cleaned['SETTLEMENTDATE'].dt.year <= 2024) &
    (merged_df_cleaned['REGIONID'].isin(['QLD1', 'VIC1', 'NSW1']))
]


# In[39]:


filtered_df


# In[40]:


px.scatter(filtered_df, x='AVAILABLEGENERATION', y='SUPPLY_DEMAND_IMBALANCE',
           color='REGIONID', title='Available Generation vs Supply-Demand Imbalance')


# # Line Plot (average monthly trend)

# In[41]:


monthly_avg = filtered_df.groupby([filtered_df['SETTLEMENTDATE'].dt.to_period('M'), 'REGIONID'])['SUPPLY_DEMAND_IMBALANCE'].mean().reset_index()
monthly_avg['SETTLEMENTDATE'] = monthly_avg['SETTLEMENTDATE'].dt.to_timestamp()

px.line(monthly_avg, x='SETTLEMENTDATE', y='SUPPLY_DEMAND_IMBALANCE', color='REGIONID',
        title='Monthly Average Supply-Demand Imbalance (2022-2024)')


# # Box Plot

# In[42]:


px.box(filtered_df, x='REGIONID', y='TOTALDEMAND', color='REGIONID',
       title='Total Demand Distribution by Region')


# # Violin Plot

# In[43]:


px.violin(filtered_df, y='AGGREGATEDISPATCHERROR', color='REGIONID', box=True, points='all',
          title='Aggregated Dispatch Error Distribution by Region')


# # Histogram

# In[44]:


px.histogram(filtered_df, x='DEMANDFORECAST', color='REGIONID', nbins=50,
             title='Distribution of Demand Forecast by Region', barmode='overlay')


# # Bar Plot (average per region)

# In[45]:


avg_by_region = filtered_df.groupby('REGIONID')['EXCESSGENERATION'].mean().reset_index()

px.bar(avg_by_region, x='REGIONID', y='EXCESSGENERATION', color='REGIONID',
       title='Average Excess Generation by Region')


# # Scatter with Marginal Histograms

# In[46]:


px.scatter(filtered_df, x='AVAILABLELOAD', y='SUPPLY_DEMAND_IMBALANCE', color='REGIONID',
           marginal_x='histogram', marginal_y='histogram',
           title='Available Load vs Supply-Demand Imbalance with Marginals')


# # Density Contour Plot

# In[47]:


px.density_contour(filtered_df, x='SS_SOLAR_CLEAREDMW', y='SUPPLY_DEMAND_IMBALANCE', color='REGIONID',
                   title='Density Contour: Solar Cleared MW vs Supply-Demand Imbalance')


# In[50]:


yearly_avg = filtered_df.groupby(['YEAR_region', 'REGIONID'])['SUPPLY_DEMAND_IMBALANCE'].mean().reset_index()

px.line(yearly_avg, x='YEAR_region', y='SUPPLY_DEMAND_IMBALANCE', color='REGIONID',
        title='Yearly Average Supply-Demand Imbalance')


# In[51]:


px.scatter(filtered_df, x='INITIALSUPPLY', y='SUPPLY_DEMAND_IMBALANCE', color='REGIONID',
           size='TOTALDEMAND', title='Initial Supply vs Supply-Demand Imbalance (size=Total Demand)')


# In[53]:


import seaborn as sns
import matplotlib.pyplot as plt

cols = ['AVAILABLEGENERATION', 'CLEAREDSUPPLY', 'TOTALDEMAND', 'DEMANDFORECAST',
        'EXCESSGENERATION', 'DISPATCHABLEGENERATION', 'DISPATCHABLELOAD',
        'INITIALSUPPLY', 'AVAILABLELOAD', 'AGGREGATEDISPATCHERROR',
        'SEMISCHEDULE_CLEAREDMW', 'SEMISCHEDULE_COMPLIANCEMW', 'UIGF',
        'SS_SOLAR_CLEAREDMW', 'SS_SOLAR_COMPLIANCEMW', 'SS_SOLAR_UIGF',
        'SS_WIND_CLEAREDMW', 'SS_WIND_COMPLIANCEMW', 'SS_WIND_UIGF',
        'SUPPLY_DEMAND_IMBALANCE']

df_corr = filtered_df[cols + ['REGIONID']]

for region in df_corr['REGIONID'].unique():
    plt.figure(figsize=(10, 8))
    region_df = df_corr[df_corr['REGIONID'] == region]

    # Select only numeric columns for correlation calculation
    numeric_df = region_df.select_dtypes(include=['float64', 'int32', 'int64'])

    corr = numeric_df.corr()
    sns.heatmap(corr[['SUPPLY_DEMAND_IMBALANCE']].sort_values(by='SUPPLY_DEMAND_IMBALANCE', ascending=False),
                annot=True, cmap='coolwarm')
    plt.title(f'Correlation with Supply-Demand Imbalance in {region}')
    plt.show()


# In[54]:


import plotly.express as px

variables = ['AVAILABLEGENERATION', 'CLEAREDSUPPLY', 'TOTALDEMAND', 'DEMANDFORECAST', 'SUPPLY_DEMAND_IMBALANCE']

fig = px.scatter_matrix(filtered_df,
                        dimensions=variables,
                        color='REGIONID',
                        title='Scatter Matrix: Key Variables vs Supply-Demand Imbalance')
fig.show()


# In[55]:





# In[ ]:




